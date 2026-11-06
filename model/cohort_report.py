"""Report common-cohort and complete-curated-cohort performance after cutoff commitment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.cohorts import METHODS, read_cohort


def _count(rate, denominator, name):
    value = float(rate) * int(denominator)
    if not np.isfinite(value) or not np.isclose(value, round(value), atol=1e-6, rtol=0):
        raise ValueError(f"{name} does not reconstruct an integer success count")
    return round(value)


def summarise_native(native):
    """Mean over repeats, with the number of repeats behind each row.

    ``test_eval`` skips a repeat at cutoffs that retain fewer than five pairs, so
    low-cutoff rows can average fewer repeats than the others.
    """
    keys = ["cohort", "method", "dock_key", "pred_cutoff"]
    summary = (
        native.groupby(keys, as_index=False)
        .mean(numeric_only=True)
        .drop(columns="rep_id")
    )
    counts = native.groupby(keys).size().rename("n_repeats").reset_index()
    return summary.merge(counts, on=keys, how="left", validate="one_to_one")


def native_record(row, method, cohort, denominator):
    """Reconstruct per-repeat numerators before changing a declared denominator."""
    label = str(row["dock_type"]).strip().lower().replace("-", "")
    scenarios = {"redocked": "redocked", "crossdocked": "crossdocked"}
    if label not in scenarios:
        raise ValueError(f"Unknown docking scenario: {row['dock_type']}")
    available = int(row["total_systems"])
    retained = int(row["systems_count"])
    if not 0 <= retained <= available <= denominator:
        raise ValueError("Invalid retained, available or declared cohort counts")
    native_mbg = _count(row["baseline_MBG_sr_top1_RMSD=1"], available, "Native MBG")
    chelate_mbg = _count(
        row["MBG_sr_all_systems_top1_RMSD=1"], available, "CHELATE MBG"
    )
    conditional_mbg = _count(row["MBG_sr_top1_RMSD=1"], retained, "Retained MBG")
    if chelate_mbg != conditional_mbg:
        raise ValueError("Conditional and all-system MBG numerators disagree")
    native_whole = _count(
        row["baseline_sr_top1_RMSD=2"], available, "Native whole molecule"
    )
    chelate_whole = _count(row["sr_top1_RMSD=2"], retained, "CHELATE whole molecule")
    return {
        "cohort": cohort,
        "method": method,
        "dock_key": scenarios[label],
        "rep_id": int(row["rep_id"]),
        "pred_cutoff": float(row["rmsd_cutoff"]),
        "denominator": denominator,
        "eligible_pairs": available,
        "retained_pairs": retained,
        "coverage": retained / denominator,
        "native_mbg_successes": native_mbg,
        "chelate_mbg_successes": chelate_mbg,
        "native_whole_successes": native_whole,
        "chelate_whole_successes": chelate_whole,
        "native_mbg_sr": native_mbg / denominator,
        "chelate_mbg_sr": chelate_mbg / denominator,
        "chelate_mbg_retained_sr": chelate_mbg / retained,
        "native_whole_sr": native_whole / denominator,
        "chelate_whole_sr": chelate_whole / denominator,
        "chelate_whole_retained_sr": chelate_whole / retained,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-evaluation", type=Path, required=True)
    parser.add_argument("--eligible-evaluation", type=Path, required=True)
    parser.add_argument("--common-cohort", type=Path, required=True)
    parser.add_argument("--curated-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing cohort report")
    common = read_cohort(args.common_cohort)
    curated = read_cohort(args.curated_cohort)
    if not common <= curated or len(curated) != 156:
        raise ValueError("Unexpected frozen common or curated test identities")
    records, inputs = [], {}
    for folder, cohort in [
        (args.common_evaluation, "common_evaluable"),
        (args.eligible_evaluation, "engine_evaluable"),
    ]:
        for method in METHODS:
            path = folder / f"catboost_evaluation_results_{method}.csv"
            content = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(content).hexdigest()
            for row in pd.read_csv(path).to_dict("records"):
                available = int(row["total_systems"])
                if cohort == "common_evaluable" and available != len(common):
                    raise ValueError(
                        "Common evaluation does not cover its declared input cohort"
                    )
                records.append(
                    native_record(
                        row,
                        method,
                        cohort,
                        len(common) if cohort == "common_evaluable" else available,
                    )
                )
                if cohort == "engine_evaluable":
                    records.append(
                        native_record(row, method, "curated_156", len(curated))
                    )
    native = pd.DataFrame(records)
    if native.duplicated(
        ["cohort", "method", "dock_key", "rep_id", "pred_cutoff"]
    ).any():
        raise ValueError("Duplicate repeat/cutoff metrics")
    external = []
    from model.test_eval import load_external_baseline_stats

    for cohort, pairs in [("curated_156", curated), ("common_evaluable", common)]:
        stats = load_external_baseline_stats([1.0, 2.0], cohort_pairs=pairs)
        for scenario, methods in stats.items():
            for method, result in methods.items():
                for family, threshold in [("mbg", 1.0), ("rmsd", 2.0)]:
                    for top_k, (mean, std) in result[family][threshold].items():
                        external.append(
                            {
                                "cohort": cohort,
                                "dock_key": scenario,
                                "method": method,
                                "metric_family": family,
                                "threshold": threshold,
                                "top_k": top_k,
                                "denominator": len(pairs),
                                "mean": mean,
                                "std": std,
                                "available_outputs_mean": result["available_outputs"][
                                    0
                                ],
                            }
                        )
    args.output.mkdir(parents=True)
    native.to_csv(args.output / "cohort_native_per_run.csv", index=False)
    summarise_native(native).to_csv(
        args.output / "cohort_native_summary.csv", index=False
    )
    pd.DataFrame(external).to_csv(
        args.output / "cohort_external_summary.csv", index=False
    )
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "input_sha256": inputs,
                "common_cohort_sha256": hashlib.sha256(
                    args.common_cohort.read_bytes()
                ).hexdigest(),
                "curated_cohort_sha256": hashlib.sha256(
                    args.curated_cohort.read_bytes()
                ).hexdigest(),
                "common_pairs": len(common),
                "curated_pairs": len(curated),
                "mbg_success_threshold_A": 1.0,
                "whole_molecule_success_threshold_A": 2.0,
                "missing_rejected_or_unlabelled_pairs": "Failures in the declared overall denominator; retained success uses retained pairs only",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
