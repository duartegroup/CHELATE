"""Identity and fixed-denominator rules for fair method comparisons.

The comparison cohort depends only on input and label availability, never on
RMSD success thresholds, predictions or selected operating cutoffs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

APO_SUFFIX = r"__[a-zA-Z0-9]{4}(?:_[a-zA-Z0-9]+)+$"
TARGET = "close_rmsd_3.5_hetero"
METHODS = ("VINA", "SMINA", "GNINA")
SCENARIOS = ("redocked_FH", "apo_docked_FH")
REPEATS = (1, 2, 3)


def canonical_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map external apo identifiers onto their original curated holo pairs."""
    system_column = "alt_system_id" if "alt_system_id" in frame else "system_id"
    systems = frame[system_column].astype(str).str.replace(APO_SUFFIX, "", regex=True)
    return pd.DataFrame(
        {"system_id": systems, "ligand_id": frame["ligand_id"].astype(str)},
        index=frame.index,
    )


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(canonical_pair_columns(frame).itertuples(index=False, name=None))


def filter_to_cohort(frame: pd.DataFrame, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    keys = canonical_pair_columns(frame).itertuples(index=False, name=None)
    mask = np.fromiter((key in pairs for key in keys), dtype=bool, count=len(frame))
    return frame.loc[mask].copy()


def eligible_poses(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """Require finite native scores and physically valid RMSD labels only."""
    native = "CNNscore" if method == "GNINA" else "docking_score"
    mask = np.ones(len(frame), dtype=bool)
    for name in [native, TARGET, "symm_rmsd"]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
        if name != native:
            mask &= values >= 0
    return frame.loc[mask].copy()


def common_evaluable_pairs(frames: dict) -> set[tuple[str, str]]:
    """Use one intersection across engines, scenarios and all three repeats."""
    expected = {
        (method, scenario, repeat)
        for method in METHODS
        for scenario in SCENARIOS
        for repeat in REPEATS
    }
    if set(frames) != expected:
        raise ValueError(
            "The common cohort requires all 18 engine/scenario/repeat inputs"
        )
    populations = [
        pair_set(eligible_poses(frames[key], key[0])) for key in sorted(expected)
    ]
    return set.intersection(*populations)


def read_cohort(path: str | Path) -> set[tuple[str, str]]:
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    pairs = pair_set(frame)
    if not pairs:
        raise ValueError("The comparison cohort is empty")
    return pairs


def fixed_cohort_success(
    frame: pd.DataFrame,
    pairs: set[tuple[str, str]],
    metric: str,
    threshold: float,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[int, float]:
    """Missing outputs and invalid RMSD labels count as failures, not successes."""
    if not pairs:
        raise ValueError("The success-rate denominator must not be empty")
    frame = filter_to_cohort(frame, pairs)
    labels = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    ranks = pd.to_numeric(frame["pose"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(labels) & (labels >= 0) & (labels <= threshold)
    return {
        k: len(pair_set(frame.loc[good & (ranks >= 1) & (ranks <= k)])) / len(pairs)
        for k in top_ks
    }


def complete_native_cohort(frame, pairs, method):
    """Refuse a nominal native denominator unsupported by the available inputs."""
    selected = filter_to_cohort(eligible_poses(frame, method), pairs)
    if pair_set(selected) != pairs:
        raise ValueError("Native inputs do not cover the declared common cohort")
    return selected


def main():
    """Freeze common comparison IDs from feature availability, without models."""
    import argparse
    import hashlib
    import io
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--receptor-cohorts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "The comparison cohort is already frozen: " + str(args.output)
        )
    frames, inputs, populations = {}, {}, []
    for method in METHODS:
        receptors = pd.read_csv(
            args.receptor_cohorts_dir / f"{method}_test.csv",
            keep_default_na=False,
            dtype=str,
        )
        for scenario in SCENARIOS:
            permitted = pair_set(
                receptors.loc[
                    (receptors.included == "True")
                    & (receptors.docking_type == scenario)
                ]
            )
            for repeat in REPEATS:
                path = (
                    args.features_dir
                    / f"features_test_{method}_{scenario}_id={repeat}.csv"
                )
                content = path.read_bytes()
                columns = [
                    "system_id",
                    "ligand_id",
                    "pose",
                    TARGET,
                    "symm_rmsd",
                    "CNNscore" if method == "GNINA" else "docking_score",
                ]
                frame = pd.read_csv(
                    io.BytesIO(content),
                    usecols=columns,
                    keep_default_na=False,
                    na_values=[""],
                )
                if frame.duplicated(["system_id", "ligand_id", "pose"]).any():
                    raise ValueError("Duplicate feature pose identifiers: " + str(path))
                if not pair_set(frame) <= permitted:
                    raise ValueError(
                        "Feature pairs lie outside the frozen receptor cohort"
                    )
                frames[(method, scenario, repeat)] = frame
                inputs[str(path)] = hashlib.sha256(content).hexdigest()
                populations.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "repeat": repeat,
                        "eligible_pairs": len(pair_set(eligible_poses(frame, method))),
                    }
                )
    pairs = common_evaluable_pairs(frames)
    if not pairs:
        raise ValueError("No common evaluable pairs; do not invent a substitute cohort")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(pairs), columns=["system_id", "ligand_id"]).to_csv(
        args.output, index=False
    )
    record = {
        "created": datetime.now().astimezone().isoformat(),
        "rule": __doc__.strip(),
        "intersection": "All engines, both scenarios and all three repeats; valid native score and RMSD labels, without success thresholds or predictions",
        "n_pairs": len(pairs),
        "input_sha256": inputs,
        "input_populations": populations,
        "cohort_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_outputs_read": False,
        "prediction_cutoffs_read": False,
        "protocol_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "receptor_cohort_sha256": {
            method: hashlib.sha256(
                (args.receptor_cohorts_dir / f"{method}_test.csv").read_bytes()
            ).hexdigest()
            for method in METHODS
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cohort": str(args.output),
                "n_pairs": len(pairs),
                "model_outputs_read": False,
            }
        )
    )


if __name__ == "__main__":
    main()
