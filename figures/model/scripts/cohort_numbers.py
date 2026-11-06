"""Render the cohort tables (NUMBERS.md section 7) from the cohort report CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd


def render_cohorts(folder, cutoffs):
    folder = Path(folder)
    native_path = folder / "cohort_native_summary.csv"
    external_path = folder / "cohort_external_summary.csv"
    if not native_path.exists() and not external_path.exists():
        return ""
    native = pd.read_csv(native_path)
    external = pd.read_csv(external_path)
    out = [
        "## 7. Cohorts: top-1 MBG ≤1 Å and whole-molecule RMSD ≤2 Å",
        "",
        "Means over three repeats. Overall success uses the declared denominator; unavailable or rejected pairs count as failures. Retained success uses retained pairs. The common cohort is fixed by input/label availability across all engines, scenarios and repeats. Engine-evaluable populations can differ. The curated population contains all 156 pairs.",
        "",
        "| Cohort | Dock | Method | Prediction cutoff (Å) | Denominator | Eligible | Retained | Coverage (%) | Native MBG (%) | CHELATE MBG overall (%) | CHELATE MBG retained (%) | Native whole (%) | CHELATE whole overall (%) | CHELATE whole retained (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = [
        "coverage",
        "native_mbg_sr",
        "chelate_mbg_sr",
        "chelate_mbg_retained_sr",
        "native_whole_sr",
        "chelate_whole_sr",
        "chelate_whole_retained_sr",
    ]
    for cohort in ["common_evaluable", "engine_evaluable", "curated_156"]:
        for dock in ["redocked", "crossdocked"]:
            for method in ["VINA", "SMINA", "GNINA"]:
                for cutoff in [float(cutoffs[method])]:
                    rows = native.loc[
                        (native.cohort == cohort)
                        & (native.dock_key == dock)
                        & (native.method == method)
                        & np.isclose(native.pred_cutoff, cutoff)
                    ]
                    if len(rows) != 1:
                        raise ValueError(
                            "Missing or duplicate cohort/cutoff summary row"
                        )
                    row = rows.iloc[0]
                    if "n_repeats" in row and row["n_repeats"] != 3:
                        raise ValueError(
                            f"{cohort} {dock} {method} {cutoff}: "
                            f"{row['n_repeats']} repeats, expected 3"
                        )
                    counts = [
                        f"{row[name]:.2f}"
                        for name in ["denominator", "eligible_pairs", "retained_pairs"]
                    ]
                    rates = [f"{100 * row[name]:.2f}" for name in metrics]
                    out.append(
                        f"| {cohort} | {dock} | {method} | {cutoff:.2f} | "
                        + " | ".join(counts + rates)
                        + " |"
                    )
    out.extend(
        [
            "",
            "External baselines on both declared populations; missing outputs and invalid labels count as failures.",
            "",
            "| Cohort | Dock | Method | Denominator | Available outputs | MBG ≤1 Å (%) | Whole RMSD ≤2 Å (%) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (cohort, dock, method), group in external.loc[external.top_k == 1].groupby(
        ["cohort", "dock_key", "method"], sort=True
    ):
        mbg = group.loc[(group.metric_family == "mbg") & (group.threshold == 1.0)]
        whole = group.loc[(group.metric_family == "rmsd") & (group.threshold == 2.0)]
        if len(mbg) != 1 or len(whole) != 1:
            raise ValueError("Missing or duplicate external-cohort metric")
        row = mbg.iloc[0]
        out.append(
            f"| {cohort} | {dock} | {method} | {row.denominator:.0f} | {row.available_outputs_mean:.2f} | {100 * row['mean']:.2f} | {100 * whole.iloc[0]['mean']:.2f} |"
        )
    return "\n".join(out)
