"""Render the training-fraction results without manually copying paper numbers."""

from pathlib import Path

import numpy as np
import pandas as pd


def render_sweep(directory: str | Path, cutoffs: dict[str, float]) -> str:
    """Read the unique sweep and reject stale operating points before reporting it."""
    paths = sorted(Path(directory).glob("mbg_sr_vs_training_fraction_predCutoff=*.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one current training-fraction sweep; found {len(paths)}"
        )
    frame = pd.read_csv(paths[0])
    if set(frame.method) != set(cutoffs):
        raise ValueError("Sweep methods differ from the committed cutoff methods")
    for method, cutoff in cutoffs.items():
        if not np.isclose(
            frame.loc[frame.method == method, "pred_cutoff"], cutoff
        ).all():
            raise ValueError(
                f"{method}: sweep does not use the committed cutoff {cutoff}"
            )
    if not np.isclose(frame.mbg_sr_threshold, 1.0).all():
        raise ValueError("Sweep success threshold differs from MBG RMSD <= 1 A")
    lines = [
        "## 6. Training-fraction sweep (Fig. S28)",
        "",
        "Fixed final hyperparameters, deterministic training-group subsets; no Optuna retuning. "
        "Success is MBG RMSD <= 1 Å among retained systems. Values are mean ± sample SD "
        "over the three test docking runs. Fractions refer to the available train+validation "
        "groups; the final models use 100%.",
        "",
        f"Input: `{paths[0].name}`.",
        "",
        "| Engine | Test | Training fraction | Cutoff (Å) | Groups used / total | Retained systems (mean) | MBG success (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (method, dock, fraction), group in frame.groupby(
        ["method", "dock_key", "fraction"], sort=True
    ):
        if len(group) != 3 or set(group.rep_id) != {1, 2, 3}:
            raise ValueError(
                f"{method}/{dock}/{fraction}: expected three distinct runs"
            )
        for column in ["systems_used", "systems_total"]:
            if group[column].nunique() != 1:
                raise ValueError(
                    f"{method}/{dock}/{fraction}: inconsistent training counts"
                )
        lines.append(
            f"| {method} | {dock} | {fraction:.0%} | {cutoffs[method]:.2f} | "
            f"{group.systems_used.iloc[0]} / {group.systems_total.iloc[0]} | "
            f"{group.systems_count.mean():.1f} | "
            f"{100 * group.mbg_sr_top1.mean():.1f} ± {100 * group.mbg_sr_top1.std():.1f} |"
        )
    return "\n".join(lines)
