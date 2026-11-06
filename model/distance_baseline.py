"""Distance-cutoff baseline for CHELATE: keep poses with a ligand N/O/S atom near a metal.

The simplest possible metal-aware post-docking filter. A pose is kept if its closest
ligand N/O/S atom is within ``d`` of a receptor metal; the kept poses are ranked by the
docking program's own score, exactly like CHELATE's retained poses. Systems with no
kept pose are discarded.

The distance comes from the CHELATE feature ``ch_4.5_min`` (minimum ligand-N/O/S to
metal distance among pairs within 4.5 Å, 0 with a pair count ``ch_4.5`` of 0 when there
is none), so the same feature CSVs serve as input and ``d`` can range up to 4.5 Å.

``d`` is chosen on the re-docked train + validation poses, never on the test set, by
two rules:

``max_all_systems``
    maximise the top-1 MBG success rate over all systems (discarded systems count as
    failures). This is the objective CHELATE's hyperparameters were tuned on.
``max_retained_at_coverage``
    maximise the top-1 MBG success rate among retained systems, subject to retaining at
    least ``--min_coverage`` of the systems (default 0.5). ``model/select_cutoff.py``
    chooses CHELATE's per-program prediction cutoffs with the same rule.

Both cutoffs are then applied to the three re-docked and three cross-docked test runs.
Poses are filtered exactly as in ``model/test_eval.py::evaluate_model``: rows with a
non-finite score or without an MBG label are dropped first. Success is MBG RMSD <= 1 Å.

Outputs (in ``--output_dir``, default ``model/model_out_distance_baseline/``):

``trainval_curve.csv``
    retained fraction and success rates for every ``d`` on train + validation;
``test_curve.csv``
    the same on the test set, per method, docking type and run;
``summary.csv``
    the two fitted cutoffs and their test results (mean ± SD over runs), next to the
    docking-score baseline;
``precision_coverage.pdf``
    success among retained systems against the retained fraction, for the distance
    filter (all ``d``) and for CHELATE (all prediction cutoffs, read from
    ``--chelate_grouped_csv``), cross-docked and re-docked.

Run from the repository root::

    python -m model.distance_baseline
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from model.model_params import (
    FEATURE_CSV_TEMPLATE,
    FEATURES_DIR,
    MBG_RMSD_THRESHOLD,
    MODEL_DIR,
    TEST_RUN_IDS,
    TRAIN_RUN_ID,
    TRAIN_RUN_TYPE,
)

METHODS = ["VINA", "SMINA", "GNINA"]
TARGET = "close_rmsd_3.5_hetero"
DIST_COL = "ch_4.5_min"
COUNT_COL = "ch_4.5"
DOCK_RUN_TYPES = {"redocked": "redocked_FH", "crossdocked": "apo_docked_FH"}
D_GRID = np.round(np.arange(1.8, 4.5001, 0.05), 2)


def score_column(method: str) -> str:
    return "CNNscore" if method == "GNINA" else "docking_score"


def load_poses(path: str, method: str) -> pd.DataFrame:
    """Load one feature CSV and keep the columns the baseline needs."""
    score = score_column(method)
    header = pd.read_csv(path, nrows=0).columns
    if score not in header and "free_energy" in header:
        score_src = "free_energy"
    else:
        score_src = score
    df = pd.read_csv(
        path,
        usecols=["system_id", "ligand_id", score_src, TARGET, DIST_COL, COUNT_COL],
        keep_default_na=False,
        na_values=[""],  # keep ligand chain "NA" as a string
    )
    df = df.rename(columns={score_src: "score"})
    for col in ("score", TARGET, DIST_COL, COUNT_COL):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Same row filter as test_eval.evaluate_model.
    df = df[np.isfinite(df["score"]) & df[TARGET].notna()].copy()
    df["uid"] = df["system_id"].astype(str) + "__" + df["ligand_id"].astype(str)
    df["min_dist"] = np.where(df[COUNT_COL] > 0, df[DIST_COL], np.inf)
    # Best score first within each system, stable so ties keep the file order
    # (as rank(method="first") does in model_utils.rerank_poses).
    ascending = score_column(method) != "CNNscore"
    return df.sort_values(["uid", "score"], ascending=[True, ascending], kind="stable")


def top1_rates(df: pd.DataFrame, d: float) -> dict[str, float]:
    """Top-1 MBG success among retained systems and over all systems at cutoff d."""
    n_all = df["uid"].nunique()
    kept = df[df["min_dist"] <= d]
    top1 = kept.groupby("uid", sort=False).head(1)
    success = (top1[TARGET] <= MBG_RMSD_THRESHOLD).sum()
    n_kept = len(top1)
    return {
        "d": d,
        "n_systems": n_all,
        "n_retained": n_kept,
        "retained_fraction": n_kept / n_all if n_all else np.nan,
        "sr_retained": success / n_kept if n_kept else np.nan,
        "sr_all_systems": success / n_all if n_all else np.nan,
    }


def baseline_rate(df: pd.DataFrame) -> float:
    top1 = df.groupby("uid", sort=False).head(1)
    return float((top1[TARGET] <= MBG_RMSD_THRESHOLD).mean())


def fit_cutoffs(curve: pd.DataFrame, min_coverage: float) -> dict[str, float]:
    """Pick d on the train + validation curve. Ties go to the larger (looser) d."""
    best_all = curve.loc[curve["sr_all_systems"][::-1].idxmax(), "d"]
    ok = curve[curve["retained_fraction"] >= min_coverage]
    best_ret = ok.loc[ok["sr_retained"][::-1].idxmax(), "d"] if len(ok) else np.nan
    return {"max_all_systems": best_all, "max_retained_at_coverage": best_ret}


def feature_path(features_dir: str, split: str, method: str, run_type: str, rid) -> str:
    return os.path.join(
        features_dir,
        FEATURE_CSV_TEMPLATE.format(
            split=split, method=method, run_type=run_type, run_id=rid
        ),
    )


def plot_precision_coverage(
    test_curve: pd.DataFrame,
    summary: pd.DataFrame,
    chelate_csv: str | None,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"VINA": "#332288", "SMINA": "#88CCEE", "GNINA": "#44AA99"}
    display_names = {"VINA": "Vina", "SMINA": "smina", "GNINA": "GNINA"}
    rule_markers = {"max_all_systems": "X", "max_retained_at_coverage": "P"}
    rule_labels = {
        "max_all_systems": "fitted d: best all-systems rate",
        "max_retained_at_coverage": "fitted d: best retained rate at ≥50 % coverage",
    }
    chel = None
    if chelate_csv and os.path.exists(chelate_csv):
        chel = pd.read_csv(chelate_csv)
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54), sharey=True)
    for ax, dock in zip(axes, ["crossdocked", "redocked"], strict=True):
        for method in METHODS:
            sub = test_curve[
                (test_curve["method"] == method) & (test_curve["dock"] == dock)
            ]
            mean = sub.groupby("d")[["retained_fraction", "sr_retained"]].mean()
            ax.plot(
                100 * mean["retained_fraction"],
                100 * mean["sr_retained"],
                color=colors[method],
                linestyle="--",
                label=f"{display_names[method]} distance filter",
            )
            fitted = summary[(summary["method"] == method) & (summary["dock"] == dock)]
            for row in fitted.itertuples(index=False):
                point = mean[np.isclose(mean.index, row.d)]
                if point.empty:
                    continue
                ax.scatter(
                    100 * point["retained_fraction"].iloc[0],
                    100 * point["sr_retained"].iloc[0],
                    color=colors[method],
                    edgecolor="black",
                    linewidth=0.6,
                    marker=rule_markers[row.rule],
                    s=60,
                    zorder=5,
                )
            if chel is not None:
                c = chel[(chel["method"] == method) & (chel["dock_key"] == dock)]
                frac = c["systems_count__mean"] / c["total_systems__mean"]
                ax.plot(
                    100 * frac,
                    100 * c[f"MBG_sr_top1_RMSD={MBG_RMSD_THRESHOLD:g}__mean"],
                    color=colors[method],
                    marker="o",
                    markersize=3,
                    label=f"{display_names[method]} CHELATE",
                )
        ax.set_title(
            "Cross-docked" if dock == "crossdocked" else "Re-docked", fontsize=8
        )
        ax.set_xlabel("Retained systems (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 MBG success, retained systems (%)", fontsize=8)
    handles, _ = axes[1].get_legend_handles_labels()
    handles += [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=rule_markers[rule],
            markerfacecolor="grey",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label=label,
        )
        for rule, label in rule_labels.items()
    ]
    fig.tight_layout()
    fig.legend(
        handles=handles,
        fontsize=6,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features_dir", default=FEATURES_DIR)
    parser.add_argument(
        "--output_dir", default=str(MODEL_DIR / "model_out_distance_baseline")
    )
    parser.add_argument("--min_coverage", type=float, default=0.5)
    parser.add_argument(
        "--chelate_grouped_csv",
        default=str(
            MODEL_DIR.parent
            / "figures"
            / "model"
            / "csv_files"
            / "plot_repro_grouped_metrics.csv"
        ),
        help="CHELATE's plot_repro_grouped_metrics.csv, for the precision-coverage plot.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainval_rows, test_rows, summary_rows = [], [], []
    for method in METHODS:
        trainval = pd.concat(
            [
                load_poses(
                    feature_path(
                        args.features_dir, split, method, TRAIN_RUN_TYPE, TRAIN_RUN_ID
                    ),
                    method,
                )
                for split in ("train", "val")
            ],
            ignore_index=True,
        )
        curve = pd.DataFrame([top1_rates(trainval, d) for d in D_GRID])
        curve["method"] = method
        trainval_rows.append(curve)
        cutoffs = fit_cutoffs(curve, args.min_coverage)
        print(
            f"[{method}] train+val baseline {100 * baseline_rate(trainval):.1f} %, "
            f"fitted d: {cutoffs}"
        )

        for dock, run_type in DOCK_RUN_TYPES.items():
            per_run = []
            for rid in TEST_RUN_IDS:
                test = load_poses(
                    feature_path(args.features_dir, "test", method, run_type, rid),
                    method,
                )
                tc = pd.DataFrame([top1_rates(test, d) for d in D_GRID])
                tc["method"], tc["dock"], tc["run"] = method, dock, rid
                tc["baseline_sr"] = baseline_rate(test)
                test_rows.append(tc)
                per_run.append(tc)
            runs = pd.concat(per_run)
            for rule, d in cutoffs.items():
                at = runs[np.isclose(runs["d"], d)]
                summary_rows.append(
                    {
                        "method": method,
                        "dock": dock,
                        "rule": rule,
                        "d": d,
                        "baseline_sr_mean": at["baseline_sr"].mean(),
                        "baseline_sr_std": at["baseline_sr"].std(ddof=1),
                        "retained_fraction_mean": at["retained_fraction"].mean(),
                        "sr_retained_mean": at["sr_retained"].mean(),
                        "sr_retained_std": at["sr_retained"].std(ddof=1),
                        "sr_all_systems_mean": at["sr_all_systems"].mean(),
                        "sr_all_systems_std": at["sr_all_systems"].std(ddof=1),
                    }
                )

    pd.concat(trainval_rows).to_csv(
        os.path.join(args.output_dir, "trainval_curve.csv"), index=False
    )
    test_curve = pd.concat(test_rows)
    test_curve.to_csv(os.path.join(args.output_dir, "test_curve.csv"), index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    plot_precision_coverage(
        test_curve,
        summary,
        args.chelate_grouped_csv,
        os.path.join(args.output_dir, "precision_coverage.pdf"),
    )
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
