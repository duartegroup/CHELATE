"""Plot the metal-identity probe figures of the SI (Figs. S25 and S27).

Fig. S26 (training MBG RMSDs by metal) is made by
``../benchmark/benchmark.ipynb``.

Inputs are the output folders of ``python -m model.predict_metal_identity`` (Vina model):

- ``probe_npose100/``: the probe reported in the paper (crystal pose featurised as
  pose 1 of 100, ``--n-poses 100``).
- ``probe_npose1/``: the same with ``--n-poses 1`` (sensitivity check).

Each folder needs ``metal_identity_predictions_long.csv``,
``system_prediction_summary.csv`` and ``bootstrap_metrics.json``; folders that are
missing are skipped. Figures, the per-metal accuracy tables and
``metal_identity_summary.json`` (the numbers quoted in the paper) go to ``figures/``.

Usage, from ``figures/metal_identity``::

    python make_metal_identity_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chelate.constants import EXPLICIT_METALS

HERE = Path(__file__).resolve().parent
PROBES = {
    "npose1": HERE / "probe_npose1",
    "npose100": HERE / "probe_npose100",
}
OUT_DIR = HERE / "figures"

# ---- Style, copied from the notebooks ----
fontsize_x_label = 8
fontsize_y_label = 8
fontsize_x_ticks = 8
fontsize_y_ticks = 8
fontsize_legend = 8
fontsize_titles = 8
cm = 1 / 2.54
figsize_two_panel = (16 * cm, 8 * cm)
VINA_COLOR = "#332288"

_EXPLICIT_ORDER = ["K", "Co", "Mn", "Fe", "Ca", "Mg", "Cu", "Na", "Ni", "Zn"]


def probe_confusion_table(selected, true_metals):
    """Show every ranked candidate, including metals absent from the true labels."""
    candidates = [metal for metal in _EXPLICIT_ORDER if metal in EXPLICIT_METALS]
    if set(candidates) != set(EXPLICIT_METALS):
        raise ValueError("Plot candidate order disagrees with the probe protocol")
    explicit = selected[selected["candidate_metal"].isin(candidates)]
    counts = pd.crosstab(explicit["true_metal"], explicit["candidate_metal"])
    counts = counts.reindex(index=true_metals, columns=candidates, fill_value=0).fillna(
        0
    )
    denominators = (
        explicit[["system_id", "ligand_id", "true_metal"]]
        .drop_duplicates()["true_metal"]
        .value_counts()
        .reindex(true_metals, fill_value=0)
    )
    return counts.div(denominators.replace(0, np.nan), axis=0).mul(100).fillna(0)


def savefig_padded(fig, path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)


def apply_axis_fonts(ax):
    ax.tick_params(axis="x", labelsize=fontsize_x_ticks)
    ax.tick_params(axis="y", labelsize=fontsize_y_ticks)


# ---------------------------------------------------------------------------
# Fig. S25: top-1 / top-3 heat maps (model.ipynb cells 11-12)
# ---------------------------------------------------------------------------
def plot_s25(probe_dir: Path, tag: str) -> dict:
    import seaborn as sns

    _pred_df = pd.read_csv(probe_dir / "metal_identity_predictions_long.csv")
    _active = [m for m in _EXPLICIT_ORDER if m in _pred_df["true_metal"].unique()]
    _expl = _pred_df[_pred_df["candidate_metal"].isin(_EXPLICIT_ORDER)].copy()

    _top1 = (
        _expl.sort_values(
            ["system_id", "ligand_id", "pred_mbg_rmsd", "candidate_metal"]
        )
        .groupby(["system_id", "ligand_id"], as_index=False)
        .first()
    )
    _top1_pct = probe_confusion_table(_top1, _active)

    _top3 = (
        _expl.sort_values(
            ["system_id", "ligand_id", "pred_mbg_rmsd", "candidate_metal"]
        )
        .groupby(["system_id", "ligand_id"], as_index=False)
        .head(3)
    )
    _top3_sys_counts = (
        _top3[["system_id", "ligand_id", "true_metal"]]
        .drop_duplicates()["true_metal"]
        .value_counts()
        .to_dict()
    )
    _top3_pct = probe_confusion_table(_top3, _active)

    def _n_labels(ax, metals, counts):
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(np.arange(len(metals)) + 0.5)
        ax2.set_yticklabels(
            [f"n={int(counts.get(m, 0))}" for m in metals], fontsize=fontsize_y_ticks
        )
        ax2.tick_params(axis="y", right=True, length=0, pad=6)
        ax2.grid(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

    fig, (ax_t1, ax_t3) = plt.subplots(1, 2, figsize=figsize_two_panel)
    sns.heatmap(
        _top1_pct,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax_t1,
        vmin=0,
        vmax=100,
        cbar=False,
        annot_kws={"size": 6},
    )
    ax_t1.set_title(
        f"(a) Top-1 accuracy (chance {100 / len(EXPLICIT_METALS):.0f}%)",
        fontsize=fontsize_titles,
        loc="left",
    )
    ax_t1.set_xlabel("Predicted Metal", fontsize=fontsize_x_label)
    ax_t1.set_ylabel("Crystallographic Metal", fontsize=fontsize_y_label)
    ax_t1.tick_params(labelsize=fontsize_x_ticks)
    sns.heatmap(
        _top3_pct,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax_t3,
        vmin=0,
        vmax=100,
        cbar=False,
        annot_kws={"size": 6},
    )
    ax_t3.set_title(
        f"(b) Top-3 accuracy (chance {300 / len(EXPLICIT_METALS):.0f}%)",
        fontsize=fontsize_titles,
        loc="left",
    )
    ax_t3.set_xlabel("Predicted Metal", fontsize=fontsize_x_label)
    ax_t3.set_ylabel("")
    ax_t3.tick_params(labelsize=fontsize_x_ticks)
    _n_labels(ax_t3, _active, _top3_sys_counts)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Accuracy (%)", fontsize=fontsize_x_label)
    cbar.ax.tick_params(labelsize=fontsize_x_ticks)
    for ext in ("pdf", "png"):
        savefig_padded(
            fig, OUT_DIR / f"S25_probe_accuracy_heatmaps_{tag}.{ext}", dpi=300
        )
    plt.close(fig)

    _top1_pct.to_csv(OUT_DIR / f"S25_top1_pct_{tag}.csv")
    _top3_pct.to_csv(OUT_DIR / f"S25_top3_pct_{tag}.csv")

    # Summary numbers for the comparison table.
    _top1["correct"] = _top1["candidate_metal"] == _top1["true_metal"]
    _top3_hit = (
        _top3.assign(hit=_top3["candidate_metal"] == _top3["true_metal"])
        .groupby(["system_id", "ligand_id", "true_metal"])
        .hit.any()
        .reset_index()
    )
    out = {
        "n_systems": len(_top1),
        "n_candidate_metals": len(EXPLICIT_METALS),
        "uniform_top1_chance_pct": 100 / len(EXPLICIT_METALS),
        "uniform_top3_chance_pct": 300 / len(EXPLICIT_METALS),
        "all_zero_in_primary_ranking": False,
        "top1_acc_pct": 100 * _top1["correct"].mean(),
        "top3_acc_pct": 100 * _top3_hit["hit"].mean(),
        "top1_pred_counts": _top1["candidate_metal"].value_counts().to_dict(),
    }
    for m in _active:
        sub = _top1[_top1["true_metal"] == m]
        out[f"top1_{m}_pct"] = 100 * sub["correct"].mean()
        out[f"top3_{m}_pct"] = (
            100 * _top3_hit[_top3_hit["true_metal"] == m]["hit"].mean()
        )
        out[f"n_{m}"] = len(sub)
    return out


# ---------------------------------------------------------------------------
# Fig. S27: predicted MBG RMSD with/without metal OHE (model.ipynb cell 13)
# ---------------------------------------------------------------------------
def plot_s27(probe_dir: Path, tag: str) -> dict:
    _summ = pd.read_csv(probe_dir / "system_prediction_summary.csv")
    with open(probe_dir / "bootstrap_metrics.json") as _fh_:
        _bs = json.load(_fh_)
    _bs_met = _bs["metrics"]
    _ci_pct = float(_bs["ci_pct"])
    _alpha_ = (100.0 - _ci_pct) / 2.0
    _active = [m for m in _EXPLICIT_ORDER if m in _summ["true_metal"].unique()]

    _rng_ohe = np.random.default_rng(42)
    _N_BOOT = 1000
    _pm_ci = {}
    for _m in _active:
        _sub = _summ[_summ["true_metal"] == _m]
        _n_m = len(_sub)
        _bs_w, _bs_wo = [], []
        for _ in range(_N_BOOT):
            _s = _sub.iloc[_rng_ohe.integers(0, _n_m, size=_n_m)]
            _bs_w.append(float(_s["true_pred_mbg_rmsd"].mean()))
            _bs_wo.append(float(_s["all_zero_pred_mbg_rmsd"].mean()))
        _pm_ci[_m] = {
            "with": {
                "mean": float(_sub["true_pred_mbg_rmsd"].mean()),
                "lo": float(np.percentile(_bs_w, _alpha_)),
                "hi": float(np.percentile(_bs_w, 100.0 - _alpha_)),
            },
            "without": {
                "mean": float(_sub["all_zero_pred_mbg_rmsd"].mean()),
                "lo": float(np.percentile(_bs_wo, _alpha_)),
                "hi": float(np.percentile(_bs_wo, 100.0 - _alpha_)),
            },
        }

    _c_with = "#332288"
    _c_without = "#88CCEE"
    fig_ohe, (ax_g, ax_m) = plt.subplots(
        1, 2, figsize=(16 * cm, 8 * cm), gridspec_kw={"width_ratios": [1, 3]}
    )
    _gw = _bs_met["mean_true_pred_mbg_rmsd"]
    _gwo = _bs_met["mean_all_zero_pred_mbg_rmsd"]
    _g_means = [_gw["mean"], _gwo["mean"]]
    _g_lo = [_gw["mean"] - _gw["ci_lower"], _gwo["mean"] - _gwo["ci_lower"]]
    _g_hi = [_gw["ci_upper"] - _gw["mean"], _gwo["ci_upper"] - _gwo["mean"]]
    ax_g.bar(
        [0, 1],
        _g_means,
        width=0.5,
        color=[_c_with, _c_without],
        yerr=[_g_lo, _g_hi],
        capsize=4,
        error_kw={"linewidth": 1.0, "ecolor": "black"},
        zorder=3,
    )
    ax_g.set_xticks([0, 1])
    ax_g.set_xticklabels(["With OHE", "Without OHE"], fontsize=fontsize_x_ticks)
    ax_g.set_ylabel("Mean Predicted MBG RMSD (Å)", fontsize=fontsize_y_label)
    ax_g.set_title(
        f"a) Overall (n = {len(_summ)})", fontsize=fontsize_titles, loc="left"
    )
    ax_g.set_ylim(0, 1)
    ax_g.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax_g.tick_params(labelsize=fontsize_x_ticks)

    _x = np.arange(len(_active))
    _bw = 0.35
    for _off, _key, _col, _lbl in [
        (-_bw / 2, "with", _c_with, "With OHE"),
        (_bw / 2, "without", _c_without, "Without OHE"),
    ]:
        _means_ = [_pm_ci[m][_key]["mean"] for m in _active]
        _lo_ = [_pm_ci[m][_key]["mean"] - _pm_ci[m][_key]["lo"] for m in _active]
        _hi_ = [_pm_ci[m][_key]["hi"] - _pm_ci[m][_key]["mean"] for m in _active]
        ax_m.bar(
            _x + _off,
            _means_,
            width=_bw,
            color=_col,
            yerr=[_lo_, _hi_],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black"},
            label=_lbl,
            zorder=3,
        )
    _xlabels = [f"{m}\n(n={len(_summ[_summ['true_metal'] == m])})" for m in _active]
    ax_m.set_xticks(_x)
    ax_m.set_xticklabels(_xlabels, fontsize=fontsize_x_ticks)
    ax_m.set_xlabel("Crystallographic Metal", fontsize=fontsize_x_label)
    ax_m.set_title("b) Per metal", fontsize=fontsize_titles, loc="left")
    ax_m.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax_m.tick_params(labelsize=fontsize_x_ticks)
    ax_m.legend(fontsize=fontsize_legend, frameon=False, loc="upper center", ncol=2)
    ax_m.set_ylim(0, 1)
    fig_ohe.tight_layout()
    for ext in ("pdf", "png"):
        savefig_padded(
            fig_ohe,
            OUT_DIR / f"S27_probe_pred_mbg_rmsd_with_without_ohe_{tag}.{ext}",
            dpi=300,
        )
    plt.close(fig_ohe)

    diff = _bs_met["mean_true_vs_all_zero_diff"]
    frac = _bs_met["frac_true_better_than_all_zero"]
    return {
        "mean_pred_with_ohe": _gw["mean"],
        "mean_pred_with_ohe_ci": (_gw["ci_lower"], _gw["ci_upper"]),
        "mean_pred_without_ohe": _gwo["mean"],
        "mean_pred_without_ohe_ci": (_gwo["ci_lower"], _gwo["ci_upper"]),
        "mean_with_minus_without": diff["mean"],
        "mean_with_minus_without_ci": (diff["ci_lower"], diff["ci_upper"]),
        "frac_with_better_than_without": frac["mean"],
        "pred_spread_across_candidates": float(
            pd.read_csv(probe_dir / "metal_identity_predictions_long.csv")
            .query("candidate_metal != 'ALL_ZERO'")
            .groupby(["system_id", "ligand_id"])
            .pred_mbg_rmsd.agg(lambda s: s.max() - s.min())
            .mean()
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for tag, probe_dir in PROBES.items():
        if not (probe_dir / "metal_identity_predictions_long.csv").exists():
            print(f"Skipping {tag}: no probe outputs in {probe_dir}")
            continue
        summary[tag] = {"probe_dir": str(probe_dir)}
        summary[tag].update(plot_s25(probe_dir, tag))
        summary[tag].update(plot_s27(probe_dir, tag))
    with open(OUT_DIR / "metal_identity_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
