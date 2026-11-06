"""Print the paper's model numbers from the evaluation CSVs.

Reads the CSV bundle in ``../csv_files`` and writes Markdown tables to the path given
as the first argument (``NUMBERS.md`` sections 1-7). The operating point is each
docking program's validation-selected cutoff from ``model/params/pred_cutoffs.json``
(``model/select_cutoff.py``); the 1.0 Å tables are given for comparison.

Usage, from ``figures/model``::

    python scripts/paper_numbers.py NUMBERS_regenerated.md
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from cohort_numbers import render_cohorts
from scipy import stats
from sweep_numbers import render_sweep

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_MODEL = os.path.dirname(HERE)
CSV_DIR = os.path.join(FIG_MODEL, "csv_files")
EC_CSV = os.path.join(FIG_MODEL, "..", "..", "model", "data", "EC_numbers.csv")
METALS_JSON = os.path.join(FIG_MODEL, "test_system_metals.json")
M = ["VINA", "SMINA", "GNINA"]
with open(
    os.path.join(FIG_MODEL, "..", "..", "model", "params", "pred_cutoffs.json")
) as _fh:
    _selected = json.load(_fh)
# Operating point: the validation-selected cutoff of each docking program.
OP = {m: float(_selected[m]) for m in M}
OP_LABEL = ", ".join(f"{m} {OP[m]:.2f} Å" for m in M)
NM = {"VINA": "Vina", "SMINA": "smina", "GNINA": "GNINA"}
DK = ["crossdocked", "redocked"]
out = []


def p(s=""):
    out.append(s)


def load(kind):
    return pd.read_csv(f"{CSV_DIR}/plot_repro_prediction_quality_{kind}.csv")


G = pd.read_csv(f"{CSV_DIR}/plot_repro_grouped_metrics.csv")


def row(m, dk, cut):
    if cut == "op":
        cut = OP[m]
    g = G[(G.method == m) & (G.dock_key == dk)]
    r = g[np.isclose(g.rmsd_cutoff, cut)]
    return r.iloc[0] if len(r) else None


def base(m, dk, col="baseline_MBG_sr_top1_RMSD=1"):
    g = G[(G.method == m) & (G.dock_key == dk)].sort_values("rmsd_cutoff")
    r = g.iloc[-1]
    return r[col + "__mean"], r[col + "__std"]


f = lambda m, s: f"{100 * m:.1f} ± {100 * s:.1f}"

p(
    "## 1. Fig. 3c headline: top-1 MBG success rate (%; MBG RMSD <= 1 Å), mean ± SD over 3 docking runs"
)
p("")
p(
    "Retained = success among systems with >=1 pose at predicted MBG RMSD <= cutoff. "
    "All systems = same CHELATE pick, but systems with no retained pose count as failures. "
    "Retained fraction = retained systems / all test systems. "
    "All values are on the common 118-pair cohort of all three docking programs, both "
    "scenarios and all repeats. Section 7 gives the per-program and all-156 tables."
)
for cut in ("op", 1.0):
    p("")
    p(
        f"### Prediction cutoff: {'operating point (' + OP_LABEL + ')' if cut == 'op' else f'{cut} Å'}"
    )
    p("")
    p(
        "| Dock | Method | Docking top-1 | CHELATE, retained | CHELATE, all systems | Retained fraction |"
    )
    p("|---|---|---:|---:|---:|---:|")
    for dk in DK:
        for m in M:
            r = row(m, dk, cut)
            c = "MBG_sr_top1_RMSD=1"
            a = "MBG_sr_all_systems_top1_RMSD=1"
            fr = (
                f"{100 * r['systems_count__mean'] / r['total_systems__mean']:.1f}% "
                f"({r['systems_count__mean']:.0f}/{r['total_systems__mean']:.0f})"
            )
            p(
                f"| {dk} | {NM[m]} | {f(*base(m, dk))} | {f(r[c + '__mean'], r[c + '__std'])} | "
                f"{f(r[a + '__mean'], r[a + '__std'])} | {fr} |"
            )

# external baselines at 1 A top-1
p("")
p("External baselines, top-1 MBG success at 1 Å (%):")
p("")
p("| Dock | Method | Top-1 |")
p("|---|---|---:|")
E = pd.read_csv(f"{CSV_DIR}/plot_repro_external_baselines.csv")
for dk in DK:
    for m in ["AUTODOCK", "GPDOCK", "METALLODOCK"]:
        e = E[
            (E.dock_key == dk)
            & (E.method == m)
            & (E.metric_family == "mbg")
            & np.isclose(E.threshold.astype(float), 1.0)
            & (E.top_k == 1)
        ]
        p(f"| {dk} | {m} | {f(e['mean'].iloc[0], e['std'].iloc[0])} |")

p("")
p("## 2. Pearson r, predicted vs true MBG RMSD")
p("")
p("Per-system r (Fig. 3a; mean ± SD over systems pooled across runs):")
p("")
p("| Dock | Method | r | n systems×runs |")
p("|---|---|---:|---:|")
C = load("corr")
for dk in DK:
    for m in M:
        v = pd.to_numeric(
            C[(C.method == m) & (C.dock_key == dk)].pearson_r, errors="coerce"
        )
        v = v[np.isfinite(v)]
        p(f"| {dk} | {NM[m]} | {v.mean():.3f} ± {v.std():.3f} | {len(v)} |")
p("")
p("Overall pooled r over all poses (Fig. S16):")
p("")
p("| Dock | Method | r |")
p("|---|---|---:|")
SC = load("scatter")
for dk in DK:
    for m in M:
        d = SC[(SC.method == m) & (SC.dock_key == dk)]
        x = pd.to_numeric(d.true_mbg_rmsd, errors="coerce")
        y = pd.to_numeric(d.pred, errors="coerce")
        k = np.isfinite(x) & np.isfinite(y)
        p(f"| {dk} | {NM[m]} | {stats.pearsonr(x[k], y[k])[0]:.3f} |")

p("")
p(
    "## 3. Binding-affinity Pearson r (Fig. S20): docking score vs experimental affinity, top-1 pose"
)
p("")
p(
    "Baseline = all systems with affinity; filtered = CHELATE-retained systems. n = systems with affinity (mean over runs)."
)
for cut in ("op", 1.0):
    p("")
    p(
        f"Prediction cutoff, {'operating point (' + OP_LABEL + ')' if cut == 'op' else f'{cut} Å'}:"
    )
    p("")
    p("| Dock | Method | Baseline (n) | Filtered (n) |")
    p("|---|---|---:|---:|")
    for dk in DK:
        for m in M:
            r = row(m, dk, cut)
            cells = [
                f"{r[pre + '_Pearson_R__mean']:.3f} ± {r[pre + '_Pearson_R__std']:.3f} ({r[pre + '_systems_count__mean']:.0f})"
                for pre in ("baseline_affinity", "affinity")
            ]
            p(f"| {dk} | {NM[m]} | " + " | ".join(cells) + " |")

p("")
p(
    "## 4. Feature importance (Fig. S23 CatBoost PredictionValuesChange; Fig. S24 SHAP), top 5 per model"
)
p("")
p("| Method | Feature importance | |mean SHAP| |")
p("|---|---|---|")
for m in M:
    cells = []
    for kind, col in (("fi", "importance"), ("shap", "mean_abs_shap")):
        x = (
            pd.read_csv(f"{CSV_DIR}/interpret_{kind}_{m}.csv")
            .sort_values(col, ascending=False)
            .head(5)
        )
        cells.append(", ".join(f"{a} ({b:.2g})" for a, b in zip(x.feature, x[col])))
    p(f"| {NM[m]} | {cells[0]} | {cells[1]} |")

# ---- enzyme / metal subsets (same logic as the notebook's Fig. S22 cell)
p("")
p(
    f"## 5. Fig. S22 subsets, cross-docked, prediction cutoff {OP_LABEL}, MBG SR (%) baseline → CHELATE (retained)"
)
p("")
EC = pd.read_csv(EC_CSV, encoding="utf-8-sig")
with open(METALS_JSON) as fh:
    lm = json.load(fh)


def has_ec(pdb):
    r = EC[EC["PDB ID"] == pdb]
    return (
        bool(len(r))
        and pd.notna(r.iloc[0]["EC raw text"])
        and r.iloc[0]["EC raw text"] != ""
    )


def subset(sc, keys, m, cut):
    df = sc[
        (sc.method == m)
        & (sc.dock_key == "crossdocked")
        & sc.system_id__ligand_id.isin(keys)
    ]
    b, c, nb, nc = [], [], [], []
    for rep in sorted(df.rep_id.unique()):
        r = df[df.rep_id == rep]
        t1 = r[r.pose == 1]
        b.append((t1.true_mbg_rmsd <= 1.0).mean() * 100 if len(t1) else 0)
        nb.append(t1.system_id__ligand_id.nunique())
        fl = r[r.pred <= cut]
        c.append(
            (
                fl.sort_values("pose")
                .groupby("system_id__ligand_id")
                .first()
                .true_mbg_rmsd
                <= 1.0
            ).mean()
            * 100
            if len(fl)
            else 0
        )
        nc.append(fl.system_id__ligand_id.nunique())
    return f"{np.mean(b):.1f} → {np.mean(c):.1f} (n {nb[0]} → {'/'.join(map(str, nc))})"


keys = SC.system_id__ligand_id.unique()
meta = pd.DataFrame({"k": keys})
meta["enz"] = [has_ec(k.split("__")[0]) for k in keys]
meta["metals"] = [
    frozenset(x[0].upper() for x in lm.get(k.rsplit("__", 1)[0], [])) for k in keys
]
groups = {
    "Enzymes": set(meta.k[meta.enz]),
    "Non-enzymes": set(meta.k[~meta.enz]),
}
for mt in ["ZN", "MG", "FE", "CA", "NI"]:
    groups[mt.capitalize()] = set(meta.k[meta.metals.apply(lambda s: mt in s)])
p("| Subset | Method | MBG SR |")
p("|---|---|---|")
for gname, gkeys in groups.items():
    for m in M:
        p(f"| {gname} | {NM[m]} | {subset(SC, gkeys, m, OP[m])} |")
p("")
p(render_sweep(CSV_DIR, OP))
cohort_tables = render_cohorts(CSV_DIR, OP)
if cohort_tables:
    p("")
    p(cohort_tables)
with open(sys.argv[1], "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
