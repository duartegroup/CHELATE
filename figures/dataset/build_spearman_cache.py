"""Build ``spearman_cache.pkl`` for Fig. S15 from the docking feature CSVs.

Fig. S15 shows, per docking program and split, the distribution of per-system Spearman
correlations between the docking score and the MBG RMSD (``close_rmsd_3.5_hetero``) or
the whole-molecule RMSD (``symm_rmsd``). The score is oriented so that higher means a
better pose (``CNNscore`` for GNINA, minus the docking score for Vina and smina), so a
negative correlation means that better-scored poses are closer to the crystal pose.

Train and validation use the re-docked run 1. The test split has six runs (re-docked and
cross-docked, three each); ``--test_aggregation`` decides how they are combined into one
value per system/ligand pair:

``mean_of_runs``
    one correlation per run, averaged over the runs of the system;
``pooled_poses``
    one correlation over the poses of all runs together (default, used for Fig. S15).

Both panels use only poses with an MBG RMSD label. This keeps their system sets
identical; systems without a crystallographic metal-binding group therefore do not
appear in either panel.

Usage, from ``figures/dataset``::

    python build_spearman_cache.py --output spearman_cache.pkl

``--check spearman_cache.pkl`` compares a rebuilt cache with the committed one.

Output format (what ``dataset_statistics.ipynb`` reads)::

    {method: {rmsd_column: {split: np.ndarray of correlations}}}
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

METHODS = ("VINA", "SMINA", "GNINA")
RMSD_COLS = ("close_rmsd_3.5_hetero", "symm_rmsd")
TEST_RUN_IDS = (1, 2, 3)
MIN_POSES = 3


def _score(df: pd.DataFrame, method: str) -> pd.Series:
    """Docking score oriented so that higher is better."""
    if method == "GNINA":
        return pd.to_numeric(df["CNNscore"], errors="coerce")
    col = "docking_score" if "docking_score" in df.columns else "free_energy"
    return -pd.to_numeric(df[col], errors="coerce")


def _read(path: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])  # ligand chain "NA"
    df["uid"] = df["system_id"].astype(str) + "__" + df["ligand_id"].astype(str)
    df["score"] = _score(df, method)
    for col in RMSD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["uid", "score", *RMSD_COLS]]


def _per_system_rho(df: pd.DataFrame, rmsd_col: str) -> pd.Series:
    """Spearman rho of score vs RMSD for every uid with enough valid poses."""
    ok = df[
        np.isfinite(df["score"])
        & np.isfinite(df["close_rmsd_3.5_hetero"])
        & np.isfinite(df[rmsd_col])
    ]
    out = {}
    for uid, grp in ok.groupby("uid"):
        if len(grp) < MIN_POSES:
            continue
        rho = stats.spearmanr(grp["score"], grp[rmsd_col])[0]
        if np.isfinite(rho):
            out[uid] = rho
    return pd.Series(out, dtype=float)


def build(features_dir: str, test_aggregation: str) -> dict:
    cache: dict = {}
    for method in METHODS:
        cache[method] = {col: {} for col in RMSD_COLS}
        f = os.path.join(features_dir, "features_{split}_{method}_{run}_id={rid}.csv")
        for split in ("train", "val"):
            df = _read(
                f.format(split=split, method=method, run="redocked_FH", rid=1), method
            )
            for col in RMSD_COLS:
                cache[method][col][split] = _per_system_rho(df, col).to_numpy()
        runs = [
            _read(f.format(split="test", method=method, run=run, rid=rid), method)
            for run in ("redocked_FH", "apo_docked_FH")
            for rid in TEST_RUN_IDS
        ]
        for col in RMSD_COLS:
            if test_aggregation == "pooled_poses":
                rho = _per_system_rho(pd.concat(runs, ignore_index=True), col)
            else:
                per_run = pd.concat([_per_system_rho(r, col) for r in runs])
                rho = per_run.groupby(level=0).mean()
            cache[method][col]["test"] = rho.to_numpy()
    return cache


def summarise(cache: dict) -> pd.DataFrame:
    rows = [
        {
            "method": m,
            "rmsd": col,
            "split": split,
            "n": len(a),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }
        for m, cols in cache.items()
        for col, splits in cols.items()
        for split, a in splits.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--features_dir", default=os.path.join(REPO, "docking", "features")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--test_aggregation",
        choices=["mean_of_runs", "pooled_poses"],
        default="pooled_poses",
    )
    parser.add_argument(
        "--check",
        help="Reference cache to compare with; exits non-zero if n, mean or std differ.",
    )
    args = parser.parse_args()

    cache = build(args.features_dir, args.test_aggregation)
    with open(args.output, "wb") as fh:
        pickle.dump(cache, fh)
    new = summarise(cache)
    print(new.to_string(index=False, float_format="%.3f"))

    if args.check:
        with open(args.check, "rb") as fh:
            ref = summarise(pickle.load(fh))
        cmp = new.merge(ref, on=["method", "rmsd", "split"], suffixes=("", "_ref"))
        bad = cmp[
            (cmp["n"] != cmp["n_ref"])
            | ~np.isclose(cmp["mean"], cmp["mean_ref"], atol=5e-4)
            | ~np.isclose(cmp["std"], cmp["std_ref"], atol=5e-4)
        ]
        if len(bad):
            print("\nDoes not reproduce the reference cache:")
            print(bad.to_string(index=False, float_format="%.3f"))
            sys.exit(1)
        print("\nReproduces the reference cache.")


if __name__ == "__main__":
    main()
