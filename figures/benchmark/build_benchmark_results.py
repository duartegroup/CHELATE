"""Build benchmark_results.pkl for benchmark.ipynb from per-pose docking CSVs.

Inputs: Vina, smina and GNINA poses from the feature CSVs in ``docking/features``;
AutoDock-GPU, GPDOCK and MetalloDock poses from the result CSVs in
``docking/results/test``; the unseen-system CSVs (Figs. S13, S14) from
``--unseen_dir`` (default: ``unseen_systems/`` next to this script, written by
``reparse_unseen_systems.py``); the metal JSON in ``data_curation/cached_data``; and
the tables in ``figures/inputs/``. All of them are in the Zenodo record.

Metric definitions:

* Every method is evaluated on the same fixed pairs (``--cohort``): ``n`` is the
  cohort size, pairs without an output or label count as failures, and a missing
  repeat file is an error.
* Whole-molecule success: symmetry-corrected RMSD <= 2.0 A within the top N.
* MBG success: ``close_rmsd_3.5_hetero`` <= 1.0 A (``mbg2``: <= 2.0 A) within
  the top N; zero is a valid aligned RMSD.
* Means and sample standard deviations are over the three docking runs.
* Pose 1 is each program's best pose (AutoDock-GPU: lowest free energy). In the
  affinity panels, scores where lower is better are negated, so a positive
  correlation means better scores go with stronger binding.
* Cumulative curves use the top-ranked pose (pose == 1) of each system.
* Affinity: top-1 score per run, averaged over runs, against PLINDER's
  ``ligand_binding_affinity`` annotation, matched per ligand. BINDING_PRIO is
  committed without that column, so it is joined back from the ``_NEW`` dataset
  CSV (same rows) on system ID, ligand instance and ligand chain. All affinity
  panels use the pairs with an affinity and a top-1 score for every method and run.
* The leakage panels (Figs. S12-S14) compare each method with Vina and smina on
  the same pairs.

The pickle also holds per-method ``n_mbg`` and a top-level ``meta`` dict with the
inputs and the comparison cohorts.

Usage, from ``figures/benchmark``::

    python build_benchmark_results.py \
        --cohort ../../docking/cohorts/comparison_common_test.csv \
        --output rebuilt.pkl
    python compare_benchmark_pickles.py rebuilt.pkl benchmark_results.pkl
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from model.cohorts import (
    canonical_pair_columns,
    filter_to_cohort,
    fixed_cohort_success,
    pair_set,
    read_cohort,
)

# Small inputs that the repository pipeline does not produce (see figures/README.md).
INPUTS = os.path.join(REPO, "figures", "inputs")
CACHED_DATA = os.path.join(REPO, "data_curation", "cached_data")

FEATURES_DIR = f"{REPO}/docking/features"
EXTERNAL_DIR = f"{REPO}/docking/results/test"
UNSEEN_DIR_DEFAULT = f"{HERE}/unseen_systems"

METAL_LIGAND_JSON_PATH = os.path.join(
    CACHED_DATA,
    "metal_ligand_data_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.json",
)
# Dataset split table used for the cluster-level leakage sets (Fig. S12).
DATASET_NEW_CSV = os.path.join(
    INPUTS,
    "final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit_NEW.csv",
)
# Binding-prioritised test split (Figs. S10, S11); affinities come from DATASET_NEW_CSV.
DATASET_CSV = os.path.join(
    INPUTS,
    "final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit_BINDING_PRIO.csv",
)
# GNINA's training set (CrossDocked2020 v1.3) and PLINDER cluster assignments (Fig. S12).
CROSSDOCKED_TRAIN_CSV = os.path.join(INPUTS, "crossdocked_v1.3_train_unique.csv")
SPLIT_CLUSTERS_CSV = os.path.join(INPUTS, "split_clusters.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_N_POSES = [1, 3, 5, 10, 20, 100]
RMSD_CUTOFFS = np.linspace(0.0, 5.0, 101)
FULL_RMSD_THRESHOLD = 2.0
MBG_RMSD_THRESHOLD = 1.0
MBG_RMSD_THRESHOLD2 = 2.0

METHODS_ALL = ["VINA", "SMINA", "GNINA", "AUTODOCK", "GPDOCK", "METALLODOCK"]
METHODS_AFFINITY = ["VINA", "SMINA", "GNINA", "AUTODOCK", "METALLODOCK"]
DOCKING_METHODS = {"VINA", "SMINA", "GNINA"}

METHOD_COLORS = {
    "VINA": "#ef476f",
    "SMINA": "#06d6a0",
    "GNINA": "#118ab2",
    "AUTODOCK": "#ffd166",
    "GPDOCK": "#7209b7",
    "METALLODOCK": "#8d99ae",
}

# Result CSVs of the external baselines; Vina, smina and GNINA come from FEATURE_TEMPLATE.
RESULT_TEMPLATES = {
    "AUTODOCK": "{method}_{scenario}_FH_id={fh_id}_nrun=100_boxExt=4.0.csv",
    "GPDOCK": "{method}_{scenario}_FH_id={fh_id}_HOH=0.csv",
    "METALLODOCK": "{method}_{scenario}_FH_no_leak_max_plus_4_ball_multi_pose_id={fh_id}.csv",
}
FEATURE_TEMPLATE = "features_test_{method}_{scenario}_FH_id={fh_id}.csv"

AFFINITY_SCORE_COL = {
    "VINA": "free_energy",
    "SMINA": "free_energy",
    "GNINA": "CNNaffinity",
    "AUTODOCK": "free_energy",
    "GPDOCK": "free_energy",
    "METALLODOCK": "mdn_score",
}

UNSEEN_GPDOCK_TEMPLATES = {
    "GPDOCK": "GPDOCK_redocked_FHUNSEEN_ONLY_id={fh_id}_HOH=0.csv",
    "VINA": "VINA_redocked_FHUNSEEN_ONLY_GPDOCK_id={fh_id}_exhaust=32_numModes=100_eRange=20.0.csv",
    "SMINA": "SMINA_redocked_FHUNSEEN_ONLY_GPDOCK_id={fh_id}_exhaust=32_numModes=100_eRange=20.0.csv",
}
UNSEEN_METALLODOCK_TEMPLATES = {
    "METALLODOCK": "METALLODOCK_redocked_FH_no_leak_max_plus_4_ball_multi_pose_UNSEEN_ONLY_id={fh_id}_min_method=uff.csv",
    "VINA": "VINA_redocked_FHUNSEEN_ONLY_METALLODOCK_id={fh_id}_exhaust=32_numModes=100_eRange=20.0.csv",
    "SMINA": "SMINA_redocked_FHUNSEEN_ONLY_METALLODOCK_id={fh_id}_exhaust=32_numModes=100_eRange=20.0.csv",
}

# Feature-CSV column names -> the names used by the metric code below.
COLUMN_RENAMES = {"symm_rmsd": "rmsd", "docking_score": "free_energy"}
NUMERIC_COLS = [
    "pose",
    "rmsd",
    "close_rmsd_3.5_hetero",
    "free_energy",
    "CNNscore",
    "CNNaffinity",
    "minimizedAffinity",
    "mdn_score",
]


# Affinity panels negate scores where lower is better (free energies, including
# AutoDock-GPU's), so that a positive correlation means better scores go with
# stronger binding.
HIGHER_IS_BETTER = ("GNINA", "METALLODOCK")
# Sample standard deviation over the three docking runs.
RUN_SD_DDOF = 1


def affinity_sign(method: str) -> int:
    return 1 if method in HIGHER_IS_BETTER else -1


# ---------------------------------------------------------------------------
# Input configuration
# ---------------------------------------------------------------------------
def make_config(unseen_dir: str | None) -> dict:
    """Return the input directories for each method and analysis."""
    test = dict.fromkeys(METHODS_ALL, ("results", EXTERNAL_DIR))
    test.update(dict.fromkeys(DOCKING_METHODS, ("features", FEATURES_DIR)))
    return {
        "test": test,
        "unseen_dir": unseen_dir or UNSEEN_DIR_DEFAULT,
        "vina_train": f"{FEATURES_DIR}/features_train_VINA_redocked_FH_id=1.csv",
    }


CONFIG: dict = {}


def _apply_comparison_cohort(frame, pairs):
    frame = filter_to_cohort(frame, pairs)
    frame.attrs["evaluation_cohort_pairs"] = set(pairs)
    return frame


def _available_labelled_pairs(frame):
    mask = np.ones(len(frame), dtype=bool)
    for column in ["rmsd", "close_rmsd_3.5_hetero", "pose"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask &= np.isfinite(values) & (values >= (1 if column == "pose" else 0))
    return pair_set(frame.loc[mask])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_csv_robust(path: str, usecols=None) -> pd.DataFrame:
    """Read a pose CSV, keep ligand chain IDs such as 'NA' as text, coerce numbers."""
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    df = df.rename(columns=COLUMN_RENAMES)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_uid(df: pd.DataFrame) -> pd.DataFrame:
    sys_col = "alt_system_id" if "alt_system_id" in df.columns else "system_id"
    df["system_id__ligand_id"] = (
        df[sys_col].astype(str) + "__" + df["ligand_id"].astype(str)
    )
    return df


def load_runs(method: str, scenario: str) -> list[pd.DataFrame]:
    kind, data_dir = CONFIG["test"][method]
    dfs = []
    for fh_id in [1, 2, 3]:
        if kind == "features":
            name = FEATURE_TEMPLATE.format(
                method=method, scenario=scenario, fh_id=fh_id
            )
            usecols = lambda c: (
                c
                in {
                    "system_id",
                    "ligand_id",
                    "pose",
                    "symm_rmsd",
                    "close_rmsd_3.5_hetero",
                    "docking_score",
                    "CNNscore",
                    "CNNaffinity",
                    "minimizedAffinity",
                }
            )
        else:
            name = RESULT_TEMPLATES[method].format(
                method=method, scenario=scenario, fh_id=fh_id
            )
            usecols = None
        matches = glob.glob(os.path.join(data_dir, glob.escape(name)))
        if not matches:
            # A fixed-cohort comparison must not silently average fewer repeats.
            raise FileNotFoundError(f"No file {name} in {data_dir}")
        frame = _add_uid(_read_csv_robust(matches[0], usecols=usecols))
        dfs.append(_apply_comparison_cohort(frame, CONFIG["cohort_pairs"]))
    return dfs


def _load_unseen_runs(templates: dict, method: str) -> list[pd.DataFrame]:
    runs = []
    for fh_id in [1, 2, 3]:
        path = os.path.join(CONFIG["unseen_dir"], templates[method].format(fh_id=fh_id))
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        try:
            df = _read_csv_robust(path)
        except pd.errors.EmptyDataError:
            continue
        if {"system_id", "ligand_id", "pose"} - set(df.columns):
            continue
        runs.append(_add_uid(df))
    return runs


def _filter_baselines_by_target(target_runs, baseline_runs):
    """Filter each baseline run to the systems present in the matching target run."""
    filtered = []
    for i in range(len(target_runs)):
        t_df = target_runs[i]
        b_df = baseline_runs[i] if i < len(baseline_runs) else None
        if t_df is None or t_df.empty or b_df is None or b_df.empty:
            filtered.append(
                pd.DataFrame(columns=b_df.columns)
                if b_df is not None
                else pd.DataFrame()
            )
            continue
        ids = set(t_df["system_id__ligand_id"].unique())
        result = b_df[b_df["system_id__ligand_id"].isin(ids)].copy()
        if "evaluation_cohort_pairs" in t_df.attrs:
            result.attrs["evaluation_cohort_pairs"] = set(
                t_df.attrs["evaluation_cohort_pairs"]
            )
        filtered.append(result)
    return filtered


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calculate_success_rate(
    df,
    unique_identifier="system_id__ligand_id",
    sr_col="rmsd",
    top_n_poses=TOP_N_POSES,
    rmsd_threshold=FULL_RMSD_THRESHOLD,
):
    if missing := {sr_col, "pose", unique_identifier} - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")
    rates = fixed_cohort_success(
        df,
        df.attrs["evaluation_cohort_pairs"],
        sr_col,
        rmsd_threshold,
        tuple(top_n_poses),
    )
    return [rates[k] for k in top_n_poses]


def calculate_mbg_success_rate(df, rmsd_threshold=MBG_RMSD_THRESHOLD):
    sr_col = "close_rmsd_3.5_hetero"
    df = df[df[sr_col].notna()].copy()
    return calculate_success_rate(df, sr_col=sr_col, rmsd_threshold=rmsd_threshold)


def _full_sr(df):
    return calculate_success_rate(df, sr_col="rmsd", rmsd_threshold=FULL_RMSD_THRESHOLD)


def _mbg_sr(df):
    return calculate_mbg_success_rate(df, rmsd_threshold=MBG_RMSD_THRESHOLD)


def _mbg_sr2(df):
    return calculate_mbg_success_rate(df, rmsd_threshold=MBG_RMSD_THRESHOLD2)


def _compute_success_rates_from_runs(runs, sr_func):
    """Mean and SD over runs; ``n`` is the size of the runs' fixed cohort."""
    if not runs or not runs[0].attrs["evaluation_cohort_pairs"]:
        return None, None, 0
    arr = np.array([sr_func(df) for df in runs])
    n = len(runs[0].attrs["evaluation_cohort_pairs"])
    return arr.mean(axis=0), arr.std(axis=0, ddof=RUN_SD_DDOF), n


def _n_mbg(runs) -> int:
    """MBG denominator: the cohort size (pairs without a label count as failures)."""
    return len(runs[0].attrs["evaluation_cohort_pairs"]) if runs else 0


def _compute_cumulative_from_runs(runs, value_col, cutoffs):
    """Top-1 success at every cutoff, per run, over each run's fixed cohort."""
    curves, ns = [], []
    for df in runs:
        if value_col not in df.columns:
            continue
        pairs = df.attrs["evaluation_cohort_pairs"]
        curves.append(
            np.array(
                [
                    fixed_cohort_success(df, pairs, value_col, cutoff, (1,))[1]
                    for cutoff in cutoffs
                ]
            )
        )
        ns.append(len(pairs))
    if not curves:
        return None, None, 0
    arr = np.array(curves)
    return (
        arr.mean(axis=0),
        arr.std(axis=0, ddof=RUN_SD_DDOF),
        int(round(np.mean(ns))),
    )


# ---------------------------------------------------------------------------
# Leakage analyses
# ---------------------------------------------------------------------------
def _build_gnina_leak_sets():
    crossdocked_df = pd.read_csv(CROSSDOCKED_TRAIN_CSV)
    reps = pd.read_csv(SPLIT_CLUSTERS_CSV)
    dataset_df = pd.read_csv(DATASET_NEW_CSV, usecols=["system_id", "split"])
    reps["pdb_id"] = reps["system_id"].astype(str).str.split("__").str[0]
    reps_test = reps[reps["split"] == "test"].copy()
    test_ids = set(
        dataset_df.loc[dataset_df["split"] == "test", "system_id"].astype(str)
    )
    cd_pdbs = set(crossdocked_df["pdb_id"].astype(str).str.lower())
    reps_test["pdb_id_lower"] = reps_test["pdb_id"].astype(str).str.lower()
    leaked = reps_test[reps_test["pdb_id_lower"].isin(cd_pdbs)]
    leaked_clusters = set(leaked["cluster"].astype(str))
    leaked_all = reps_test[reps_test["cluster"].astype(str).isin(leaked_clusters)]
    print(f"  Leaked systems (PDB ID match): {leaked['pdb_id'].nunique()}")
    print(f"  Leaked systems (cluster level): {leaked_all['pdb_id'].nunique()}")
    return (
        set(leaked["system_id"].astype(str)) & test_ids,
        set(leaked_all["system_id"].astype(str)) & test_ids,
    )


def _filter_runs_excluding(runs, exclude_ids):
    filtered = []
    for frame in runs:
        result = (
            frame[~frame["system_id"].astype(str).isin(exclude_ids)].copy()
            if "system_id" in frame
            else pd.DataFrame(columns=frame.columns)
        )
        if "evaluation_cohort_pairs" in frame.attrs:
            result.attrs["evaluation_cohort_pairs"] = {
                pair
                for pair in frame.attrs["evaluation_cohort_pairs"]
                if pair[0] not in exclude_ids
            }
        filtered.append(result)
    return filtered


def _build_gnina_series():
    gnina = load_runs("GNINA", "redocked")
    vina = load_runs("VINA", "redocked")
    smina = load_runs("SMINA", "redocked")
    leaked_pdb, leaked_cluster = _build_gnina_leak_sets()
    g_pdb = _filter_runs_excluding(gnina, leaked_pdb)
    g_cl = _filter_runs_excluding(gnina, leaked_cluster)
    C = METHOD_COLORS
    return [
        {"name": "GNINA (all test)", "runs": gnina, "color": C["GNINA"], "ls": "-"},
        {
            "name": "VINA (all test matched)",
            "runs": _filter_baselines_by_target(gnina, vina),
            "color": C["VINA"],
            "ls": "-",
        },
        {
            "name": "SMINA (all test matched)",
            "runs": _filter_baselines_by_target(gnina, smina),
            "color": C["SMINA"],
            "ls": "-",
        },
        {"name": "GNINA (no PDB leak)", "runs": g_pdb, "color": C["GNINA"], "ls": "--"},
        {
            "name": "VINA (no PDB leak matched)",
            "runs": _filter_baselines_by_target(g_pdb, vina),
            "color": C["VINA"],
            "ls": "--",
        },
        {
            "name": "SMINA (no PDB leak matched)",
            "runs": _filter_baselines_by_target(g_pdb, smina),
            "color": C["SMINA"],
            "ls": "--",
        },
        {
            "name": "GNINA (no cluster leak)",
            "runs": g_cl,
            "color": "#073b4c",
            "ls": "--",
        },
        {
            "name": "VINA (no cluster leak matched)",
            "runs": _filter_baselines_by_target(g_cl, vina),
            "color": C["VINA"],
            "ls": ":",
        },
        {
            "name": "SMINA (no cluster leak matched)",
            "runs": _filter_baselines_by_target(g_cl, smina),
            "color": C["SMINA"],
            "ls": ":",
        },
    ]


def _build_unseen_series(method: str, templates: dict):
    unseen = _load_unseen_runs(templates, method)
    u_vina = _filter_baselines_by_target(unseen, _load_unseen_runs(templates, "VINA"))
    u_smina = _filter_baselines_by_target(unseen, _load_unseen_runs(templates, "SMINA"))
    # These are different systems from the curated 156-pair test set. The three
    # methods are compared on one fixed cohort: the pairs labelled in every run.
    populations = [
        _available_labelled_pairs(frame)
        for group in [unseen, u_vina, u_smina]
        for frame in group
    ]
    if len(populations) != 9:
        raise ValueError(
            "A matched unseen comparison requires three repeats for all three methods"
        )
    pairs = set.intersection(*populations)
    if not pairs:
        raise ValueError("The matched unseen comparison has no common labelled pairs")
    CONFIG.setdefault("comparison_cohorts", {})[method + "_unseen"] = sorted(pairs)
    groups = [(method, unseen), ("VINA", u_vina), ("SMINA", u_smina)]
    return [
        {
            "name": name + " (matched unseen only)",
            "runs": [_apply_comparison_cohort(frame, pairs) for frame in group],
            "color": METHOD_COLORS[name],
            "ls": "--",
        }
        for name, group in groups
    ]


def _compute_series_results(series):
    out = []
    for s in series:
        full_mean, full_std, n_full = _compute_success_rates_from_runs(
            s["runs"], _full_sr
        )
        mbg1_mean, mbg1_std, n_mbg1 = _compute_success_rates_from_runs(
            s["runs"], _mbg_sr
        )
        mbg2_mean, mbg2_std, n_mbg2 = _compute_success_rates_from_runs(
            s["runs"], _mbg_sr2
        )
        out.append(
            {
                "name": s["name"],
                "color": s["color"],
                "ls": s["ls"],
                "n": max(n_full, n_mbg1, n_mbg2),
                "full_mean": full_mean,
                "full_std": full_std,
                "mbg1_mean": mbg1_mean,
                "mbg1_std": mbg1_std,
                "mbg2_mean": mbg2_mean,
                "mbg2_std": mbg2_std,
                "n_mbg": _n_mbg(s["runs"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# MBG RMSD by metal (Vina, training set, top-1 pose)
# ---------------------------------------------------------------------------
def compute_mbg_rmsd_by_metal():
    df = _read_csv_robust(
        CONFIG["vina_train"],
        usecols=["system_id", "ligand_id", "pose", "close_rmsd_3.5_hetero"],
    )
    top1 = df[df["pose"] == 1].copy()
    with open(METAL_LIGAND_JSON_PATH) as f:
        metal_ligand_data = json.load(f)
    sid_to_metal = {
        sid: e["metals"][0][0]
        for sid, e in metal_ligand_data.items()
        if e.get("metals")
    }
    top1["metal"] = top1["system_id"].map(sid_to_metal)
    top1["mbg_val"] = top1["close_rmsd_3.5_hetero"]
    top1 = top1.dropna(subset=["metal", "mbg_val"])
    top_metals = top1["metal"].value_counts().head(10).index.tolist()
    return {
        "top_metals": top_metals,
        "metal_rmsds": {
            m: top1.loc[top1["metal"] == m, "mbg_val"].tolist() for m in top_metals
        },
    }


def _matched_affinity_runs(method_runs, affinity_dict):
    """Match all affinity panels on available top-1 scores in every repeat.

    ``affinity_dict`` is keyed by ``system_id__ligand_id``: ligands of the same
    system can differ in whether PLINDER annotates an affinity.
    """
    prepared = {}
    populations = []
    for method, runs in method_runs.items():
        if len(runs) != 3:
            raise ValueError("Affinity comparisons require three repeats per method")
        frames = []
        for frame in runs:
            keys = canonical_pair_columns(frame)
            score = pd.to_numeric(frame[AFFINITY_SCORE_COL[method]], errors="coerce")
            affinity = pd.to_numeric(
                (keys.system_id + "__" + keys.ligand_id).map(affinity_dict),
                errors="coerce",
            )
            top1 = pd.to_numeric(frame["pose"], errors="coerce") == 1
            valid = top1 & np.isfinite(score) & np.isfinite(affinity)
            selected = keys.loc[valid].copy()
            selected["system_id__ligand_id"] = (
                selected.system_id + "__" + selected.ligand_id
            )
            sign = affinity_sign(method)
            selected["docking_score"] = sign * score.loc[valid]
            selected["affinity"] = affinity.loc[valid]
            if selected.duplicated(["system_id", "ligand_id"]).any():
                raise ValueError("Duplicate top-1 affinity inputs")
            populations.append(pair_set(selected))
            frames.append(selected)
        prepared[method] = frames
    pairs = set.intersection(*populations) if populations else set()
    return {
        method: [filter_to_cohort(frame, pairs) for frame in runs]
        for method, runs in prepared.items()
    }, pairs


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def compute_all_results(skip_metal: bool = False) -> dict:
    results = {
        "top_n_poses": TOP_N_POSES,
        "rmsd_cutoffs": RMSD_CUTOFFS.tolist(),
        "scenarios": {},
        "gnina_unseen": None,
        "gpdock_unseen": None,
        "metallodock_unseen": None,
        "mbg_rmsd_by_metal": None,
    }
    ligand_key = ["system_id", "ligand_instance", "ligand_asym_id"]
    # Identifiers as text: a ligand chain "NA" must not be read as missing.
    id_read = {
        "dtype": dict.fromkeys(ligand_key, str),
        "keep_default_na": False,
        "na_values": {"ligand_binding_affinity": [""]},
    }
    dataset_df = pd.read_csv(DATASET_CSV, usecols=ligand_key, **id_read).merge(
        pd.read_csv(
            DATASET_NEW_CSV,
            usecols=[*ligand_key, "ligand_binding_affinity"],
            low_memory=False,
            **id_read,
        ),
        on=ligand_key,
        how="left",
        validate="one_to_one",
    )
    # One value per ligand: ligands of one system can differ in whether they have
    # an affinity annotation.
    affinity_dict = dict(
        zip(
            dataset_df.system_id
            + "__"
            + dataset_df.ligand_instance
            + "."
            + dataset_df.ligand_asym_id,
            dataset_df.ligand_binding_affinity,
            strict=True,
        )
    )

    for scenario in ["redocked", "apo_docked"]:
        print(f"\n-- {scenario} --")
        sr_data, cum_data, aff_data = {}, {}, {}
        all_runs = {method: load_runs(method, scenario) for method in METHODS_ALL}
        affinity_runs, affinity_pairs = _matched_affinity_runs(
            {method: all_runs[method] for method in METHODS_AFFINITY}, affinity_dict
        )
        CONFIG["comparison_cohorts"]["affinity_" + scenario] = sorted(affinity_pairs)
        for method in METHODS_ALL:
            print(f"  {method}")
            runs = all_runs[method]
            full_mean, full_std, n = _compute_success_rates_from_runs(runs, _full_sr)
            mbg_mean, mbg_std, _ = _compute_success_rates_from_runs(runs, _mbg_sr)
            sr_data[method] = {
                "full_mean": full_mean,
                "full_std": full_std,
                "mbg_mean": mbg_mean,
                "mbg_std": mbg_std,
                "n": n,
                "n_mbg": _n_mbg(runs),
                "n_available": float(np.mean([len(pair_set(frame)) for frame in runs])),
                "n_labelled_available": float(
                    np.mean([len(_available_labelled_pairs(frame)) for frame in runs])
                ),
            }
            cf_mean, cf_std, _ = _compute_cumulative_from_runs(
                runs, "rmsd", RMSD_CUTOFFS
            )
            cm_mean, cm_std, _ = _compute_cumulative_from_runs(
                runs, "close_rmsd_3.5_hetero", RMSD_CUTOFFS
            )
            cum_data[method] = {
                "full_mean": cf_mean,
                "full_std": cf_std,
                "mbg_mean": cm_mean,
                "mbg_std": cm_std,
                "n": n,
            }

            if method not in METHODS_AFFINITY:
                continue
            sign = affinity_sign(method)
            run_dfs = affinity_runs[method]
            if not run_dfs or len(run_dfs[0]) < 2:
                continue
            mean_df = (
                pd.concat(run_dfs)
                .groupby("system_id__ligand_id")
                .agg(
                    docking_score=("docking_score", "mean"),
                    affinity=("affinity", "first"),
                )
                .reset_index()
            )
            pr, pp = scipy_stats.pearsonr(mean_df["docking_score"], mean_df["affinity"])
            sr, sp = scipy_stats.spearmanr(
                mean_df["docking_score"], mean_df["affinity"]
            )
            xlabel = {"GNINA": "CNNaffinity", "METALLODOCK": "MDN score"}.get(
                method, r"$-$Docking score" if sign < 0 else "Docking score"
            )
            aff_data[method] = {
                "scores": mean_df["docking_score"].tolist(),
                "affinities": mean_df["affinity"].tolist(),
                "n": len(mean_df),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "xlabel": xlabel,
            }
        results["scenarios"][scenario] = {
            "success_rates": sr_data,
            "cumulative": cum_data,
            "affinity": aff_data,
        }

    print("\n-- Leakage analyses --")
    results["gnina_unseen"] = _compute_series_results(_build_gnina_series())
    results["gpdock_unseen"] = _compute_series_results(
        _build_unseen_series("GPDOCK", UNSEEN_GPDOCK_TEMPLATES)
    )
    results["metallodock_unseen"] = _compute_series_results(
        _build_unseen_series("METALLODOCK", UNSEEN_METALLODOCK_TEMPLATES)
    )
    if not skip_metal:
        print("\n-- MBG RMSD by metal --")
        results["mbg_rmsd_by_metal"] = compute_mbg_rmsd_by_metal()
    return results


def portable_artifact_path(path: str, *, source_code: bool = False) -> str:
    """Describe an input location without storing a machine-specific path."""
    relative = os.path.relpath(os.path.abspath(path), REPO)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return "<input>/" + os.path.basename(path)
    prefix = "<repository>" if source_code else "<Zenodo record>/working_package"
    return prefix + "/" + relative.replace(os.sep, "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cohort",
        required=True,
        help="Fixed common system/ligand cohort for the main-test comparisons",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--unseen_dir",
        default=None,
        help="Unseen-system CSV directory (default: unseen_systems/ next to this script).",
    )
    parser.add_argument(
        "--skip_metal",
        action="store_true",
        help="Skip the slow training-set MBG-by-metal panel.",
    )
    args = parser.parse_args()
    CONFIG.update(make_config(args.unseen_dir))
    CONFIG["cohort_pairs"] = read_cohort(args.cohort)
    CONFIG["comparison_cohorts"] = {"main_test": sorted(CONFIG["cohort_pairs"])}
    data = compute_all_results(skip_metal=args.skip_metal)
    data["meta"] = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "test_inputs": {
            m: portable_artifact_path(d) for m, (_, d) in CONFIG["test"].items()
        },
        "unseen_dir": portable_artifact_path(CONFIG["unseen_dir"]),
        "vina_train": portable_artifact_path(CONFIG["vina_train"]),
        "builder": portable_artifact_path(__file__, source_code=True),
        "cohort_source": portable_artifact_path(args.cohort, source_code=True),
        "comparison_cohorts": CONFIG["comparison_cohorts"],
        "cohort_rule": "Fixed common input/label-availability cohort; missing outputs count as failures. Separate unseen populations use matched three-method cohorts.",
    }
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
