"""Evaluate pre-trained CatBoost models on the metalloprotein docking test set.

The test set is docked several times with identical settings but different run ids
(``TEST_RUN_IDS``). Docking is stochastic, so these repeat runs give different poses.
Every metric below is therefore computed once per run and reported as a mean and
standard deviation across runs, rather than as a single number from one run.

Run from the repository root with ``python -m model.test_eval``.
Outputs CSV files to ``model/model_out/``:

``catboost_evaluation_results_{method}.csv``
    Evaluation results for each docking method, one row per repeat docking run.
``interpret_fi_{method}.csv``
    Feature importance (PredictionValuesChange).
``interpret_shap_{method}.csv``
    Mean absolute and signed SHAP values.
``plot_repro_grouped_metrics.csv``
    Flattened aggregate metrics keyed by method / dock-type / cutoff.
``plot_repro_external_baselines.csv``
    External baseline statistics.
``plot_repro_prediction_quality_scatter.csv``
    Raw predicted vs. true MBG RMSD pairs.
``plot_repro_prediction_quality_corr.csv``
    Per-system Pearson correlations.
``plot_repro_prediction_quality_cutoff.csv``
    Cutoff-dependent coverage and retention data.
``plot_repro_manifest.json``
    Index of all exported files.
"""

from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy import stats

from model import model_utils
from model.cohorts import (
    complete_native_cohort,
    filter_to_cohort,
    fixed_cohort_success,
    read_cohort,
)
from model.model_params import (
    BASELINE_RESULTS_HETERO_DIR as _BASELINE_RESULTS_HETERO_DIR,
)
from model.model_params import (
    EXTRA_BASELINE_METHODS,
    FULL_RMSD_THRESHOLD,
    MBG_RMSD_THRESHOLD,
    MODEL_DIR,
    SR_THRESHOLDS,
    TEST_RUN_IDS,
)
from model.model_params import FEATURES_DIR as _FEATURES_DIR
from model.model_utils import calculate_success_rate, get_feature_list


def _read_csv_maybe_split_header(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV file, handling rare files where the header is split across two lines.

    Some docking output CSVs end the header line with a trailing comma and
    continue the header on the next line.  This function detects that pattern
    and merges the two lines before parsing.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame.
    """
    with open(csv_path) as f:
        content = f.read()
    lines = content.split("\n")
    if lines and lines[0].endswith(",") and len(lines) > 1:
        lines[0] = lines[0] + lines.pop(1)
        content = "\n".join(lines)
    return pd.read_csv(io.StringIO(content))


def _make_uid(df: pd.DataFrame) -> pd.Series:
    """
    Build a unique system/ligand identifier series from a DataFrame.

    Uses ``alt_system_id`` when present (e.g. for external baselines that use an
    alternative naming convention), otherwise falls back to ``system_id``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``ligand_id`` column and either ``system_id`` or
        ``alt_system_id``.

    Returns
    -------
    pd.Series
        String series of the form ``"<system_id>__<ligand_id>"``.
    """
    sys_col = "alt_system_id" if "alt_system_id" in df.columns else "system_id"
    return df[sys_col].astype(str) + "__" + df["ligand_id"].astype(str)


def _success_rate_topk(
    df: pd.DataFrame,
    sr_col: str,
    sr_threshold: float,
    top_ks: tuple[int, ...] = (1, 3, 5),
    cohort_pairs: set[tuple[str, str]] | None = None,
) -> dict[int, float]:
    """
    Compute top-k success rates using the existing ``pose`` column for ranking.

    Thin wrapper around ``model_utils.calculate_success_rate`` that builds the
    ``system_id__ligand_id`` key first, since the baseline CSVs read by this module do
    not carry it. Passing ``rerank_col=None`` keeps the ranking the poses already have,
    which is what distinguishes this from the re-ranking call used elsewhere.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``system_id``/``ligand_id``, ``pose``, and ``sr_col`` columns.
        Poses are assumed to be pre-ranked (pose 1 = best).
    sr_col : str
        RMSD column used to define success (value <= ``sr_threshold``).
    sr_threshold : float
        RMSD threshold in Angstrom.
    top_ks : tuple of int, optional
        Top-k values to evaluate.

    Returns
    -------
    dict of int to float
        Mapping from top-k value to fraction of systems succeeding in the top-k.
    """
    if cohort_pairs is not None:
        return fixed_cohort_success(df, cohort_pairs, sr_col, sr_threshold, top_ks)
    df = df.copy()
    df["system_id__ligand_id"] = _make_uid(df)
    rates = calculate_success_rate(
        df,
        rerank_col=None,
        sr_col=sr_col,
        sr_threshold=sr_threshold,
        top_n=list(top_ks),
    )
    return {int(k): float(v) for k, v in rates.items()}


def run_sd(values) -> float:
    """Sample SD (ddof=1) over repeat docking runs, as for the native methods; 0 for one run."""
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def _load_baseline_runs(method: str, dock_key: str) -> list[pd.DataFrame]:
    """
    Load the repeat-run docking CSVs for external baseline methods.

    Looks for up to three run IDs (1, 2, 3) in ``_BASELINE_RESULTS_HETERO_DIR``
    and returns only the files that exist.

    Parameters
    ----------
    method : str
        Baseline method name: ``'AUTODOCK'``, ``'GPDOCK'``, or ``'METALLODOCK'``.
    dock_key : str
        Docking scenario: ``'redocked'`` or ``'crossdocked'``.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per found run file (0-3 entries), in run-id order.
    """
    scenario = "redocked" if dock_key == "redocked" else "apo_docked"
    templates = {
        "AUTODOCK": "AUTODOCK_{scenario}_FH_id={run_id}_nrun=100_boxExt=4.0.csv",
        "GPDOCK": "GPDOCK_{scenario}_FH_id={run_id}_HOH=0.csv",
        "METALLODOCK": "METALLODOCK_{scenario}_FH_no_leak_max_plus_4_ball_multi_pose_id={run_id}.csv",
    }
    data_dir = _BASELINE_RESULTS_HETERO_DIR
    runs = []
    for run_id in TEST_RUN_IDS:
        fname = templates[method].format(scenario=scenario, run_id=run_id)
        p = os.path.join(data_dir, fname)
        if not os.path.exists(p):
            continue
        runs.append(_read_csv_maybe_split_header(p))
    return runs


def load_external_baseline_stats(
    thresholds: list[float],
    top_ks: tuple[int, ...] = (1, 3, 5),
    true_cov_cutoffs: list[float] | None = None,
    cohort_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, dict]:
    """
    Compute mean/std baseline success rates for external methods from docking CSVs.

    Parameters
    ----------
    thresholds : list of float
        RMSD thresholds at which to compute success rates.
    top_ks : tuple of int, optional
        Top-k values for success rate computation.
    true_cov_cutoffs : list of float or None, optional
        Additional cutoffs for true MBG coverage curves.

    Returns
    -------
    dict
        Nested dict: stats[dock_key][method]["rmsd"|"mbg"][threshold][top_k] -> (mean, std).
        stats[dock_key][method]["coverage"] -> (mean_n_systems, std_n_systems).
        stats[dock_key][method]["true_coverage_curve"][cutoff] -> (mean, std)
        (if true_cov_cutoffs given).
    """
    stats_out = {"redocked": {}, "crossdocked": {}}

    for dock_key in ["redocked", "crossdocked"]:
        for method in EXTRA_BASELINE_METHODS:
            runs = _load_baseline_runs(method, dock_key)
            if cohort_pairs is not None and len(runs) != len(TEST_RUN_IDS):
                # A fixed-cohort comparison must not silently average fewer repeats
                # or drop a method.
                raise FileNotFoundError(
                    f"{method} {dock_key}: found {len(runs)} of "
                    f"{len(TEST_RUN_IDS)} repeat result files"
                )
            if not runs:
                print(f"[BASELINE] WARNING: no runs found for {method} {dock_key}")
                continue

            if cohort_pairs is not None:
                runs = [filter_to_cohort(frame, cohort_pairs) for frame in runs]
            method_stats = {"rmsd": {}, "mbg": {}, "coverage": None}

            n_systems = []
            for df in runs:
                if "pose" not in df.columns:
                    continue
                n_systems.append(float(_make_uid(df).nunique()))
            if n_systems:
                arr_n = np.array(n_systems, dtype=float)
                method_stats["coverage"] = (float(arr_n.mean()), run_sd(arr_n))
            if cohort_pairs is not None:
                method_stats["available_outputs"] = method_stats["coverage"]
                method_stats["coverage"] = (float(len(cohort_pairs)), 0.0)

            for thr in thresholds:
                rmsd_per_run = []
                mbg_per_run = []
                for df in runs:
                    if "rmsd" not in df.columns or "pose" not in df.columns:
                        continue
                    rmsd_per_run.append(
                        _success_rate_topk(
                            df, "rmsd", thr, top_ks=top_ks, cohort_pairs=cohort_pairs
                        )
                    )

                    mbg_col = "close_rmsd_3.5_hetero"
                    if mbg_col not in df.columns:
                        continue
                    df_mbg = (
                        df
                        if cohort_pairs is not None
                        else df[df[mbg_col].notna()].copy()
                    )
                    if df_mbg.empty and cohort_pairs is None:
                        continue
                    mbg_per_run.append(
                        _success_rate_topk(
                            df_mbg,
                            mbg_col,
                            thr,
                            top_ks=top_ks,
                            cohort_pairs=cohort_pairs,
                        )
                    )

                if rmsd_per_run:
                    method_stats["rmsd"][thr] = {}
                    for top_k in top_ks:
                        vals = np.array([d[top_k] for d in rmsd_per_run], dtype=float)
                        method_stats["rmsd"][thr][top_k] = (
                            float(vals.mean()),
                            run_sd(vals),
                        )

                if mbg_per_run:
                    method_stats["mbg"][thr] = {}
                    for top_k in top_ks:
                        vals = np.array([d[top_k] for d in mbg_per_run], dtype=float)
                        method_stats["mbg"][thr][top_k] = (
                            float(vals.mean()),
                            run_sd(vals),
                        )

            if true_cov_cutoffs:
                mbg_col = "close_rmsd_3.5_hetero"
                true_cov_curve: dict[float, tuple] = {}
                for cov_cutoff in true_cov_cutoffs:
                    cov_per_run = []
                    avg_per_run = []
                    for df in runs:
                        if mbg_col not in df.columns:
                            continue
                        uid = _make_uid(df)
                        df_tmp = df.copy()
                        if cohort_pairs is not None:
                            values = pd.to_numeric(df_tmp[mbg_col], errors="coerce")
                            df_tmp[mbg_col] = values.where(
                                np.isfinite(values) & (values >= 0)
                            )
                        df_tmp["_uid"] = uid
                        per_sys = df_tmp.groupby("_uid")[mbg_col].min()
                        n_cov = int((per_sys <= cov_cutoff).sum())
                        meets = df_tmp[df_tmp[mbg_col] <= cov_cutoff]
                        avg_p = float(len(meets) / n_cov) if n_cov > 0 else 0.0
                        cov_per_run.append(n_cov)
                        avg_per_run.append(avg_p)
                    if cov_per_run:
                        arr = np.array(cov_per_run, dtype=float)
                        true_cov_curve[float(cov_cutoff)] = (
                            float(arr.mean()),
                            run_sd(arr),
                        )
                method_stats["true_coverage_curve"] = true_cov_curve

            stats_out[dock_key][method] = method_stats

    return stats_out


def prepare_dataset(df_path: str) -> pd.DataFrame:
    """
    Load a feature CSV and remove rows with NaN or infinite docking scores.

    Parameters
    ----------
    df_path : str
        Path to the feature CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``system_id__ligand_id`` column added and
        rows with invalid docking scores removed.
    """
    df = pd.read_csv(df_path)
    df = df.copy()
    df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]

    initial_rows = len(df)
    if "docking_score" in df.columns:
        df = df[df["docking_score"].notna()]
        df = df[np.isfinite(df["docking_score"])]
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(
                f"[INFO] Removed {removed_rows} rows with NaN/inf in docking_score ({initial_rows} -> {len(df)} rows)"
            )

    return df


def generate_model_interpretability_csvs(
    model: CatBoostRegressor,
    method: str,
    target: str,
    feature_names: list[str],
    csv_folder: str,
    top_n: int = 20,
) -> None:
    """
    Compute feature importance and SHAP values for a trained model and save to CSV.

    Uses the combined train+val feature data as the reference pool.  Writes two
    CSVs per method: ``interpret_fi_{method}.csv`` (PredictionValuesChange
    importances) and ``interpret_shap_{method}.csv`` (mean absolute and signed
    SHAP values).

    Parameters
    ----------
    model : CatBoostRegressor
        Trained CatBoost model.
    method : str
        Docking method label used for file naming.
    target : str
        Target RMSD column name (used only to load reference data).
    feature_names : list of str
        Ordered list of feature column names.
    csv_folder : str
        Output directory for the CSV files.
    top_n : int, optional
        Number of top features to include in each CSV.
    """
    train_df, val_df = model_utils.load_train_val_for_method(
        method=method, target=target
    )
    ref_df = pd.concat([train_df, val_df], ignore_index=True)

    missing_cols = [c for c in feature_names if c not in ref_df.columns]
    if missing_cols:
        print(
            f"[INTERPRET] WARNING: {method} missing {len(missing_cols)} features in reference data; "
            "skipping interpretability CSVs"
        )
        return

    X_ref = ref_df[feature_names].copy()
    X_ref = X_ref.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if X_ref.empty:
        print(
            f"[INTERPRET] WARNING: no finite rows available for {method} interpretability"
        )
        return

    pool = Pool(X_ref[feature_names], feature_names=feature_names)

    importances = np.asarray(
        model.get_feature_importance(pool, type="PredictionValuesChange"), dtype=float
    )
    if importances.size == len(feature_names):
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = (
            fi_df.sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        fi_df.insert(0, "method", method)
        fi_path = os.path.join(csv_folder, f"interpret_fi_{method}.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"[INTERPRET] Saved feature importance CSV: {fi_path}")
    else:
        print(f"[INTERPRET] WARNING: unexpected importance size for {method}")

    shap_values = np.asarray(
        model.get_feature_importance(pool, type="ShapValues"), dtype=float
    )
    if shap_values.ndim == 2 and shap_values.shape[1] >= 2:
        shap_feature_vals = shap_values[:, :-1]
        mean_abs_shap = np.nanmean(np.abs(shap_feature_vals), axis=0)
        mean_signed_shap = np.nanmean(shap_feature_vals, axis=0)
        if mean_abs_shap.size == len(feature_names):
            shap_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": mean_abs_shap,
                    "mean_signed_shap": mean_signed_shap,
                }
            )
            shap_df = (
                shap_df.sort_values("mean_abs_shap", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            shap_df.insert(0, "method", method)
            shap_path = os.path.join(csv_folder, f"interpret_shap_{method}.csv")
            shap_df.to_csv(shap_path, index=False)
            print(f"[INTERPRET] Saved SHAP CSV: {shap_path}")
        else:
            print(f"[INTERPRET] WARNING: SHAP attribution size mismatch for {method}")
    else:
        print(
            f"[INTERPRET] WARNING: unexpected SHAP output shape for {method}: {shap_values.shape}"
        )


def _apply_pred_cutoff(
    df: pd.DataFrame,
    pred_cutoff: float,
    keep_best_pose: bool,
) -> pd.DataFrame:
    """
    Filter a pose DataFrame by predicted RMSD cutoff.

    When ``keep_best_pose`` is ``True``, systems where all poses exceed the
    cutoff retain their single best-predicted pose instead of being dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``pred`` column (predicted MBG RMSD) and a
        ``system_id__ligand_id`` grouping column.
    pred_cutoff : float
        Threshold; rows with ``pred > pred_cutoff`` are removed.
    keep_best_pose : bool
        Whether to keep the best-predicted pose when no pose passes the cutoff.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if not keep_best_pose:
        return df[df["pred"] <= pred_cutoff].copy()

    filtered_list = []
    for _, group in df.groupby("system_id__ligand_id"):
        under_cutoff = group[group["pred"] <= pred_cutoff]
        if len(under_cutoff) > 0:
            filtered_list.append(under_cutoff)
        else:
            filtered_list.append(group.loc[group["pred"].idxmin()].to_frame().T)

    if not filtered_list:
        return df.iloc[0:0].copy()
    return pd.concat(filtered_list, ignore_index=True)


def collect_prediction_quality_inputs(
    model: CatBoostRegressor,
    target: str,
    dfs: list[pd.DataFrame],
    features: list[str],
    method: str,
    dock_key: str,
    cutoffs: list[float],
    keep_best_pose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collect the raw prediction-quality data used by the summary figures.

    Runs model prediction on each repeat docking run, then assembles three DataFrames:
    a scatter table (predicted vs. true RMSD), a per-system Pearson correlation
    table, and a per-cutoff retention/coverage statistics table.

    Parameters
    ----------
    model : CatBoostRegressor
        Trained model.
    target : str
        True MBG RMSD column name.
    dfs : list of pd.DataFrame
        One feature DataFrame per repeat docking run, in run-id order.
    features : list of str
        Feature column names.
    method : str
        Docking method label.
    dock_key : str
        Docking scenario label (``'redocked'`` or ``'crossdocked'``).
    cutoffs : list of float
        Predicted RMSD cutoff values for the retention analysis.
    keep_best_pose : bool
        Whether to retain the best-predicted pose when no pose passes a cutoff.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        ``(scatter_df, corr_df, cutoff_df)``
    """
    score_col = model_utils.get_score_col_for_method(method)
    scatter_rows = []
    corr_rows = []
    cutoff_rows = []

    for idx, df_in in enumerate(dfs, start=1):
        df = df_in.copy()
        if score_col in df.columns:
            df = df[df[score_col].notna()]
            df = df[np.isfinite(df[score_col])]

        _true_cov_by_cutoff: dict[float, tuple] = {}
        if target in df.columns:
            _df_tc = df.copy()
            _df_tc["_uid"] = (
                _df_tc["system_id"].astype(str) + "__" + _df_tc["ligand_id"].astype(str)
            )
            for _c in cutoffs:
                _per_sys = _df_tc.groupby("_uid")[target].min()
                _n = int((_per_sys <= _c).sum())
                _meets = _df_tc[_df_tc[target] <= _c]
                _avg = float(len(_meets) / _n) if _n > 0 else 0.0
                _true_cov_by_cutoff[_c] = (_n, _avg)

        df = df[df[target].notna()].copy()
        if df.empty:
            continue

        df["pred"] = model.predict(df[features])
        if "system_id__ligand_id" not in df.columns:
            df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]

        df = model_utils.rerank_poses(df, score_col)

        scatter_part = df[["pred", target, "system_id__ligand_id", "pose"]].copy()
        scatter_part["method"] = method
        scatter_part["dock_key"] = dock_key
        scatter_part["rep_id"] = idx
        scatter_part = scatter_part.rename(columns={target: "true_mbg_rmsd"})
        scatter_rows.append(scatter_part)

        for sys_id, grp in df.groupby("system_id__ligand_id"):
            x = pd.to_numeric(grp["pred"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp[target], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size < 3:
                continue
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue
            r_val, p_val = stats.pearsonr(x, y)
            corr_rows.append(
                {
                    "method": method,
                    "dock_key": dock_key,
                    "rep_id": idx,
                    "system_id__ligand_id": sys_id,
                    "pearson_r": float(r_val),
                    "pearson_p": float(p_val),
                    "n_poses": int(x.size),
                }
            )

        total_systems = int(df["system_id__ligand_id"].nunique())
        avg_poses_all = float(len(df) / total_systems) if total_systems > 0 else np.nan
        successful_systems_total = int(
            (
                df.groupby("system_id__ligand_id")[target].min() <= MBG_RMSD_THRESHOLD
            ).sum()
        )
        for cutoff in cutoffs:
            filtered = _apply_pred_cutoff(df, cutoff, keep_best_pose=keep_best_pose)
            retained = (
                int(filtered["system_id__ligand_id"].nunique())
                if not filtered.empty
                else 0
            )
            if total_systems > 0:
                all_ids = set(df["system_id__ligand_id"].astype(str).unique().tolist())
                retained_ids = (
                    set(filtered["system_id__ligand_id"].astype(str).unique().tolist())
                    if not filtered.empty
                    else set()
                )
                discarded_ids = all_ids - retained_ids

                per_system_true_min = df.groupby("system_id__ligand_id")[target].min()
                has_true_good_pose = per_system_true_min <= MBG_RMSD_THRESHOLD
                correctly_discarded_count = int(
                    sum(
                        (not bool(has_true_good_pose.get(sys_id, False)))
                        for sys_id in discarded_ids
                    )
                )
                n_unsuccessful = total_systems - successful_systems_total
                frac_correctly_discarded = (
                    float(correctly_discarded_count / n_unsuccessful)
                    if n_unsuccessful > 0
                    else np.nan
                )
            else:
                frac_correctly_discarded = np.nan

            if retained > 0:
                avg_poses = float(len(filtered) / retained)
                success_by_system = (
                    filtered.groupby("system_id__ligand_id")[target].min()
                    <= MBG_RMSD_THRESHOLD
                )
                n_success_retained = int(success_by_system.sum())
                if successful_systems_total > 0:
                    frac_retained_success = float(
                        n_success_retained / successful_systems_total
                    )
                else:
                    frac_retained_success = np.nan
            else:
                avg_poses = 0.0
                n_success_retained = 0
                frac_retained_success = np.nan

            if filtered.empty:
                theoretical_sr = 0.0
                ideal_retained = 0
            else:
                ideal_retained = int(filtered["system_id__ligand_id"].nunique())
                n_retained_with_good_pose = int(
                    (
                        filtered.groupby("system_id__ligand_id")[target].min()
                        <= MBG_RMSD_THRESHOLD
                    ).sum()
                )
                theoretical_sr = float(n_retained_with_good_pose / ideal_retained)

            _tc_n, _tc_avg = _true_cov_by_cutoff.get(cutoff, (np.nan, np.nan))
            cutoff_rows.append(
                {
                    "method": method,
                    "dock_key": dock_key,
                    "rep_id": idx,
                    "rmsd_cutoff": float(cutoff),
                    "systems_total": total_systems,
                    "systems_retained": retained,
                    "avg_poses_kept": avg_poses,
                    "avg_poses_all": avg_poses_all,
                    "successful_systems_total": successful_systems_total,
                    "successful_systems_retained": n_success_retained,
                    "frac_retained_with_success": frac_retained_success,
                    "frac_correctly_discarded": frac_correctly_discarded,
                    "ideal_systems_retained": ideal_retained,
                    "theoretical_sr_top1": theoretical_sr,
                    "true_mbg_coverage": _tc_n,
                    "true_mbg_avg_poses": _tc_avg,
                }
            )

    scatter_df = (
        pd.concat(scatter_rows, ignore_index=True) if scatter_rows else pd.DataFrame()
    )
    corr_df = pd.DataFrame(corr_rows)
    cutoff_df = pd.DataFrame(cutoff_rows)
    return scatter_df, corr_df, cutoff_df


def print_top_system_improvements(
    df: pd.DataFrame, target: str, score_col: str, n: int = 5, label: str = ""
) -> None:
    """
    Print a table of the top-N systems with the largest MBG RMSD improvement after reranking.

    Compares the MBG RMSD of the baseline top-1 pose (best docking score) with the
    pose that has the lowest predicted MBG RMSD, and prints the difference. This is a
    diagnostic of the model's ranking only. It is not CHELATE's filtered pick, which is
    the best-scored pose among those predicted below a cutoff (see ``evaluate_model``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of docked poses that has already been scored with a ``pred``
        column from the model.  Must contain ``system_id__ligand_id``,
        ``score_col``, ``pred``, and ``target`` columns.
    target : str
        True MBG RMSD column name.
    score_col : str
        Docking score column used to select the baseline best pose.
    n : int, optional
        Number of top-improvement systems to print.
    label : str, optional
        Label string appended to the section header for identification.
    """
    uid = "system_id__ligand_id"
    results = []

    for sys_lig, grp in df.groupby(uid):
        if score_col == "CNNscore":
            baseline_row = grp.loc[grp[score_col].idxmax()]
        else:
            baseline_row = grp.loc[grp[score_col].idxmin()]

        model_row = grp.loc[grp["pred"].idxmin()]

        baseline_rmsd = baseline_row[target]
        model_rmsd = model_row[target]
        improvement = baseline_rmsd - model_rmsd

        selected_pose = int(model_row["pose"]) if "pose" in model_row.index else -1

        system_id = (
            baseline_row["system_id"]
            if "system_id" in baseline_row.index
            else sys_lig.split("__")[0]
        )
        ligand_id = (
            baseline_row["ligand_id"]
            if "ligand_id" in baseline_row.index
            else sys_lig.split("__")[-1]
        )

        results.append(
            {
                "system_id": system_id,
                "ligand_id": ligand_id,
                "baseline_MBG": round(float(baseline_rmsd), 3),
                "model_MBG": round(float(model_rmsd), 3),
                "improvement": round(float(improvement), 3),
                "selected_pose": selected_pose,
            }
        )

    if not results:
        print(f"  [{label}] No results to report.")
        return

    res_df = pd.DataFrame(results).sort_values("improvement", ascending=False)

    header = (
        f"{'=' * 60}\n  Top-{n} system improvements, lowest-predicted pose vs "
        f"docking-score top-1 {label}\n{'=' * 60}"
    )
    print(header)
    print(
        f"  {'system_id':<36} {'lig':<8} {'pose':>5}  "
        f"{'base MBG':>9}  {'pred-min MBG':>12}  {'Δ MBG':>8}"
    )
    print(f"  {'-' * 36} {'-' * 8} {'-' * 5}  {'-' * 9}  {'-' * 12}  {'-' * 8}")
    for _, r in res_df.head(n).iterrows():
        print(
            f"  {r['system_id']:<36} {r['ligand_id']:<8} {r['selected_pose']:>5}  "
            f"{r['baseline_MBG']:>9.3f}  {r['model_MBG']:>12.3f}  "
            f"{r['improvement']:>+8.3f}"
        )
    print()


def evaluate_model(
    model: CatBoostRegressor,
    target: str,
    dfs: list[pd.DataFrame],
    features: list[str],
    cutoffs: list[float],
    method: str,
    output_folder: str,
    keep_best_pose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a model over prediction cutoffs, once per repeat docking run.

    Parameters
    ----------
    model : CatBoostRegressor
        Trained model.
    target : str
        Target column name (e.g. ``'close_rmsd_3.5_hetero'``).
    dfs : list of pd.DataFrame
        One feature DataFrame per repeat docking run, in run-id order.
    features : list of str
        Feature column names.
    cutoffs : list of float
        Prediction cutoff values to sweep.
    method : str
        Docking method label (e.g. ``'VINA_redocked'``).
    output_folder : str
        Folder for CSV output.
    keep_best_pose : bool, optional
        If True, keep best-predicted pose per system when none meet the cutoff.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics per cutoff and per repeat docking run.
    """
    records = []
    for idx, df in enumerate(dfs):
        df = df.copy()

        score = "CNNscore" if "GNINA" in method else "docking_score"

        initial_rows = len(df)
        if score in df.columns:
            df = df[df[score].notna()]
            df = df[np.isfinite(df[score])]
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                print(
                    f"[{method}] Rep {idx + 1}: Removed {removed_rows} rows with NaN/inf in {score} ({initial_rows} -> {len(df)} rows)"
                )

        # Drop poses without a metal-binding-group label (e.g. no ligand N/O/S within
        # 3.5 A of a metal, or no symmetry-corrected mapping) *before* computing the
        # baseline, so the baseline and CHELATE are scored on the same poses and systems.
        df = df[df[target].notna()].copy()
        all_ids = df["system_id__ligand_id"].unique().tolist()

        baselines_sr = {}
        baselines_MBG = {}
        for threshold in SR_THRESHOLDS:
            baselines_sr[threshold] = calculate_success_rate(
                df, rerank_col=score, sr_col="symm_rmsd", sr_threshold=threshold
            )
            baselines_MBG[threshold] = calculate_success_rate(
                df, rerank_col=score, sr_col=target, sr_threshold=threshold
            )

        df["pred"] = model.predict(df[features])

        _score_col = "CNNscore" if "GNINA" in method else "docking_score"
        if _score_col in df.columns:
            print_top_system_improvements(
                df, target, score_col=_score_col, n=5, label=f"{method} rep {idx + 1}"
            )

        binding_stats_base = model_utils.get_binding_affinity_stats(
            model_utils.rerank_poses(df, score), method
        )

        for cutoff in cutoffs:
            if keep_best_pose:
                filtered_list = []
                for _system_id, group in df.groupby("system_id__ligand_id"):
                    under_cutoff = group[group["pred"] <= cutoff]
                    if len(under_cutoff) > 0:
                        filtered_list.append(under_cutoff)
                    else:
                        best_pose = group.loc[group["pred"].idxmin()].to_frame().T
                        filtered_list.append(best_pose)
                filtered = pd.concat(filtered_list, ignore_index=True)
            else:
                filtered = df[df["pred"] <= cutoff]

            unique_systems_count = filtered["system_id__ligand_id"].nunique()
            if unique_systems_count < 5:
                print(
                    f"[{method}] Rep {idx + 1}: Skipping cutoff={cutoff} — only {unique_systems_count} unique systems after filtering (need >= 5)"
                )
                continue

            # Re-rank the surviving poses so that pose 1 is CHELATE's pick, not the
            # docking program's original top pose (which may have been filtered out).
            binding_stats = model_utils.get_binding_affinity_stats(
                model_utils.rerank_poses(filtered, score), method
            )

            count = filtered["system_id__ligand_id"].nunique()
            sr1s_sr = {}
            sr1s_MBG = {}
            sr1s_MBG_all = {}
            for threshold in SR_THRESHOLDS:
                # Denominator = every system, i.e. discarded systems count as failures.
                sr1s_MBG_all[threshold] = model_utils.calculate_success_rate_static(
                    filtered,
                    all_ids=all_ids,
                    rerank_col=score,
                    sr_col=target,
                    sr_threshold=threshold,
                    top_n=[1, 3, 5],
                )
                sr1s_sr[threshold] = calculate_success_rate(
                    filtered,
                    rerank_col=score,
                    sr_col="symm_rmsd",
                    sr_threshold=threshold,
                )
                sr1s_MBG[threshold] = calculate_success_rate(
                    filtered, rerank_col=score, sr_col=target, sr_threshold=threshold
                )

            def safe_float(val: object) -> float:
                try:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return np.nan
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            curr = {
                "rep_id": idx + 1,
                "rmsd_cutoff": cutoff,
                "systems_count": count,
                "total_systems": df["system_id__ligand_id"].nunique(),
                "baseline_affinity_systems_count": safe_float(binding_stats_base["N"]),
                "baseline_affinity_Pearson_R": safe_float(
                    binding_stats_base["Pearson_R"]
                ),
                "baseline_affinity_P_value": safe_float(binding_stats_base["P_value"]),
                "baseline_affinity_Kendall_Tau": safe_float(
                    binding_stats_base["Kendall_Tau"]
                ),
                "baseline_affinity_Kendall_P_value": safe_float(
                    binding_stats_base["Kendall_P_value"]
                ),
                "baseline_affinity_RMSE": safe_float(binding_stats_base["RMSE"]),
                "affinity_systems_count": safe_float(binding_stats["N"]),
                "affinity_Pearson_R": safe_float(binding_stats["Pearson_R"]),
                "affinity_P_value": safe_float(binding_stats["P_value"]),
                "affinity_Kendall_Tau": safe_float(binding_stats["Kendall_Tau"]),
                "affinity_Kendall_P_value": safe_float(
                    binding_stats["Kendall_P_value"]
                ),
                "affinity_RMSE": safe_float(binding_stats["RMSE"]),
            }
            for threshold_ in SR_THRESHOLDS:
                for top_k in [1, 3, 5]:
                    sr_val = sr1s_sr[threshold_][top_k]
                    baseline_val = baselines_sr[threshold_][top_k]
                    sr_MBG_val = sr1s_MBG[threshold_][top_k]
                    baseline_MBG_val = baselines_MBG[threshold_][top_k]

                    curr[f"sr_top{top_k}_RMSD={threshold_}"] = sr_val
                    curr[f"baseline_sr_top{top_k}_RMSD={threshold_}"] = baseline_val
                    curr[f"improvement_top{top_k}_RMSD={threshold_}"] = (
                        sr_val - baseline_val
                    )
                    curr[f"MBG_sr_top{top_k}_RMSD={threshold_}"] = sr_MBG_val
                    curr[f"baseline_MBG_sr_top{top_k}_RMSD={threshold_}"] = (
                        baseline_MBG_val
                    )
                    curr[f"improvement_top{top_k}_MBG_RMSD={threshold_}"] = (
                        sr_MBG_val - baseline_MBG_val
                    )
                    curr[f"MBG_sr_all_systems_top{top_k}_RMSD={threshold_}"] = (
                        sr1s_MBG_all[threshold_][top_k]
                    )

            records.append(curr)

    result_df = pd.DataFrame(records)

    non_numeric_cols = ["dock_type", "method"]
    for col in result_df.columns:
        if col not in non_numeric_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    return result_df


def aggregate_results_for_plots(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-run evaluation results into per-cutoff mean/std tables.

    Groups ``results_df`` by ``rmsd_cutoff`` and computes mean and std for all
    metric columns, producing a grouped DataFrame with MultiIndex columns
    suitable for plotting.

    Parameters
    ----------
    results_df : pd.DataFrame
        Per-run evaluation DataFrame returned by ``evaluate_model``.

    Returns
    -------
    pd.DataFrame
        Grouped DataFrame indexed by ``rmsd_cutoff`` with MultiIndex columns
        ``(metric, stat)`` where stat is ``'mean'`` or ``'std'``.
    """
    agg_dict = {
        "systems_count": ["mean", "std"],
        "total_systems": "mean",
        "baseline_affinity_systems_count": ["mean", "std"],
        "baseline_affinity_Pearson_R": ["mean", "std"],
        "baseline_affinity_P_value": ["mean", "std"],
        "baseline_affinity_Kendall_Tau": ["mean", "std"],
        "baseline_affinity_Kendall_P_value": ["mean", "std"],
        "baseline_affinity_RMSE": ["mean", "std"],
        "affinity_systems_count": ["mean", "std"],
        "affinity_Pearson_R": ["mean", "std"],
        "affinity_P_value": ["mean", "std"],
        "affinity_Kendall_Tau": ["mean", "std"],
        "affinity_Kendall_P_value": ["mean", "std"],
        "affinity_RMSE": ["mean", "std"],
    }
    for threshold in SR_THRESHOLDS:
        for top_k in [1, 3, 5]:
            agg_dict.update(
                {
                    f"sr_top{top_k}_RMSD={threshold}": ["mean", "std"],
                    f"baseline_sr_top{top_k}_RMSD={threshold}": ["mean", "std"],
                    f"improvement_top{top_k}_RMSD={threshold}": ["mean", "std"],
                    f"MBG_sr_top{top_k}_RMSD={threshold}": ["mean", "std"],
                    f"baseline_MBG_sr_top{top_k}_RMSD={threshold}": ["mean", "std"],
                    f"improvement_top{top_k}_MBG_RMSD={threshold}": ["mean", "std"],
                    f"MBG_sr_all_systems_top{top_k}_RMSD={threshold}": [
                        "mean",
                        "std",
                    ],
                }
            )
    return results_df.groupby("rmsd_cutoff").agg(agg_dict)


def _flatten_grouped_df(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a grouped DataFrame with MultiIndex columns to single-level CSV-friendly names.

    Tupled column names ``(metric, stat)`` are joined as ``metric__stat``.  The
    columns can be split back on ``__``.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        DataFrame produced by ``aggregate_results_for_plots`` with a MultiIndex
        column level and a ``rmsd_cutoff`` index.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with string column names and ``rmsd_cutoff`` as an
        ordinary column.
    """
    out = grouped_df.copy().reset_index()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            metric, stat = col
            if metric == "rmsd_cutoff":
                flat_cols.append("rmsd_cutoff")
            elif stat:
                flat_cols.append(f"{metric}__{stat}")
            else:
                flat_cols.append(str(metric))
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    return out


def export_plot_repro_csvs(
    all_grouped: dict[str, dict[str, pd.DataFrame]],
    external_baselines: dict[str, dict],
    prediction_quality_inputs: dict[str, pd.DataFrame],
    csv_folder: str,
) -> dict[str, str]:
    """
    Export all plot-input data to CSVs so figures can be reproduced offline.

    Writes the following files into ``csv_folder`` and returns their paths in a
    manifest dict:
    - ``plot_repro_grouped_metrics.csv``
    - ``plot_repro_external_baselines.csv``
    - ``plot_repro_prediction_quality_scatter.csv``
    - ``plot_repro_prediction_quality_corr.csv``
    - ``plot_repro_prediction_quality_cutoff.csv``
    - ``plot_repro_manifest.json``

    Parameters
    ----------
    all_grouped : dict
        method -> dock_key -> grouped DataFrame, as returned by
        ``aggregate_results_for_plots``.
    external_baselines : dict
        Nested stats dict for external baseline methods, as returned by
        ``load_external_baseline_stats``.
    prediction_quality_inputs : dict
        Dict with keys ``'scatter'``, ``'corr'``, ``'cutoff'`` pointing to the
        corresponding DataFrames.
    csv_folder : str
        Output directory; created if it does not exist.

    Returns
    -------
    dict
        Mapping from logical key to absolute file path for each saved file.
    """
    os.makedirs(csv_folder, exist_ok=True)
    saved: dict[str, str] = {}

    grouped_rows = []
    for method, dock_map in all_grouped.items():
        for dock_key, grouped_df in dock_map.items():
            flat = _flatten_grouped_df(grouped_df)
            flat.insert(0, "dock_key", dock_key)
            flat.insert(0, "method", method)
            grouped_rows.append(flat)
    grouped_all = (
        pd.concat(grouped_rows, ignore_index=True) if grouped_rows else pd.DataFrame()
    )
    grouped_path = os.path.join(csv_folder, "plot_repro_grouped_metrics.csv")
    grouped_all.to_csv(grouped_path, index=False)
    saved["grouped_metrics"] = grouped_path

    baseline_rows = []
    for dock_key, method_map in external_baselines.items():
        for method, stat_map in method_map.items():
            coverage = stat_map.get("coverage")
            if coverage is not None:
                baseline_rows.append(
                    {
                        "dock_key": dock_key,
                        "method": method,
                        "metric_family": "coverage",
                        "threshold": np.nan,
                        "top_k": np.nan,
                        "mean": float(coverage[0]),
                        "std": float(coverage[1]),
                    }
                )
            for metric_family in ["mbg", "rmsd"]:
                threshold_map = stat_map.get(metric_family, {})
                for threshold, topk_map in threshold_map.items():
                    for top_k, vals in topk_map.items():
                        baseline_rows.append(
                            {
                                "dock_key": dock_key,
                                "method": method,
                                "metric_family": metric_family,
                                "threshold": float(threshold),
                                "top_k": int(top_k),
                                "mean": float(vals[0]),
                                "std": float(vals[1]),
                            }
                        )
            true_cov_curve = stat_map.get("true_coverage_curve", {})
            for cutoff, vals in true_cov_curve.items():
                baseline_rows.append(
                    {
                        "dock_key": dock_key,
                        "method": method,
                        "metric_family": "true_coverage_curve",
                        "threshold": float(cutoff),
                        "top_k": np.nan,
                        "mean": float(vals[0]),
                        "std": float(vals[1]),
                    }
                )
    baseline_df = pd.DataFrame(baseline_rows)
    baselines_path = os.path.join(csv_folder, "plot_repro_external_baselines.csv")
    baseline_df.to_csv(baselines_path, index=False)
    saved["external_baselines"] = baselines_path

    scatter_df = prediction_quality_inputs.get("scatter", pd.DataFrame())
    corr_df = prediction_quality_inputs.get("corr", pd.DataFrame())
    cutoff_df = prediction_quality_inputs.get("cutoff", pd.DataFrame())

    scatter_path = os.path.join(csv_folder, "plot_repro_prediction_quality_scatter.csv")
    corr_path = os.path.join(csv_folder, "plot_repro_prediction_quality_corr.csv")
    cutoff_path = os.path.join(csv_folder, "plot_repro_prediction_quality_cutoff.csv")

    scatter_df.to_csv(scatter_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    cutoff_df.to_csv(cutoff_path, index=False)

    saved["prediction_quality_scatter"] = scatter_path
    saved["prediction_quality_corr"] = corr_path
    saved["prediction_quality_cutoff"] = cutoff_path

    manifest = {
        "files": saved,
        "notes": {
            "grouped_metrics": "flattened aggregate metrics used in clean summary plots",
            "external_baselines": "baseline-only method stats (coverage/MBG/RMSD)",
            "prediction_quality_scatter": "raw pointwise predicted vs true MBG RMSD",
            "prediction_quality_corr": "per-system Pearson correlations",
            "prediction_quality_cutoff": "cutoff-dependent coverage/retention/theoretical limits",
        },
    }
    manifest_path = os.path.join(csv_folder, "plot_repro_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    saved["manifest"] = manifest_path

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained CatBoost models on the test set"
    )
    parser.add_argument(
        "--keep_best_pose",
        action="store_true",
        help="Keep the best predicted pose per system when none meet the cutoff threshold",
    )
    parser.add_argument(
        "--opt_rmsd_cutoff",
        type=float,
        default=1.0,
        help="RMSD cutoff the models were optimised for (selects the model files).",
    )
    parser.add_argument(
        "--model_dir",
        default=str(MODEL_DIR / "params"),
        help="Directory with {METHOD}_rmsd={cutoff}_best_model.cbm from train_model.py.",
    )
    parser.add_argument(
        "--features_dir",
        default=_FEATURES_DIR,
        help="Directory containing the test feature CSVs from generate_features.py.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(MODEL_DIR / "model_out"),
        help="Directory in which to write evaluation and plot-reproduction CSVs.",
    )

    parser.add_argument(
        "--cohort",
        help="CSV of fixed system_id/ligand_id pairs shared by all methods and repeats; missing external outputs count as failures",
    )
    args = parser.parse_args()
    cohort_pairs = read_cohort(args.cohort) if args.cohort else None
    target = "close_rmsd_3.5_hetero"

    opt_rmsd_cutoff = args.opt_rmsd_cutoff
    methods_to_run = ["VINA", "SMINA", "GNINA"]

    # The MBG RMSD cutoffs at which success rates are evaluated
    rmsd_cutoffs = [
        0.2,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    ]

    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)

    all_grouped_results = {}
    all_results = []
    prediction_quality_parts = {
        "scatter": [],
        "corr": [],
        "cutoff": [],
    }

    for docking_program in methods_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {docking_program}")
        print(f"{'=' * 60}")

        model_path = os.path.join(
            args.model_dir, f"{docking_program}_rmsd={opt_rmsd_cutoff}_best_model.cbm"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[{docking_program}] Trained model not found: {model_path}\n"
                "Run `python -m model.train_model` first to generate it."
            )
        model = CatBoostRegressor()
        model.load_model(model_path)
        print(f"[{docking_program}] Loaded model from {model_path}")

        features = get_feature_list(docking_program=docking_program)
        print(f"[{docking_program}] Using {len(features)} features for evaluation.")

        generate_model_interpretability_csvs(
            model=model,
            method=docking_program,
            target=target,
            feature_names=features,
            csv_folder=output_folder,
        )

        # 3 redocked and 3 cross-docked test sets per docking program
        test_files = {
            "redocked": [
                os.path.join(
                    args.features_dir,
                    f"features_test_{docking_program}_redocked_FH_id={i}.csv",
                )
                for i in TEST_RUN_IDS
            ],
            "crossdocked": [
                os.path.join(
                    args.features_dir,
                    f"features_test_{docking_program}_apo_docked_FH_id={i}.csv",
                )
                for i in TEST_RUN_IDS
            ],
        }

        print(f"[{docking_program}] Loading test datasets...")
        redocked_dfs = [prepare_dataset(f) for f in test_files["redocked"]]
        crossdocked_dfs = [prepare_dataset(f) for f in test_files["crossdocked"]]
        if cohort_pairs is not None:
            redocked_dfs = [
                complete_native_cohort(frame, cohort_pairs, docking_program)
                for frame in redocked_dfs
            ]
            crossdocked_dfs = [
                complete_native_cohort(frame, cohort_pairs, docking_program)
                for frame in crossdocked_dfs
            ]

        for k, rdf in enumerate(redocked_dfs):
            print(
                f"  Redocked {k + 1}: {rdf['system_id'].nunique()} systems, shape {rdf.shape}"
            )

        redocked_dfs = [
            model_utils.add_metal_columns(df, docking_type="redocked_FH")
            for df in redocked_dfs
        ]
        crossdocked_dfs = [
            model_utils.add_metal_columns(df, docking_type="apo_docked_FH")
            for df in crossdocked_dfs
        ]

        method = docking_program

        redocked_results = evaluate_model(
            model,
            target,
            redocked_dfs,
            features,
            rmsd_cutoffs,
            f"{method}_redocked",
            output_folder,
            keep_best_pose=args.keep_best_pose,
        )
        redocked_results["dock_type"] = "Redocked"
        redocked_results["method"] = method

        crossdocked_results = evaluate_model(
            model,
            target,
            crossdocked_dfs,
            features,
            rmsd_cutoffs,
            f"{method}_crossdocked",
            output_folder,
            keep_best_pose=args.keep_best_pose,
        )
        crossdocked_results["dock_type"] = "Cross-docked"
        crossdocked_results["method"] = method

        redocked_grouped = aggregate_results_for_plots(redocked_results)
        crossdocked_grouped = aggregate_results_for_plots(crossdocked_results)

        red_scatter, red_corr, red_cutoff = collect_prediction_quality_inputs(
            model=model,
            target=target,
            dfs=redocked_dfs,
            features=features,
            method=method,
            dock_key="redocked",
            cutoffs=rmsd_cutoffs,
            keep_best_pose=args.keep_best_pose,
        )
        cross_scatter, cross_corr, cross_cutoff = collect_prediction_quality_inputs(
            model=model,
            target=target,
            dfs=crossdocked_dfs,
            features=features,
            method=method,
            dock_key="crossdocked",
            cutoffs=rmsd_cutoffs,
            keep_best_pose=args.keep_best_pose,
        )
        if not red_scatter.empty:
            prediction_quality_parts["scatter"].append(red_scatter)
        if not cross_scatter.empty:
            prediction_quality_parts["scatter"].append(cross_scatter)
        if not red_corr.empty:
            prediction_quality_parts["corr"].append(red_corr)
        if not cross_corr.empty:
            prediction_quality_parts["corr"].append(cross_corr)
        if not red_cutoff.empty:
            prediction_quality_parts["cutoff"].append(red_cutoff)
        if not cross_cutoff.empty:
            prediction_quality_parts["cutoff"].append(cross_cutoff)

        method_results = pd.concat([redocked_results, crossdocked_results])
        all_results.append(method_results)
        all_grouped_results[method] = {
            "redocked": redocked_grouped,
            "crossdocked": crossdocked_grouped,
        }

        csv_path = os.path.join(
            output_folder, f"catboost_evaluation_results_{method}.csv"
        )
        method_results.to_csv(csv_path, index=False)
        print(f"[{docking_program}] Saved evaluation results to {csv_path}")

    external_baselines = load_external_baseline_stats(
        [MBG_RMSD_THRESHOLD, FULL_RMSD_THRESHOLD],
        top_ks=(1, 3, 5),
        true_cov_cutoffs=rmsd_cutoffs,
        cohort_pairs=cohort_pairs,
    )

    prediction_quality_inputs = {
        "scatter": pd.concat(prediction_quality_parts["scatter"], ignore_index=True)
        if prediction_quality_parts["scatter"]
        else pd.DataFrame(),
        "corr": pd.concat(prediction_quality_parts["corr"], ignore_index=True)
        if prediction_quality_parts["corr"]
        else pd.DataFrame(),
        "cutoff": pd.concat(prediction_quality_parts["cutoff"], ignore_index=True)
        if prediction_quality_parts["cutoff"]
        else pd.DataFrame(),
    }

    export_plot_repro_csvs(
        all_grouped=all_grouped_results,
        external_baselines=external_baselines,
        prediction_quality_inputs=prediction_quality_inputs,
        csv_folder=output_folder,
    )
    if cohort_pairs is not None:
        with open(os.path.join(output_folder, "comparison_cohort.json"), "w") as handle:
            json.dump(
                {
                    "source": args.cohort,
                    "pairs": sorted(cohort_pairs),
                    "n_pairs": len(cohort_pairs),
                    "rule": "One fixed input/label-availability cohort for every engine, scenario, repeat and external baseline; missing outputs count as failures.",
                },
                handle,
                indent=2,
            )
    print(f"Saved CSV bundle to {output_folder}")
    print("Done.")


if __name__ == "__main__":
    main()
