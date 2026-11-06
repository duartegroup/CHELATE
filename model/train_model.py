"""
Description
-----------
Train a CatBoostRegressor for each docking program (VINA, SMINA, GNINA) using
Optuna hyperparameter optimisation. The objective is to maximise the improvement in
metal-binding-group (MBG) top-1 success rate over the docking-score baseline on the
validation set, i.e. ``val_sr_MBG_top1 - baseline_sr_val``. After optimisation the best
model is retrained on the combined train+val set and saved together with its
hyperparameters to ``model/params/``. Run from the repository root with
``python -m model.train_model``.

Output files per method and RMSD cutoff (written to ``model/params/``):
    ``{METHOD}_rmsd={cutoff}_best_model.cbm``            — CatBoost model
    ``{METHOD}_rmsd={cutoff}_best_hyperparameters.json`` — Optuna best params

The cutoff is part of the filename because a single invocation can sweep several RMSD
cutoffs (``--rmsd_cutoff``, default ``[1.0]``), each of which defines a different
success-rate objective and therefore yields a different model.
"""

from __future__ import annotations

import os
import sys

# CUDA_VISIBLE_DEVICES must be set before CatBoost is imported, so --device is read
# straight from sys.argv here rather than after argparse runs. The imports below are
# therefore deliberately not at the top of the file (ruff: E402).
_device_idx = next((i for i, a in enumerate(sys.argv) if a == "--device"), None)
if _device_idx is not None and _device_idx + 1 < len(sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_device_idx + 1]
    print(f"[INFO] CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

import argparse
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from model import model_utils
from model.model_params import (
    DOCK_TYPE_SLUG,
    FEATURE_CSV_TEMPLATE,
    FEATURES_DIR,
    MODEL_DIR,
    TEST_RUN_IDS,
    TRAIN_RUN_ID,
    TRAIN_RUN_TYPE,
)
from model.prediction_cutoffs import cutoff_map_tag, load_prediction_cutoffs

if TYPE_CHECKING:
    import optuna

# Set to 'CPU' via --cpu flag in main(); GPU is used when available by default.
_TASK_TYPE: str = "GPU"

# Defaults for the CLI arguments of the same name.
N_TRIALS: int = 500
RMSD_CUTOFFS: list[float] = [1.0]
METHODS: list[str] = ["VINA", "SMINA", "GNINA"]


def train_and_validate(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    target: str,
    rmsd_cutoff: float,
    calculate_sr_fn_val: Callable,
    output_folder: str = str(MODEL_DIR / "model_output"),
    weighting_params: dict[str, float] | None = None,
    docking_program: str = "VINA",
) -> pd.DataFrame:
    """
    Fit a CatBoostRegressor on ``train_df`` and score it on ``val_df``.

    Performs one full fit-and-validate cycle: trains the model, predicts on the
    validation set, filters the predictions by ``rmsd_cutoff``, and computes the
    resulting MBG top-1 success rate and validation MSE.

    Called once per Optuna trial. Writes a results CSV to
    ``{output_folder}/{name}/results_{name}.csv`` and returns the same data as a
    DataFrame.

    Parameters
    ----------
    name : str
        Sub-folder name used for result CSV output (e.g. ``'optuna'``).
    train_df : pandas.DataFrame
        Training feature matrix; must contain all columns from
        ``model_utils.get_feature_list`` plus ``target``.
    val_df : pandas.DataFrame
        Validation feature matrix with the same schema as ``train_df``.
    args : argparse.Namespace
        Must expose ``iterations``, ``learning_rate``, and ``depth``.
    target : str
        Name of the regression target column (e.g. ``'close_rmsd_3.5_hetero'``).
    rmsd_cutoff : float
        Threshold applied to ``catboost_prediction`` when computing the
        filter-based success rate.
    calculate_sr_fn_val : callable
        ``model_utils.get_static_calc_sr`` partial pre-configured for the validation
        set. Called with the filtered DataFrame, it returns success rates keyed by
        top-k. The full validation set is already bound as its fixed denominator, so
        filtering rows out lowers the success rate rather than shrinking the reference.
    output_folder : str, optional
        Root directory for output sub-folders (default ``'model/model_output'``).
    weighting_params : dict of str to float, or None, optional
        If provided, must contain ``smooth_k`` and ``smooth_p`` for
        ``compute_sample_weights``; ``None`` means uniform weights.
    docking_program : str, optional
        ``'GNINA'`` selects ``CNNscore`` as the docking-score feature;
        all others use ``'docking_score'`` (default ``'VINA'``).

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with columns ``rmsd_cutoff``, ``val_sr_MBG_top1``,
        and ``val_mse``.
    """
    features = model_utils.get_feature_list(docking_program=docking_program)

    os.makedirs(f"{output_folder}/{name}", exist_ok=True)

    X_train, y_train = train_df[features], train_df[target]
    train_weights = None
    if weighting_params is not None:
        train_weights = _compute_sample_weights(
            train_df[target].values, mode="smooth", params=weighting_params
        )
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=model_utils.SEED,
        loss_function="RMSE",
        task_type=_TASK_TYPE,
        verbose=False,
    )
    model.fit(X_train, y_train, sample_weight=train_weights)

    val_pred = val_df.copy()
    val_pred["catboost_prediction"] = model.predict(val_df[features])

    val_filt = val_pred[val_pred["catboost_prediction"] <= rmsd_cutoff]
    val_sr = calculate_sr_fn_val(val_filt)[1] if not val_filt.empty else 0

    val_mse = ((val_pred[target] - val_pred["catboost_prediction"]) ** 2).mean()

    df_res = pd.DataFrame(
        [{"rmsd_cutoff": rmsd_cutoff, "val_sr_MBG_top1": val_sr, "val_mse": val_mse}]
    )
    df_res.to_csv(f"{output_folder}/{name}/results_{name}.csv", index=False)
    return df_res


def objective(
    trial: optuna.trial.Trial,
    data_args: tuple[pd.DataFrame, pd.DataFrame, Callable, tuple],
) -> float:
    """
    Evaluate one Optuna trial and return the MBG top-1 success-rate improvement.

    Suggests ``learning_rate``, ``depth``, ``iterations``, ``smooth_k``, and
    ``smooth_p`` for a single Optuna trial, fits and validates the model via
    ``train_and_validate``, and returns the improvement in metal-binding-group top-1
    success rate over the docking-score baseline on the validation set.

    Parameters
    ----------
    trial : optuna.Trial
        Active Optuna trial used to sample hyperparameters.
    data_args : tuple
        ``(train_df, val_df, calculate_sr_fn_val, other_args)`` where ``other_args``
        is ``(target, rmsd_cutoff, output_folder, docking_program, baseline_sr_val)``.

    Returns
    -------
    float
        ``val_sr_MBG_top1 - baseline_sr_val``; Optuna maximises this value.
    """
    lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    iterations = trial.suggest_int("iterations", 100, 1000)

    train_df, val_df, calculate_sr_fn_val, other_args = data_args
    target, rmsd_cutoff, output_folder, docking_program, baseline_sr_val = other_args

    smooth_k = trial.suggest_float("smooth_k", 1e-2, 10.0, log=True)
    smooth_p = trial.suggest_float("smooth_p", 0.5, 3.0)
    weighting_params = {"smooth_k": smooth_k, "smooth_p": smooth_p}

    df_res = train_and_validate(
        name="optuna",
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        args=argparse.Namespace(iterations=iterations, learning_rate=lr, depth=depth),
        target=target,
        rmsd_cutoff=rmsd_cutoff,
        calculate_sr_fn_val=calculate_sr_fn_val,
        output_folder=output_folder,
        weighting_params=weighting_params,
        docking_program=docking_program,
    )

    result = df_res.iloc[0]
    return result["val_sr_MBG_top1"] - baseline_sr_val


def retrain_and_save_final_model(
    best_params: dict[str, float | int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str,
    output_folder: str,
    docking_program: str,
    rmsd_cutoff: float,
) -> CatBoostRegressor:
    """
    Retrain CatBoost on the combined train+val set using the best Optuna hyperparameters
    and the smooth sample weighting they were tuned with, and write the model and
    parameter JSON to ``output_folder``.

    Parameters
    ----------
    best_params : dict
        Optuna best-trial params; must contain ``learning_rate``, ``depth``,
        ``iterations``, ``smooth_k``, and ``smooth_p``.
    train_df : pandas.DataFrame
        Training split.
    val_df : pandas.DataFrame
        Validation split (concatenated with ``train_df`` for final training).
    features : list[str]
        Ordered list of feature column names from ``model_utils.get_feature_list``.
    target : str
        Regression target column name.
    output_folder : str
        Directory where the ``.cbm`` model and ``.json`` params are written.
    docking_program : str
        Docking method name; embedded in output filenames.
    rmsd_cutoff : float
        RMSD cutoff the model was optimised for; embedded in output filenames.
    Returns
    -------
    catboost.CatBoostRegressor
        The trained model.
    """
    combined = pd.concat([train_df, val_df], ignore_index=True)

    # Same smooth weighting the hyperparameters were tuned with in ``objective``.
    train_weights = _compute_sample_weights(
        combined[target].values,
        mode="smooth",
        params={
            "smooth_k": best_params["smooth_k"],
            "smooth_p": best_params["smooth_p"],
        },
    )

    model = CatBoostRegressor(
        iterations=int(best_params["iterations"]),
        learning_rate=float(best_params["learning_rate"]),
        depth=int(best_params["depth"]),
        random_seed=model_utils.SEED,
        loss_function="RMSE",
        task_type=_TASK_TYPE,
        verbose=False,
    )
    model.fit(combined[features], combined[target], sample_weight=train_weights)

    model_path = os.path.join(
        output_folder, f"{docking_program}_rmsd={rmsd_cutoff}_best_model.cbm"
    )
    model.save_model(model_path)
    print(f"[{docking_program}] Model saved to {model_path}")

    params_path = os.path.join(
        output_folder, f"{docking_program}_rmsd={rmsd_cutoff}_best_hyperparameters.json"
    )
    params_to_save = {
        "docking_program": docking_program,
        "rmsd_cutoff": rmsd_cutoff,
        "learning_rate": best_params["learning_rate"],
        "depth": best_params["depth"],
        "iterations": best_params["iterations"],
        "smooth_k": best_params["smooth_k"],
        "smooth_p": best_params["smooth_p"],
    }
    with open(params_path, "w") as f:
        json.dump(params_to_save, f, indent=2)
    print(f"[{docking_program}] Hyperparameters saved to {params_path}")

    return model


def _parse_fraction_grid(fractions_arg: str) -> list[float]:
    """
    Parse a comma-separated string of training-fraction values.

    Parameters
    ----------
    fractions_arg : str
        Comma-separated float values, each in the range ``(0, 1]``.

    Returns
    -------
    list of float
        Sorted, de-duplicated list of valid fraction values.

    Raises
    ------
    ValueError
        If any token cannot be converted to float, falls outside ``(0, 1]``,
        or if no valid fractions are provided.
    """
    vals = []
    for tok in fractions_arg.split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            f = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid fraction value: {s!r}") from exc
        if not (0.0 < f <= 1.0):
            raise ValueError(f"Fraction must be in (0, 1], got {f}")
        vals.append(round(f, 6))
    if not vals:
        raise ValueError("At least one valid fraction is required")
    return sorted(set(vals))


def _compute_sample_weights(
    rmsd_values: np.ndarray, mode: str, params: dict[str, float]
) -> np.ndarray:
    """
    Compute per-sample training weights using either a binary threshold or smooth exponential scheme.

    Parameters
    ----------
    rmsd_values : np.ndarray
        Per-sample MBG RMSD values.
    mode : str
        ``'binary'``: two-class step function; ``'smooth'``: ``exp(-k * |r|^p)`` decay.
    params : dict
        For ``'binary'``: ``weight_threshold``, ``weight_minor``, ``weight_major``.
        For ``'smooth'``: ``smooth_k``, ``smooth_p``.

    Returns
    -------
    np.ndarray
        Per-sample weight array of the same length as ``rmsd_values``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'binary'`` or ``'smooth'``.
    """
    if mode == "binary":
        return np.where(
            rmsd_values > params["weight_threshold"],
            params["weight_minor"],
            params["weight_major"],
        )
    if mode == "smooth":
        return np.exp(-params["smooth_k"] * np.abs(rmsd_values) ** params["smooth_p"])
    raise ValueError(f"Unknown weighting mode: {mode!r}")


def _subset_combined_by_system_fraction(
    combined_df: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    """
    Sample a deterministic subset of systems, retaining all poses for selected systems.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined train+val DataFrame with a ``system_id`` column.
    fraction : float
        Fraction of unique systems to retain, in ``(0, 1]``.  At least one system is kept.
    seed : int
        NumPy default-RNG seed for reproducibility.

    Returns
    -------
    tuple of (pd.DataFrame, int, int)
        ``(subset_df, n_keep, total_systems)``.
    """
    sys_ids = np.array(sorted(combined_df["system_id"].astype(str).unique()))
    total_systems = len(sys_ids)
    n_keep = max(1, int(np.ceil(total_systems * fraction)))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(sys_ids, size=n_keep, replace=False).tolist())
    out_df = combined_df[combined_df["system_id"].astype(str).isin(chosen)].copy()
    return out_df, n_keep, total_systems


def _load_saved_hyperparams(
    method: str, opt_rmsd_cutoff: float, model_dir: str
) -> dict[str, float | int]:
    """
    Load the best Optuna hyperparameters written by ``retrain_and_save_final_model``.

    Parameters
    ----------
    method : str
        Docking method label (e.g. ``'VINA'``).
    opt_rmsd_cutoff : float
        RMSD cutoff the model was optimised for.
    model_dir : str
        Directory containing ``{METHOD}_rmsd={cutoff}_best_hyperparameters.json``.

    Returns
    -------
    dict
        Hyperparameter dictionary loaded from JSON.

    Raises
    ------
    FileNotFoundError
        If the expected JSON file does not exist.
    """
    params_path = os.path.join(
        model_dir, f"{method}_rmsd={opt_rmsd_cutoff}_best_hyperparameters.json"
    )
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Saved hyperparameters not found for {method}: {params_path}"
        )
    with open(params_path) as fh:
        return json.load(fh)


def _train_fraction_model_from_saved_hparams(
    method: str,
    target: str,
    opt_rmsd_cutoff: float,
    model_dir: str,
    fraction: float,
    subset_seed: int,
    features_dir: str = FEATURES_DIR,
    threads: int | None = None,
) -> tuple[CatBoostRegressor, list[str], dict]:
    """
    Evaluate the final model or fit a deterministic train+val fraction.

    At 100%, load the committed final model without fitting. Smaller fractions use
    saved hyperparameters and the smooth sample weighting used in final training.

    Parameters
    ----------
    method : str
        Docking method label (e.g. ``'VINA'``).
    target : str
        Target RMSD column name.
    opt_rmsd_cutoff : float
        RMSD cutoff the model was optimised for.
    model_dir : str
        Directory containing saved hyperparameter JSON files.
    fraction : float
        Fraction of unique training systems to retain, in ``(0, 1]``.
    subset_seed : int
        RNG seed for system sub-sampling.
    features_dir : str, optional
        Directory holding the train/val feature CSVs.

    Returns
    -------
    tuple of (CatBoostRegressor, list of str, dict)
        ``(model, features, fit_meta)`` where ``fit_meta`` records the
        fraction, system counts, and row counts used for training.
    """
    train_df, val_df = model_utils.load_train_val_for_method(
        method=method, target=target, features_dir=features_dir
    )
    combined = pd.concat([train_df, val_df], ignore_index=True)
    subset_df, n_keep, total_systems = _subset_combined_by_system_fraction(
        combined_df=combined,
        fraction=fraction,
        seed=subset_seed,
    )
    features = model_utils.get_feature_list(docking_program=method)
    fit_meta = {
        "fraction": float(fraction),
        "systems_used": int(n_keep),
        "systems_total": int(total_systems),
        "rows_used": len(subset_df),
        "rows_total": len(combined),
    }
    if fraction == 1.0:
        model = CatBoostRegressor()
        model.load_model(
            os.path.join(model_dir, f"{method}_rmsd={opt_rmsd_cutoff}_best_model.cbm")
        )
        return model, features, fit_meta
    best_params = _load_saved_hyperparams(
        method=method, opt_rmsd_cutoff=opt_rmsd_cutoff, model_dir=model_dir
    )
    weights = _compute_sample_weights(
        subset_df[target].values,
        mode="smooth",
        params={
            "smooth_k": float(best_params["smooth_k"]),
            "smooth_p": float(best_params["smooth_p"]),
        },
    )
    model = CatBoostRegressor(
        iterations=int(best_params["iterations"]),
        learning_rate=float(best_params["learning_rate"]),
        depth=int(best_params["depth"]),
        random_seed=model_utils.SEED,
        loss_function="RMSE",
        task_type=_TASK_TYPE,
        verbose=False,
        thread_count=threads,
        allow_writing_files=False,
    )
    model.fit(subset_df[features], subset_df[target], sample_weight=weights)
    return model, features, fit_meta


def _prepare_test_dataset(df_path: str) -> pd.DataFrame:
    """
    Load a feature CSV and remove rows with NaN or infinite docking scores.

    Parameters
    ----------
    df_path : str
        Path to the feature CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``system_id__ligand_id`` column added.
    """
    df = pd.read_csv(df_path)
    df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]
    if "docking_score" in df.columns:
        df = df[df["docking_score"].notna()]
        df = df[np.isfinite(df["docking_score"])]
    return df


def _eval_mbg_sr_at_pred_cutoff(
    model: CatBoostRegressor,
    target: str,
    dfs: list[pd.DataFrame],
    features: list[str],
    method: str,
    pred_cutoff: float,
    mbg_sr_threshold: float,
    keep_best_pose: bool = False,
) -> list[dict]:
    """
    Compute the MBG top-1 success rate per docking run, after filtering by predicted
    RMSD cutoff.

    The test set is docked several times with identical settings but different run ids
    (``TEST_RUN_IDS``). Docking is stochastic, so these repeat runs give different
    poses; evaluating each one separately shows how much the reported success rate
    varies from run to run rather than reporting a single number from one run.

    Parameters
    ----------
    model : CatBoostRegressor
        Trained model used to predict MBG RMSD.
    target : str
        True MBG RMSD column name (e.g. ``'close_rmsd_3.5_hetero'``).
    dfs : list of pd.DataFrame
        One feature DataFrame per repeat docking run, in run-id order.
    features : list of str
        Feature column names passed to ``model.predict``.
    method : str
        Docking method label; determines score column for NaN filtering.
    pred_cutoff : float
        Poses with predicted RMSD above this value are removed.
    mbg_sr_threshold : float
        True RMSD threshold used to declare a pose successful.
    keep_best_pose : bool, optional
        If ``True``, retain the best-predicted pose for systems where all poses
        exceed the cutoff rather than dropping the system entirely.

    Returns
    -------
    list of dict
        One dict per repeat docking run with keys ``rep_id`` (the 1-based index of
        the run), ``mbg_sr_top1``, and ``systems_count``.
    """
    out = []
    score_col = model_utils.get_score_col_for_method(method)
    for idx, df_in in enumerate(dfs, start=1):
        df = df_in.copy()
        if score_col in df.columns:
            df = df[df[score_col].notna()]
            df = df[np.isfinite(df[score_col])]
        df = df[df[target].notna()].copy()
        if df.empty:
            out.append({"rep_id": idx, "mbg_sr_top1": np.nan, "systems_count": 0})
            continue
        df["pred"] = model.predict(df[features])
        if keep_best_pose:
            filtered_list = []
            for _, group in df.groupby("system_id__ligand_id"):
                under_cutoff = group[group["pred"] <= pred_cutoff]
                if len(under_cutoff) > 0:
                    filtered_list.append(under_cutoff)
                else:
                    filtered_list.append(group.loc[group["pred"].idxmin()].to_frame().T)
            filtered = pd.concat(filtered_list, ignore_index=True)
        else:
            filtered = df[df["pred"] <= pred_cutoff].copy()
        sys_count = (
            int(filtered["system_id__ligand_id"].nunique()) if not filtered.empty else 0
        )
        if filtered.empty:
            sr_top1 = 0.0
        else:
            sr_top1 = float(
                model_utils.calculate_success_rate(
                    filtered,
                    rerank_col=score_col,
                    sr_col=target,
                    sr_threshold=mbg_sr_threshold,
                )[1]
            )
        out.append({"rep_id": idx, "mbg_sr_top1": sr_top1, "systems_count": sys_count})
    return out


def _baseline_mbg_sr_per_rep(
    dfs: list[pd.DataFrame],
    method: str,
    target: str,
    mbg_sr_threshold: float,
) -> dict[int, float]:
    """
    Compute the baseline top-1 MBG success rate per repeat docking run, using the
    docking program's own ranking rather than the model's.

    No prediction cutoff is applied; poses are ranked solely by the docking score.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        One feature DataFrame per repeat docking run, in run-id order.
    method : str
        Docking method label; determines the score column used for ranking.
    target : str
        True MBG RMSD column name.
    mbg_sr_threshold : float
        RMSD threshold in Å used to declare a pose successful.

    Returns
    -------
    dict of int to float
        Mapping from 1-based run index to top-1 MBG success rate.
    """
    score_col = model_utils.get_score_col_for_method(method)
    out = {}
    for idx, df_in in enumerate(dfs, start=1):
        df = df_in.copy()
        if score_col in df.columns:
            df = df[df[score_col].notna()]
            df = df[np.isfinite(df[score_col])]
        df = df[df[target].notna()].copy()
        if df.empty:
            out[idx] = np.nan
            continue
        out[idx] = float(
            model_utils.calculate_success_rate(
                df,
                rerank_col=score_col,
                sr_col=target,
                sr_threshold=mbg_sr_threshold,
            )[1]
        )
    return out


def run_train_fraction_sweep(
    output_folder: str,
    methods: list[str],
    target: str,
    opt_rmsd_cutoff: float,
    model_dir: str,
    fractions: list[float],
    pred_cutoff: float | dict[str, float],
    mbg_sr_threshold: float,
    keep_best_pose: bool,
    subset_seed: int,
    features_dir: str = FEATURES_DIR,
    threads: int | None = None,
) -> None:
    """
    Sweep training-data fractions, retrain models, evaluate on the test set, and save results.

    Fractions below 100% fit CatBoost using the saved hyperparameters. The 100%
    point loads the committed final model, so it matches that model's predictions.
    Results are written as a CSV to ``output_folder``.

    Parameters
    ----------
    output_folder : str
        Directory where the output CSV is written.
    methods : list of str
        Docking method labels to sweep (e.g. ``['VINA', 'SMINA', 'GNINA']``).
    target : str
        Target MBG RMSD column name (e.g. ``'close_rmsd_3.5_hetero'``).
    opt_rmsd_cutoff : float
        RMSD cutoff the models were optimised for.
    model_dir : str
        Directory containing saved hyperparameter JSON files.
    fractions : list of float
        Training fractions to evaluate, each in ``(0, 1]``.
    pred_cutoff : float or dict
        Shared prediction cutoff, or one cutoff per engine, applied on the test set.
    mbg_sr_threshold : float
        True RMSD threshold used to define success.
    keep_best_pose : bool
        Whether to retain the best-predicted pose for systems where all poses
        exceed ``pred_cutoff``.
    subset_seed : int
        Base seed for system sub-sampling; perturbed per method and fraction.
    features_dir : str, optional
        Directory holding the test-split feature CSVs. Defaults to
        ``model_params.FEATURES_DIR``.
    """
    os.makedirs(output_folder, exist_ok=True)
    all_records = []

    for method in methods:
        print(f"\n[FRACTION_SWEEP] Method: {method}")
        method_cutoff = (
            pred_cutoff[method] if isinstance(pred_cutoff, dict) else pred_cutoff
        )
        test_files = {
            dock_type: [
                os.path.join(
                    features_dir,
                    FEATURE_CSV_TEMPLATE.format(
                        split="test", method=method, run_type=run_type, run_id=i
                    ),
                )
                for i in TEST_RUN_IDS
            ]
            for dock_type, run_type in DOCK_TYPE_SLUG.items()
        }
        test_sets = {
            "redocked": [
                model_utils.add_metal_columns(
                    _prepare_test_dataset(p), docking_type="redocked_FH"
                )
                for p in test_files["redocked"]
            ],
            "crossdocked": [
                model_utils.add_metal_columns(
                    _prepare_test_dataset(p), docking_type="apo_docked_FH"
                )
                for p in test_files["crossdocked"]
            ],
        }
        baseline_by_dock = {
            dk: _baseline_mbg_sr_per_rep(
                dfs=test_sets[dk],
                method=method,
                target=target,
                mbg_sr_threshold=mbg_sr_threshold,
            )
            for dk in ["redocked", "crossdocked"]
        }

        for fraction in fractions:
            seed = int(
                subset_seed
                + round(fraction * 1000)
                + (10000 * (methods.index(method) + 1))
            )
            print(
                f"[FRACTION_SWEEP] Evaluating {method} with fraction={fraction:.2f} (seed={seed})"
            )
            model, features, fit_meta = _train_fraction_model_from_saved_hparams(
                method=method,
                target=target,
                opt_rmsd_cutoff=opt_rmsd_cutoff,
                model_dir=model_dir,
                fraction=fraction,
                subset_seed=seed,
                features_dir=features_dir,
                threads=threads,
            )
            for dock_key in ["redocked", "crossdocked"]:
                rep_metrics = _eval_mbg_sr_at_pred_cutoff(
                    model=model,
                    target=target,
                    dfs=test_sets[dock_key],
                    features=features,
                    method=method,
                    pred_cutoff=method_cutoff,
                    mbg_sr_threshold=mbg_sr_threshold,
                    keep_best_pose=keep_best_pose,
                )
                for row in rep_metrics:
                    all_records.append(
                        {
                            "method": method,
                            "dock_key": dock_key,
                            "fraction": fraction,
                            "pred_cutoff": method_cutoff,
                            "mbg_sr_threshold": mbg_sr_threshold,
                            "baseline_mbg_sr_top1": baseline_by_dock[dock_key].get(
                                row["rep_id"], np.nan
                            ),
                            "systems_used": fit_meta["systems_used"],
                            "systems_total": fit_meta["systems_total"],
                            "rows_used": fit_meta["rows_used"],
                            "rows_total": fit_meta["rows_total"],
                            "task_type": _TASK_TYPE,
                            **row,
                        }
                    )

    if not all_records:
        print("[FRACTION_SWEEP] WARNING: No records produced.")
        return

    out_df = pd.DataFrame(all_records)
    cutoffs = (
        pred_cutoff
        if isinstance(pred_cutoff, dict)
        else dict.fromkeys(methods, pred_cutoff)
    )
    cutoff_tag = cutoff_map_tag(cutoffs, methods)
    csv_path = os.path.join(
        output_folder, f"mbg_sr_vs_training_fraction_predCutoff={cutoff_tag}.csv"
    )
    out_df.to_csv(csv_path, index=False)
    print(f"[FRACTION_SWEEP] Saved CSV: {csv_path}")

    summary = (
        out_df.groupby(["method", "dock_key", "fraction"], as_index=False)[
            "mbg_sr_top1"
        ]
        .mean()
        .sort_values(["method", "dock_key", "fraction"])
    )
    for method in methods:
        for dock_key in ["redocked", "crossdocked"]:
            s = summary[
                (summary["method"] == method) & (summary["dock_key"] == dock_key)
            ]
            if len(s) < 2:
                continue
            first = float(s.iloc[0]["mbg_sr_top1"])
            last = float(s.iloc[-1]["mbg_sr_top1"])
            print(
                f"[FRACTION_SWEEP] {method} {dock_key}: "
                f"SR({s.iloc[0]['fraction']:.2f})={first:.4f}, "
                f"SR({s.iloc[-1]['fraction']:.2f})={last:.4f}, "
                f"delta={last - first:+.4f}"
            )


def main() -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train CatBoost models with Optuna tuning for all methods"
    )
    parser.add_argument(
        "--target",
        choices=[
            "close_rmsd_3.5_hetero",
            "symm_rmsd",
        ],
        default="close_rmsd_3.5_hetero",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Docking methods to optimise, one model per method.",
    )
    parser.add_argument(
        "--rmsd_cutoff",
        type=float,
        nargs="+",
        default=RMSD_CUTOFFS,
        help=(
            "RMSD cutoffs in Angstrom to optimise for, one model per cutoff. Each "
            "cutoff defines the threshold below which a pose counts as successful, "
            "and therefore a different optimisation objective."
        ),
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default=N_TRIALS,
        help="Number of trials for Optuna optimisation.",
    )
    parser.add_argument(
        "--features_dir",
        default=FEATURES_DIR,
        help="Directory containing the feature CSVs written by generate_features.py.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='CUDA_VISIBLE_DEVICES value (e.g. "0"). for GPU acceleration. If not set, uses all available GPUs.',
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CatBoost to use CPU even when a GPU is available.",
    )
    parser.add_argument(
        "--train_fraction_sweep",
        action="store_true",
        help="After Optuna training, sweep training-data fractions and evaluate on the test set.",
    )
    parser.add_argument(
        "--train_fraction_grid",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated fractions in (0,1] for the fraction sweep.",
    )
    parser.add_argument(
        "--fraction_pred_cutoff",
        type=float,
        default=None,
        help=(
            "Prediction cutoff for the fraction sweep; default: each program's "
            "selected cutoff from model/params/pred_cutoffs.json."
        ),
    )
    parser.add_argument(
        "--fraction_mbg_sr_threshold",
        type=float,
        default=1.0,
        help="MBG RMSD threshold for success rate in fraction sweep.",
    )
    parser.add_argument(
        "--fraction_keep_best_pose",
        action="store_true",
        help="Keep best-predicted pose per system when none meet the cutoff.",
    )
    parser.add_argument(
        "--fraction_subset_seed",
        type=int,
        default=model_utils.SEED,
        help="Base random seed for deterministic subset sampling.",
    )
    parser.add_argument(
        "--sweep_model_dir",
        default=None,
        help="Directory containing saved hyperparameter JSON files for fraction sweep. Defaults to model/params/.",
    )
    args = parser.parse_args()

    global _TASK_TYPE
    if args.cpu:
        _TASK_TYPE = "CPU"

    model_utils.seed()

    methods_to_run = args.method
    cutoffs_to_run = args.rmsd_cutoff
    n_trials = args.ntrials
    features_dir = args.features_dir

    # Models are saved here as {METHOD}_rmsd={cutoff}_best_model.cbm, which is where
    # test_eval.py reads them from. Copy them to chelate/params/ to ship them.
    output_folder = str(MODEL_DIR / "params")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Methods to run : {methods_to_run}")
    print(f"RMSD cutoffs   : {cutoffs_to_run}")

    study_direction = (
        "maximize"  # we want to maximize the improvement in success rate over baseline
    )

    score_col_map = {
        "GNINA": "CNNscore",
        "SMINA": "docking_score",
        "VINA": "docking_score",
    }

    data_by_method = {}
    for docking_program in methods_to_run:
        score_col = score_col_map[docking_program]

        features_train = pd.read_csv(
            os.path.join(
                features_dir,
                FEATURE_CSV_TEMPLATE.format(
                    split="train",
                    method=docking_program,
                    run_type=TRAIN_RUN_TYPE,
                    run_id=TRAIN_RUN_ID,
                ),
            )
        )
        features_val = pd.read_csv(
            os.path.join(
                features_dir,
                FEATURE_CSV_TEMPLATE.format(
                    split="val",
                    method=docking_program,
                    run_type=TRAIN_RUN_TYPE,
                    run_id=TRAIN_RUN_ID,
                ),
            )
        )

        if "docking_score" in features_train.columns:
            features_train = features_train.rename(columns={"docking_score": score_col})
        if "docking_score" in features_val.columns:
            features_val = features_val.rename(columns={"docking_score": score_col})

        print(
            f"[{docking_program}] Train shape: {features_train.shape}, Val shape: {features_val.shape}"
        )
        print(f"[{docking_program}] Adding metal columns...")

        features_train = model_utils.add_metal_columns(
            features_train, docking_type="redocked_FH"
        )
        features_val = model_utils.add_metal_columns(
            features_val, docking_type="redocked_FH"
        )

        for df in [features_train, features_val]:
            df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]

        # Drop invalid target rows
        features_train = features_train[features_train[args.target] >= 0]
        features_val = features_val[features_val[args.target] >= 0]

        print(
            f"[{docking_program}] After filtering — train: {features_train.shape}, val: {features_val.shape}"
        )
        print(
            f"[{docking_program}] Unique systems — train: {features_train['system_id'].nunique()}, "
            f"val: {features_val['system_id'].nunique()}"
        )

        data_by_method[docking_program] = {
            "score_col": score_col,
            "train_df": features_train,
            "val_df": features_val,
            "combined_df": pd.concat([features_train, features_val], ignore_index=True),
        }

    for docking_program in methods_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Docking program: {docking_program}")
        print(f"{'=' * 60}")

        score_col = data_by_method[docking_program]["score_col"]
        features_train = data_by_method[docking_program]["train_df"]
        features_val = data_by_method[docking_program]["val_df"]

        for rmsd_cutoff in cutoffs_to_run:
            print(f"\n  [{docking_program}] RMSD cutoff = {rmsd_cutoff}")

            # Baseline success rate (depends on rmsd_cutoff)
            calculate_MBG_sr_static_val = model_utils.get_static_calc_sr(
                features_val["system_id__ligand_id"].unique().tolist(),
                rerank_col=score_col,
                sr_col=args.target,
                sr_threshold=rmsd_cutoff,
            )
            baseline_sr_val = model_utils.calculate_success_rate(
                features_val,
                rerank_col=score_col,
                sr_col=args.target,
                sr_threshold=rmsd_cutoff,
            )[1]
            baseline_sr_train = model_utils.calculate_success_rate(
                features_train,
                rerank_col=score_col,
                sr_col=args.target,
                sr_threshold=rmsd_cutoff,
            )[1]
            print(
                f"  [{docking_program}] Baseline Val SR  ({args.target}, cutoff={rmsd_cutoff}): {baseline_sr_val:.4f}"
            )
            print(
                f"  [{docking_program}] Baseline Train SR ({args.target}, cutoff={rmsd_cutoff}): {baseline_sr_train:.4f}"
            )

            assert (
                model_utils.calculate_success_rate(
                    features_val,
                    rerank_col=score_col,
                    sr_col=args.target,
                    sr_threshold=rmsd_cutoff,
                )[1]
                == calculate_MBG_sr_static_val(features_val)[1]
            ), "Baseline SR mismatch!"

            data_args = (
                features_train,
                features_val,
                calculate_MBG_sr_static_val,
                (
                    args.target,
                    rmsd_cutoff,
                    output_folder,
                    docking_program,
                    baseline_sr_val,
                ),
            )

            print(
                f"  [{docking_program}] Starting Optuna (Direction={study_direction}, n_trials={n_trials})"
            )
            study = optuna.create_study(
                direction=study_direction,
                sampler=optuna.samplers.TPESampler(seed=model_utils.SEED),
            )
            study.optimize(
                lambda trial, data=data_args: objective(trial, data),
                n_trials=n_trials,
            )

            print(f"  [{docking_program}] Best trial: {study.best_trial.params}")
            print(f"  [{docking_program}] Best value: {study.best_value}")

            features = model_utils.get_feature_list(docking_program=docking_program)
            print(f"  [{docking_program}] Features: {len(features)} columns")

            # Retrain final model on train+val and save
            retrain_and_save_final_model(
                best_params=study.best_trial.params,
                train_df=features_train,
                val_df=features_val,
                features=features,
                target=args.target,
                output_folder=output_folder,
                docking_program=docking_program,
                rmsd_cutoff=rmsd_cutoff,
            )

    print("\nAll optimisations complete.")
    print(f"Best models and hyperparameters saved in folder: {output_folder}")

    if args.train_fraction_sweep:
        print(
            f"Running training fraction sweep with prediction cutoff {args.fraction_pred_cutoff or 'per program'} and MBG SR threshold {args.fraction_mbg_sr_threshold}..."
        )
        fraction_grid = _parse_fraction_grid(args.train_fraction_grid)
        print(f"\n[FRACTION_SWEEP] Fractions: {fraction_grid}")
        sweep_opt_rmsd_cutoff = cutoffs_to_run[0]
        sweep_model_dir = args.sweep_model_dir or output_folder
        run_train_fraction_sweep(
            output_folder=str(MODEL_DIR / "model_out"),
            methods=methods_to_run,
            target=args.target,
            opt_rmsd_cutoff=sweep_opt_rmsd_cutoff,
            model_dir=sweep_model_dir,
            fractions=fraction_grid,
            pred_cutoff=(
                load_prediction_cutoffs()
                if args.fraction_pred_cutoff is None
                else args.fraction_pred_cutoff
            ),
            mbg_sr_threshold=args.fraction_mbg_sr_threshold,
            keep_best_pose=args.fraction_keep_best_pose,
            subset_seed=args.fraction_subset_seed,
            features_dir=features_dir,
        )


if __name__ == "__main__":
    main()
