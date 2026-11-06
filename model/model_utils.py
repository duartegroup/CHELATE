"""
Description
-----------
Shared helpers for the pose re-ranking pipeline.

``add_target_system_column`` and ``get_binding_affinity_stats`` join the curated
PLINDER metadata (`DATASET_CSV`) onto a feature DataFrame. The dataset CSV, the
metal/ligand JSON and the binding-affinity lookup are read once on first use and
cached in module-level globals, so the files are never parsed more than once per
process.

``get_system_metal`` resolves which metals a system contains, and
``add_metal_columns`` turns that into the ``has_{Metal}`` one-hot columns consumed
by the models. ``get_metal_cols`` returns those column names.

``rerank_poses`` re-orders the poses of each system-ligand pair by a scoring
column, and ``calculate_success_rate`` / ``get_static_calc_sr`` compute top-N
success rates at the RMSD thresholds in `SR_THRESHOLDS`.

``get_feature_list`` returns the ordered feature columns expected by the CatBoost
models, matching the schema written by
[generate_features.py](../docking/generate_features.py), and
``load_train_val_for_method`` loads the corresponding train/val CSVs.

Benchmark paths and training parameters come from [model_params.py](model_params.py).
Metal definitions and the feature schema (``get_feature_list``, ``get_metal_cols``,
``get_score_col_for_method``) live in the installable ``chelate`` package. The schema
is re-exported here, so the benchmark scripts and the package share the column list.
"""

import functools
import json
import os
import random
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.metrics import mean_squared_error

from chelate.constants import EXPLICIT_METALS, METAL_ATOMS_SET
from chelate.receptor import extract_metal_symbols_from_pdb

# The model scripts import the feature schema from here.
from chelate.schema import get_feature_list, get_metal_cols, get_score_col_for_method
from model.model_params import DATASET_CSV as _DATASET_CSV
from model.model_params import (
    FEATURE_CSV_TEMPLATE,
    MODEL_DIR,
    SEED,
    TRAIN_RUN_ID,
    TRAIN_RUN_TYPE,
)
from model.model_params import FEATURES_DIR as _FEATURES_DIR
from model.model_params import METAL_LIGAND_DATA_PATH as _METAL_LIGAND_DATA_PATH

_plinder_df: pd.DataFrame | None = None
_metal_ligand_data: dict | None = None
_binding_dict: dict | None = None

# Re-exported for callers that import the schema from here.
__all__ = ["get_feature_list", "get_metal_cols", "get_score_col_for_method"]


def _get_plinder_df() -> pd.DataFrame:
    """
    Return the curated PLINDER dataset, reading it from disk on first call.

    The DataFrame is cached in the module-level ``_plinder_df`` global, so the CSV is
    parsed only once per process. Two convenience columns are added on load:
    ``ligand_id`` (``{ligand_instance}.{ligand_asym_id}``) and the composite key
    ``system_id__ligand_id`` used to group poses throughout this module.

    Returns
    -------
    pd.DataFrame
        Contents of ``DATASET_CSV`` with the two derived columns.
    """
    global _plinder_df
    if _plinder_df is None:
        df = pd.read_csv(_DATASET_CSV, keep_default_na=False, na_values=[""])
        df["ligand_id"] = (
            df["ligand_instance"].astype(str) + "." + df["ligand_asym_id"].astype(str)
        )
        df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]
        _plinder_df = df
    return _plinder_df


def _get_metal_ligand_data() -> dict:
    """
    Return the per-system metal/ligand data, reading it from disk on first call.

    The file is produced by ``generate_metal_ligand_data.py`` in the data curation
    pipeline and records, for every curated system, the metals in the binding site and
    their crystallographic positions. It is the source of metal identities for redocked
    systems in ``get_system_metal``; cross-docked systems are not covered by it and are
    resolved from the receptor PDB instead.

    The parsed JSON is cached in the module-level ``_metal_ligand_data`` global, so the
    file is read only once per process.

    Returns
    -------
    dict
        Mapping of ``system_id`` to its metal data, where each entry has a
        ``"metals"`` list of ``[symbol, [x, y, z]]`` pairs. Systems whose binding site
        contains several metals have one entry per metal.
    """
    global _metal_ligand_data
    if _metal_ligand_data is None:
        with open(_METAL_LIGAND_DATA_PATH) as f:
            _metal_ligand_data = json.load(f)
    return _metal_ligand_data


def _get_binding_dict() -> dict:
    """
    Return the experimental binding-affinity lookup, building it on first call.

    Derived from the curated dataset by pairing its ``system_id__ligand_id`` and
    ``ligand_binding_affinity`` columns, and cached in the module-level
    ``_binding_dict`` global. Used by ``get_binding_affinity_stats`` to correlate
    docking scores against measured affinities.

    Affinities are only available for the subset of systems where PLINDER records a
    measurement; the remainder keep the ``NaN`` from the dataset CSV, and callers are
    expected to drop them before computing statistics.

    Returns
    -------
    dict
        Mapping of ``system_id__ligand_id`` to ``ligand_binding_affinity``, covering
        every row of the curated dataset.
    """
    global _binding_dict
    if _binding_dict is None:
        df = _get_plinder_df()
        _binding_dict = dict(
            zip(df["system_id__ligand_id"], df["ligand_binding_affinity"], strict=False)
        )
    return _binding_dict


def seed(seed: int = SEED) -> None:
    """
    Set the global random seed for NumPy and Python's ``random`` module.

    Call once at the start of a run to make pose sampling and any other stochastic
    steps reproducible. Note that this covers only the two global generators: it does
    not affect CatBoost, whose seed is passed separately via its ``random_seed``
    parameter, nor any generator created explicitly with its own state.

    Parameters
    ----------
    seed : int, optional
        Integer seed value. Defaults to ``SEED`` from
        [model_params.py](model_params.py), so that all scripts seed identically
        unless told otherwise.

    Returns
    -------
    None
    """
    np.random.seed(seed)
    random.seed(seed)


def add_target_system_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``target_system`` column to the DataFrame, looked up by ``system_id``.
    This new column is required for cross-docked systems, where the apo receptor filename is built from the ``target_system``.

    The mapping is built from the curated dataset (see ``_get_plinder_df``) by pairing
    its ``system_id`` and ``target_system`` columns, and is then applied to ``df`` with
    ``pd.Series.map``. ``target_system`` is the apo structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``system_id`` column.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, with an additional ``target_system`` column.
    """
    df_ref = _get_plinder_df()
    system_to_target = dict(
        zip(df_ref["system_id"], df_ref["target_system"], strict=False)
    )
    df["target_system"] = df["system_id"].map(system_to_target)
    return df


@functools.cache
def get_system_metal(
    system_id: str,
    ligand_id: str | None = None,
    target_system: str | None = None,
    split: str | None = None,
    docking_type: str | None = None,
) -> list[str]:
    """
    Return the metal element symbols for a given system.

    Which source is used depends on ``split`` and ``docking_type``, since
    cross-docked systems were not pre-processed into the metal/ligand JSON.

    - **Redocked** (``split`` is ``'train'``/``'val'``, or ``docking_type`` is
      ``'redocked_FH'``): the symbols are read from the cached metal/ligand JSON,
      keyed by ``system_id`` alone. The other arguments are ignored, so they may all
      be left as ``None``. A ``system_id`` missing from the JSON yields ``[]``.
    - **Cross-docked** (``docking_type`` is ``'apo_docked_FH'``): the metals are
      parsed out of the apo receptor PDB that ``generate_features.py`` writes to
      ``docking/{split}_systems/{system_id}/``, whose name is built from
      ``system_id``, ``target_system`` and ``ligand_id`` — so all four are required.
      A missing file triggers a warning and yields ``[]``, i.e. all metal one-hot
      columns are 0 for that system.
    - **Any other combination** raises ``ValueError``. In particular, leaving
      ``docking_type`` as ``None`` is only valid for the train and val splits.

    In the cross-docked case the PDB is first scanned textually by
    ``chelate.receptor.extract_metal_symbols_from_pdb``; if that finds nothing, RDKit
    is used as a fallback.

    Results are cached with ``functools.cache`` on the full argument tuple, so all
    arguments must be hashable and changes to the underlying files are not picked up
    within a process.

    Parameters
    ----------
    system_id : str
        PLINDER system identifier.
    ligand_id : str, optional
        Ligand instance identifier. Required for cross-docked lookups, ignored
        otherwise.
    target_system : str, optional
        Apo structure the system was docked into, used to build the receptor
        filename. Required for cross-docked lookups, ignored otherwise.
    split : str, optional
        Dataset split (``'train'``, ``'val'`` or ``'test'``). Selects the redocked
        branch for train/val, and locates the systems directory for cross-docked
        lookups.
    docking_type : str, optional
        Docking scenario: ``'redocked'`` or ``'apo_docked'``.

    Returns
    -------
    list of str
        Upper-case metal element symbols, e.g. ``['ZN']``. Empty if the system has no
        metals recorded.

    Raises
    ------
    FileNotFoundError
        If the cross-docked receptor file is missing.
    ValueError
        If ``split`` and ``docking_type`` do not describe a supported scenario.
    """
    if split in {"train", "val"} or docking_type == "redocked_FH":  # redocked only
        # Normalize to uppercase symbols so downstream one-hot matching is robust.
        metals = [
            str(m[0]).upper()
            for m in _get_metal_ligand_data().get(system_id, {}).get("metals", [])
        ]
        return metals

    elif docking_type == "apo_docked_FH":  # crossdocked only
        systems_root = Path(
            os.environ.get("SYSTEMS_ROOT", MODEL_DIR.parent / "docking")
        )
        data_path = systems_root / f"{split}_systems"
        apo_receptor_path = f"{data_path}/{system_id}/{system_id}__{target_system}_merged_{ligand_id}.pdb"
        if not os.path.exists(apo_receptor_path):
            # Returning no metals would silently zero the metal one-hot features and
            # change every cross-docked prediction.
            raise FileNotFoundError(
                f"Cross-docked receptor not found: {apo_receptor_path}. The receptors "
                "are in the Zenodo record (docking/test_systems/); set SYSTEMS_ROOT "
                "to the folder that holds test_systems/."
            )
        metals = extract_metal_symbols_from_pdb(apo_receptor_path)

        # Fallback to RDKit parsing if text parsing yielded nothing.
        if not metals:
            metal_source_mol = Chem.MolFromPDBFile(apo_receptor_path)
            if metal_source_mol is not None:
                for atom in metal_source_mol.GetAtoms():
                    symbol = atom.GetSymbol().upper()
                    if symbol in METAL_ATOMS_SET:
                        metals.add(symbol)

        return [str(m).upper() for m in metals]
    else:
        raise ValueError(
            f"Unsupported combination of split={split} and docking_type={docking_type} for get_system_metal. Please provide a valid combination."
        )


def add_metal_columns(
    df: pd.DataFrame, docking_type: str | None = None
) -> pd.DataFrame:
    """
    Add one-hot metal indicator columns and a raw ``metals`` list column to a DataFrame.

    Supported metals are ``{K, Co, Mn, Fe, Ca, Mg, Cu, Na, Ni, Zn}``.  A
    ``has_GenericMetal`` column is also added for any other metal present.  Prints
    a summary of metal counts per unique system/ligand pair.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing at least a ``system_id`` column.
    docking_type : str, optional
        ``'redocked_FH'`` or ``'apo_docked_FH'``.  Controls how metal symbols
        are resolved via ``get_system_metal``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with additional ``metals``, ``has_{Metal}``, and
        ``has_GenericMetal`` columns.
    """
    df_copy = df.copy()

    specific_metal_upper = {m.upper() for m in EXPLICIT_METALS}

    if "system_id" not in df_copy.columns:
        raise ValueError("add_metal_columns requires a 'system_id' column")

    if docking_type == "apo_docked_FH" and "target_system" not in df_copy.columns:
        df_copy = add_target_system_column(df_copy)

    metals_series = df_copy.apply(
        lambda row: get_system_metal(
            row["system_id"],
            row.get("ligand_id", None),
            row.get("target_system", None),
            row.get("split", None),
            docking_type=docking_type,
        ),
        axis=1,
    )
    metals_series = metals_series.apply(
        lambda metals_list: (
            [str(m).upper() for m in metals_list]
            if isinstance(metals_list, (list, tuple, set))
            else []
        )
    )

    new_cols = {"metals": metals_series}
    for metal in EXPLICIT_METALS:
        metal_upper = metal.upper()
        new_cols[f"has_{metal}"] = metals_series.apply(
            lambda metals_list, symbol=metal_upper: 1 if symbol in metals_list else 0
        )
    new_cols["has_GenericMetal"] = metals_series.apply(
        lambda metals_list: (
            1 if any(m not in specific_metal_upper for m in metals_list) else 0
        )
    )

    new_cols_to_add = {k: v for k, v in new_cols.items() if k not in df_copy.columns}
    if new_cols_to_add:
        df_copy = pd.concat(
            [df_copy, pd.DataFrame(new_cols_to_add, index=df_copy.index)], axis=1
        )

    return df_copy


def rerank_poses(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Rerank poses within each system/ligand group based on a scoring column.

    ``CNNscore`` is ranked descending (higher is better); all other columns are
    ranked ascending (lower is better).  The resulting ``pose`` column contains
    integer ranks starting at 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``system_id__ligand_id`` grouping column and the
        scoring column ``col``.
    col : str
        Name of the column to rank by (e.g. ``'docking_score'`` or
        ``'CNNscore'``).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the ``pose`` column updated to reflect the new
        ranking.
    """
    df_clean = df.copy()

    if col == "CNNscore":
        df_clean["pose"] = df_clean.groupby("system_id__ligand_id")[col].rank(
            method="first", ascending=False
        )
    else:
        df_clean["pose"] = df_clean.groupby("system_id__ligand_id")[col].rank(
            method="first", ascending=True
        )

    return df_clean


def calculate_success_rate(
    df: pd.DataFrame,
    unique_identifier: str = "system_id__ligand_id",
    rerank_col: str | None = "docking_score",
    sr_col: str = "symm_rmsd",
    sr_threshold: float = 2,
    top_n: list[int] | None = None,
) -> dict[int, float]:
    """
    Compute top-k success rates, optionally re-ranking poses by a scoring column.

    A system counts as successful at k if any of its k best-ranked poses is within
    ``sr_threshold`` of the reference. The denominator is the number of groups present
    in ``df``; use ``calculate_success_rate_static`` when it has to stay fixed across
    differently filtered inputs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of docked poses including ``unique_identifier``, ``rerank_col``,
        and ``sr_col`` columns.
    unique_identifier : str, optional
        Column that identifies each system/ligand pair.
    rerank_col : str or None, optional
        Scoring column used to re-rank poses within each group. Pass ``None`` to keep
        the existing ``pose`` column, i.e. to evaluate the ranking the poses already
        carry rather than imposing a new one.
    sr_col : str, optional
        RMSD column used to assess whether a pose is a success (value ≤
        ``sr_threshold``).
    sr_threshold : float, optional
        RMSD threshold in Å below which a pose is considered successful.
    top_n : list of int, optional
        List of top-k values to compute success rates for. Defaults to
        ``[1, 3, 5, 10, 20, 30, 50, 100]``.

    Returns
    -------
    dict
        Mapping from top-k value to fraction of systems with a successful pose
        in the top-k.  Keys are ``[1, 3, 5, 10, 20, 30, 50, 100]``.
    """
    top_n_poses = np.array(
        top_n if top_n is not None else [1, 3, 5, 10, 20, 30, 50, 100]
    )

    if rerank_col is not None:
        df = rerank_poses(df, rerank_col)
    min_poses = df.groupby(unique_identifier).apply(
        lambda g: (
            g.loc[g[sr_col] <= sr_threshold, "pose"].min()
            if not g.loc[g[sr_col] <= sr_threshold].empty
            else np.inf
        ),
        include_groups=False,
    )
    success_rates = [(min_poses <= n).mean() for n in top_n_poses]
    return dict(zip(top_n_poses, success_rates, strict=False))


def calculate_success_rate_static(
    df: pd.DataFrame,
    *,
    all_ids: list,
    rerank_col: str,
    sr_col: str,
    sr_threshold: float = 2,
    top_n: list[int] | None = None,
    unique_identifier: str = "system_id__ligand_id",
) -> dict[int, float]:
    """
    Calculate top-k success rates against a fixed denominator.

    Unlike ``calculate_success_rate``, the denominator is ``len(all_ids)`` rather than
    the number of groups present in ``df``. Identifiers in ``all_ids`` that are absent
    from ``df`` count as failures at every k, so filtering poses out of ``df`` lowers
    the success rate instead of shrinking the reference set. This is what makes the
    value comparable across Optuna trials, where it is used as the optimisation target.

    Usually called via ``get_static_calc_sr``, which binds everything but ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Poses to score, with the grouping column, ``rerank_col`` and ``sr_col``.
    all_ids : sequence of str
        Every ``system_id__ligand_id`` originally present before filtering. Defines the
        denominator.
    rerank_col : str
        Scoring column used to re-rank poses within each group.
    sr_col : str
        RMSD column used to assess pose success.
    sr_threshold : float, optional
        RMSD threshold in Angstrom below which a pose counts as successful.
    top_n : list of int, optional
        Top-k values to compute success rates for. Defaults to
        ``[1, 3, 5, 10, 20, 30, 50, 100]``.
    unique_identifier : str, optional
        Column identifying a system-ligand pair.

    Returns
    -------
    dict[int, float]
        Success rate per entry in ``top_n``.
    """
    if top_n is None:
        top_n = [1, 3, 5, 10, 20, 30, 50, 100]

    df = rerank_poses(df, rerank_col)

    min_poses = (
        df.loc[df[sr_col] <= sr_threshold]
        .groupby(unique_identifier)["pose"]
        .min()
        .reindex(all_ids, fill_value=np.inf)
    )

    total = len(all_ids)
    return {n: (min_poses <= n).sum() / total for n in top_n}


def get_static_calc_sr(
    all_ids: list,
    rerank_col: str,
    sr_col: str,
    sr_threshold: float = 2,
    top_n: list[int] | None = None,
) -> Callable:
    """
    Return a success-rate function with a fixed denominator equal to ``len(all_ids)``.

    Applies when a subset of poses has been filtered and the denominator is required to
    remain the full original system count rather than the filtered count.

    Parameters
    ----------
    all_ids : sequence of str
        Every ``system_id__ligand_id`` originally present before filtering.
    rerank_col : str
        Scoring column used to re-rank poses within each group.
    sr_col : str
        RMSD column used to assess pose success.
    sr_threshold : float, optional
        RMSD threshold in Å below which a pose is considered successful.
    top_n : list of int, optional
        List of top-k values to compute success rates for. Defaults to
        ``[1, 3, 5, 10, 20, 30, 50, 100]``.

    Returns
    -------
    callable
        ``calculate_success_rate_static`` with everything but ``df`` bound, i.e. a
        function ``f(df, unique_identifier=...)`` returning a dict of top-k success
        rates using the fixed denominator.
    """
    return functools.partial(
        calculate_success_rate_static,
        all_ids=all_ids,
        rerank_col=rerank_col,
        sr_col=sr_col,
        sr_threshold=sr_threshold,
        top_n=top_n if top_n is not None else [1, 3, 5, 10, 20, 30, 50, 100],
    )


def get_binding_affinity_stats(df: pd.DataFrame, method: str) -> dict[str, float]:
    """
    Compute binding affinity correlation statistics for the top-ranked pose per system.

    Selects pose rank 1 for each system/ligand pair, maps experimental binding
    affinities from the global plinder dataset, and computes Pearson r, Kendall τ,
    and RMSE between the docking score and the experimental affinity.

    Parameters
    ----------
    df : pd.DataFrame
        Docked-pose DataFrame.  Must have ``system_id__ligand_id`` and a scoring
        column (``docking_score`` or ``CNNaffinity`` for GNINA).
    method : str
        Docking method label used to determine the appropriate score column
        (``'CNNaffinity'`` if ``'GNINA'`` in method, else ``'docking_score'``).

    Returns
    -------
    dict
        Keys: ``'N'``, ``'Pearson_R'``, ``'P_value'``, ``'Kendall_Tau'``,
        ``'Kendall_P_value'``, ``'RMSE'``.  All values are ``float``; statistics
        are ``NaN`` if fewer than 2 valid data points are available.
    """
    try:
        df_copy = df.copy()
        df_copy["affinity"] = df_copy["system_id__ligand_id"].map(_get_binding_dict())
        df_copy["has_affinity"] = df_copy["affinity"].notnull()

        if "pose" not in df_copy.columns:
            raise ValueError(
                "'pose' column is missing. Cannot compute binding affinity stats."
            )

        df_copy = df_copy[df_copy["pose"] == 1]
        df_copy = df_copy[df_copy["has_affinity"]]

        if "GNINA" in method:
            df_copy["ds"] = pd.to_numeric(df_copy["CNNaffinity"], errors="coerce")
        else:
            # Invert docking score so that higher is better for correlation, matching the affinity direction.
            df_copy["ds"] = -pd.to_numeric(df_copy["docking_score"], errors="coerce")

        # CSV inputs can contain numeric values mixed with empty strings or other
        # text, which gives these columns an object dtype. Convert explicitly before
        # calling NumPy; invalid values become NaN and are filtered below.
        df_copy["affinity"] = pd.to_numeric(df_copy["affinity"], errors="coerce")

        # Drop rows where the score or affinity is NaN / infinite.
        df_copy = df_copy.dropna(subset=["ds", "affinity"])
        df_copy = df_copy[np.isfinite(df_copy["ds"]) & np.isfinite(df_copy["affinity"])]

        n = len(df_copy)
        if n < 2:
            raise ValueError(
                f"Not enough data points for correlation ({n} rows after filtering). "
                f"Need at least 2 rows with valid docking score and binding affinity."
            )

        pearson_r, p_value = stats.pearsonr(df_copy["ds"], df_copy["affinity"])
        kendall_tau, kendall_p_value = stats.kendalltau(
            df_copy["ds"], df_copy["affinity"]
        )
        rmse = np.sqrt(mean_squared_error(df_copy["affinity"], df_copy["ds"]))

    # Deliberately broad: these statistics are reported alongside an evaluation run and
    # must never abort it. Any failure degrades to a NaN result, which callers already
    # handle, rather than propagating.
    except Exception as e:
        print(f"[WARNING] Binding affinity stats failed for method='{method}': {e}")
        print(traceback.format_exc())
        return {
            "N": 0.0,
            "Pearson_R": np.nan,
            "P_value": np.nan,
            "Kendall_Tau": np.nan,
            "Kendall_P_value": np.nan,
            "RMSE": np.nan,
        }

    return {
        "N": float(n),
        "Pearson_R": float(pearson_r),
        "P_value": float(p_value),
        "Kendall_Tau": float(kendall_tau),
        "Kendall_P_value": float(kendall_p_value),
        "RMSE": float(rmse),
    }


def load_train_val_for_method(
    method: str,
    target: str,
    features_dir: str = _FEATURES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation feature CSVs for a docking method.

    Reads from ``features_dir``, adds metal OHE columns and the
    ``system_id__ligand_id`` composite key, and drops rows with negative target.

    Parameters
    ----------
    method : str
        Docking method label (e.g. ``'VINA'``).
    target : str
        Target RMSD column name used to filter invalid rows.
    features_dir : str, optional
        Directory containing the feature CSV files.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        ``(train_df, val_df)`` ready for model training.
    """
    train_path = os.path.join(
        features_dir,
        FEATURE_CSV_TEMPLATE.format(
            split="train", method=method, run_type=TRAIN_RUN_TYPE, run_id=TRAIN_RUN_ID
        ),
    )
    val_path = os.path.join(
        features_dir,
        FEATURE_CSV_TEMPLATE.format(
            split="val", method=method, run_type=TRAIN_RUN_TYPE, run_id=TRAIN_RUN_ID
        ),
    )
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    train_df = add_metal_columns(train_df, docking_type=TRAIN_RUN_TYPE)
    val_df = add_metal_columns(val_df, docking_type=TRAIN_RUN_TYPE)
    for df in (train_df, val_df):
        df["system_id__ligand_id"] = df["system_id"] + "__" + df["ligand_id"]
    train_df = train_df[train_df[target] >= 0].copy()
    val_df = val_df[val_df[target] >= 0].copy()
    return train_df, val_df
