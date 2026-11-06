"""
Description
-----------
The CHELATE feature schema: the exact ordered list of columns a trained model expects.

CatBoost matches columns by name when handed a :class:`pandas.DataFrame`, but falls
back to positional order for plain arrays, so the order here must match the training
schema. :func:`get_feature_list` returns the bundled models' training order.
:func:`chelate.predict.predict_poses` also reindexes on ``model.feature_names_`` to
support custom models with a different feature order.
"""

from chelate.constants import EXPLICIT_METALS

# Contact-count features between ligand and receptor atom subsets. Each name encodes a
# pair of atom subsets and a distance cutoff in Angstrom, e.g. ``cd_4.0`` counts contacts
# between ligand O/N/S (``c``) and receptor O/N/S (``d``) within 4.0 A; see ``CUTOFFS`` in
# generate_features.py for the subset letters.
_CONTACT_PAIRS: list[str] = [
    "ab_3.5",
    "ab_4.5",
    "ab_5.5",
    "cd_3.5",
    "cd_4.0",
    "cd_4.5",
    "cd_5.0",
    "cd_5.5",
    "cf_3.5",
    "cf_4.0",
    "cf_4.5",
    "cf_5.0",
    "cf_5.5",
    "ed_3.5",
    "ed_4.0",
    "ed_4.5",
    "ed_5.0",
    "ed_5.5",
    "ef_3.5",
    "ef_4.0",
    "ef_4.5",
    "ef_5.0",
    "ef_5.5",
    "ch_3.0",
    "ch_3.5",
    "ch_4.0",
    "ch_4.5",
    "eh_4.0",
    "eh_4.5",
    "eh_5.0",
    "eh_5.5",
    "gh_5.5",
]

# Closest approach per subset pair. These have no per-statistic variants.
_MINDIST_FEATURES: list[str] = [
    "mindist_cd",
    "mindist_cf",
    "mindist_ed",
    "mindist_ef",
]

# Coordination-number, coordination-angle and pose-rank features. ``cn_*_3.5`` decompose
# the metal coordination number within 3.5 A into ligand and protein contributions (and,
# for the ligand, per donor element). ``angle_*`` summarise the donor-metal-donor angles,
# including the deviation from ideal tetrahedral and octahedral geometry. ``pose_rank``
# and ``pose_rank_norm`` carry the rank the docking program itself assigned to the pose.
_ANG_CN_POSE_RANK_FEATURES: list[str] = [
    "cn_lig_3.5",
    "cn_prot_3.5",
    "cn_total_3.5",
    "cn_lig_fraction_3.5",
    "cn_lig_O_3.5",
    "cn_lig_N_3.5",
    "cn_lig_S_3.5",
    "angle_count",
    "angle_mean",
    "angle_std",
    "angle_min",
    "angle_max",
    "angle_dev_tetrahedral",
    "angle_dev_octahedral",
    "pose_rank",
    "pose_rank_norm",
]

_CONTACT_STATS: tuple[str, ...] = ("_min", "_max", "_std", "_mean")
_MBG_CHARGE_CUTOFFS: list[str] = ["3.0", "3.5", "4.0", "4.5", "5.0", "5.5"]


def get_metal_cols() -> list[str]:
    """
    Return the metal indicator column names used in the feature schema.

    These are the columns written by ``add_metal_columns`` and consumed as the trailing
    block of ``get_feature_list``. One column per metal in ``EXPLICIT_METALS``, which
    holds the metals frequent enough in the curated dataset to warrant their own
    encoding, plus ``has_GenericMetal`` covering every remaining metal collectively.
    The order matches ``EXPLICIT_METALS`` and must stay stable, since it is part of
    the feature order returned by ``get_feature_list``.

    Returns
    -------
    list of str
        Column names of the form ``has_{Metal}``, followed by ``has_GenericMetal``.
    """
    res = [f"has_{metal}" for metal in EXPLICIT_METALS]
    res.append("has_GenericMetal")
    return res


def get_score_col_for_method(method: str) -> str:
    """
    Return the name of the column holding the docking score for a given method.

    GNINA is scored by its CNN pose score (``CNNscore``, higher is better), while VINA
    and SMINA report a binding free energy in ``docking_score`` (lower is better). The
    direction matters when ranking: see ``rerank_poses``.

    Parameters
    ----------
    method : str
        Docking method: ``'VINA'``, ``'SMINA'`` or ``'GNINA'``.

    Returns
    -------
    str
        ``'CNNscore'`` for GNINA, ``'docking_score'`` for VINA and SMINA.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three supported docking methods.
    """
    if method not in {"VINA", "SMINA", "GNINA"}:
        raise ValueError(f"Unknown docking method: {method!r}")
    return "CNNscore" if "GNINA" in method else "docking_score"


def get_feature_list(docking_program: str = "VINA") -> list[str]:
    """
    Return the feature column list for the given docking program.

    The list is assembled from the module-level feature-name groups: the contact counts
    (``_CONTACT_PAIRS``) and their closest-approach columns (``_MINDIST_FEATURES``),
    followed by one block per contact statistic in ``_CONTACT_STATS``, the metal-binding
    group charge statistics, the docking score, the coordination/angle/pose-rank
    features, and finally the metal one-hot columns from ``get_metal_cols``.

    Parameters
    ----------
    docking_program : str
        Docking program: ``'VINA'``, ``'SMINA'`` or ``'GNINA'``. GNINA uses
        ``CNNscore``; the others use ``docking_score``.

    Returns
    -------
    list[str]
        Ordered list of feature column names matching the bundled model for the given
        docking program.

    Raises
    ------
    ValueError
        If ``docking_program`` is not one of the three supported docking methods.
    """
    ds_col = get_score_col_for_method(docking_program)

    return (
        _CONTACT_PAIRS
        + _MINDIST_FEATURES
        + [f"{pair}{stat}" for stat in _CONTACT_STATS for pair in _CONTACT_PAIRS]
        + [
            f"mbg_charge_{stat}_{cutoff}"
            for cutoff in _MBG_CHARGE_CUTOFFS
            for stat in ("max", "mean", "std")
        ]
        + [ds_col]
        + _ANG_CN_POSE_RANK_FEATURES
        + get_metal_cols()
    )


def metal_one_hot(metals: list[str]) -> dict[str, int]:
    """
    Build the ``has_{Metal}`` indicator columns for one system.

    Matching is case-insensitive. Any metal outside ``EXPLICIT_METALS`` sets
    ``has_GenericMetal`` instead of a column of its own, and a system with several
    metals sets several columns.

    Parameters
    ----------
    metals : list of str
        Metal element symbols present in the system, e.g. ``['Zn']`` or ``['ZN', 'MG']``.

    Returns
    -------
    dict of str to int
        One entry per column from :func:`get_metal_cols`, each 0 or 1.
    """
    present = {str(m).upper() for m in metals}
    explicit_upper = {m.upper() for m in EXPLICIT_METALS}

    cols = {f"has_{metal}": int(metal.upper() in present) for metal in EXPLICIT_METALS}
    cols["has_GenericMetal"] = int(any(m not in explicit_upper for m in present))
    return cols
