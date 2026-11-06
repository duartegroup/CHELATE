"""
Description
-----------
The CHELATE feature engine: turn one docked pose plus its receptor into the numeric
feature row the re-ranking models were trained on.

``compute_contact_features`` is the entry point. It measures the geometry around the
metal site — distance-based contact counts between ligand and receptor atom subsets,
Gasteiger charge statistics near the metal, the coordination-number decomposition, and
the donor-metal-donor angles — and returns them as a named :class:`pandas.Series`.

The feature names, and the ``CUTOFFS`` table they are derived from, are part of the
trained models' input schema; see :func:`chelate.schema.get_feature_list` for the full
ordered column list.
"""

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree

from chelate.constants import METAL_ELEMENT_SYMBOLS

CUTOFFS: dict[str, float] = {
    "ab_3.5": 3.5,
    "ab_4.5": 4.5,
    "ab_5.5": 5.5,
    "cd_3.5": 3.5,
    "cd_4.0": 4.0,
    "cd_4.5": 4.5,
    "cd_5.0": 5.0,
    "cd_5.5": 5.5,
    "cf_3.5": 3.5,
    "cf_4.0": 4.0,
    "cf_4.5": 4.5,
    "cf_5.0": 5.0,
    "cf_5.5": 5.5,
    "ed_3.5": 3.5,
    "ed_4.0": 4.0,
    "ed_4.5": 4.5,
    "ed_5.0": 5.0,
    "ed_5.5": 5.5,
    "ef_3.5": 3.5,
    "ef_4.0": 4.0,
    "ef_4.5": 4.5,
    "ef_5.0": 5.0,
    "ef_5.5": 5.5,
    "ch_3.0": 3.0,
    "ch_3.5": 3.5,
    "ch_4.0": 4.0,
    "ch_4.5": 4.5,
    "eh_4.0": 4.0,
    "eh_4.5": 4.5,
    "eh_5.0": 5.0,
    "eh_5.5": 5.5,
    "gh_5.5": 5.5,
}


def _coords_from_rdkit(mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    Extract heavy-atom 3-D coordinates from an RDKit molecule as a float32 array.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        Molecule with a single conformer.

    Returns
    -------
    np.ndarray
        Shape (N, 3) float32 array of atomic coordinates.
    """
    if mol.GetNumAtoms() == 0:
        return np.empty((0, 3), dtype=np.float32)
    return mol.GetConformer().GetPositions().astype(np.float32)


def _min_distance(kdtree1: cKDTree, coords2: np.ndarray) -> float:
    """
    Return the minimum distance from any point in *coords2* to the nearest point in *kdtree1*.
    Returns ``np.inf`` if either set is empty.

    Parameters
    ----------
    kdtree1 : cKDTree
        Pre-built KD-tree for the first point set.
    coords2 : np.ndarray
        Shape (M, 3) query coordinates.

    Returns
    -------
    float
        Minimum pairwise distance, or ``np.inf`` if either set is empty.
    """
    if len(kdtree1.data) == 0 or len(coords2) == 0:
        return np.inf
    dists: np.ndarray
    dists, _ = kdtree1.query(coords2, k=1)
    return float(dists.min())


def _coordination_angle_features(
    metal_coords: np.ndarray,
    donor_coords_list: list[tuple[str, np.ndarray]],
    cutoff: float = 3.5,
) -> np.ndarray:
    """
    Compute all donor-metal-donor angles within a coordination-sphere cutoff.

    Parameters
    ----------
    metal_coords : np.ndarray
        Shape (M, 3) coordinates of metal atoms.
    donor_coords_list : list[tuple[str, np.ndarray]]
        Named donor coordinate sets, e.g. ``[("lig_ons", coords), ("prot_ons", coords)]``.
        All sets are pooled before computing angles.
    cutoff : float, optional
        Maximum metal-donor distance in Å used to select coordinating atoms (default: 3.5).

    Returns
    -------
    np.ndarray
        Float32 array of all pairwise donor-metal-donor angles in degrees.
        Empty if fewer than two donors are within the cutoff of any metal.
    """
    donor_arrays: list[np.ndarray] = [
        coords for _, coords in donor_coords_list if len(coords)
    ]
    if len(metal_coords) == 0 or len(donor_arrays) == 0:
        return np.empty(0, dtype=np.float32)

    all_donors: np.ndarray = np.vstack(donor_arrays)
    angles: list[float] = []

    for metal in metal_coords:
        dists: np.ndarray = np.linalg.norm(all_donors - metal, axis=1)
        nearby: np.ndarray = all_donors[dists < cutoff]
        if len(nearby) < 2:
            continue

        vecs: np.ndarray = nearby - metal
        norms: np.ndarray = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos_a: float = float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0))
                angles.append(np.degrees(np.arccos(cos_a)))

    if len(angles) == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(angles, dtype=np.float32)


def _empty_feature_series() -> pd.Series:
    """
    Return a zero-valued feature Series matching the full feature schema.

    Used as a safe fallback when contact-feature computation is not possible (e.g. no ligand
    heavy atoms).

    Returns
    -------
    pd.Series
        Feature series with all numeric fields set to 0 / 0.0, and ``mindist_*`` set to
        ``np.inf``.
    """
    feats: dict[str, float] = {}
    for key in CUTOFFS:
        feats[key] = 0
        feats[f"{key}_mean"] = 0.0
        feats[f"{key}_min"] = 0.0
        feats[f"{key}_max"] = 0.0
        feats[f"{key}_std"] = 0.0

    for cut in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        feats[f"mbg_charge_max_{cut}"] = 0.0
        feats[f"mbg_charge_mean_{cut}"] = 0.0
        feats[f"mbg_charge_std_{cut}"] = 0.0

    feats["mindist_cd"] = np.inf
    feats["mindist_cf"] = np.inf
    feats["mindist_ed"] = np.inf
    feats["mindist_ef"] = np.inf

    feats["cn_lig_3.5"] = 0.0
    feats["cn_prot_3.5"] = 0.0
    feats["cn_total_3.5"] = 0.0
    feats["cn_lig_fraction_3.5"] = 0.0
    feats["cn_lig_O_3.5"] = 0.0
    feats["cn_lig_N_3.5"] = 0.0
    feats["cn_lig_S_3.5"] = 0.0

    feats["angle_count"] = 0.0
    feats["angle_mean"] = 0.0
    feats["angle_std"] = 0.0
    feats["angle_min"] = 0.0
    feats["angle_max"] = 0.0
    feats["angle_dev_tetrahedral"] = 0.0
    feats["angle_dev_octahedral"] = 0.0

    feats["pose_rank"] = 0.0
    feats["pose_rank_norm"] = 0.0
    return pd.Series(feats)


def compute_gasteiger_charges(mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    Compute Gasteiger partial charges, returning zeros for metal atoms and on failure.

    Metal atoms are temporarily stripped before calling RDKit's Gasteiger routine
    (which cannot handle them), and the charges are mapped back to the original atom indices.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        Input molecule. May contain metal atoms.

    Returns
    -------
    np.ndarray
        Float32 array of length ``mol.GetNumAtoms()`` with partial charges; metal atoms
        and any atoms with non-finite charges are set to 0.0.
    """
    all_charges: np.ndarray = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
    non_metal_idxs: list[int] = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol().capitalize() not in METAL_ELEMENT_SYMBOLS
    ]
    if not non_metal_idxs:
        return all_charges

    rw_mol = Chem.RWMol(mol)
    metal_idxs: list[int] = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol().capitalize() in METAL_ELEMENT_SYMBOLS
    ]
    for idx in sorted(metal_idxs, reverse=True):
        rw_mol.RemoveAtom(idx)
    charge_mol = rw_mol.GetMol()

    try:
        AllChem.ComputeGasteigerCharges(charge_mol)
    except Exception:
        warnings.warn(
            f"Gasteiger charge computation failed for molecule with {len(metal_idxs)} metal atoms; returning zero charges.",
            stacklevel=2,
        )
        return all_charges

    for new_idx, old_idx in enumerate(non_metal_idxs):
        atom: Chem.Atom = charge_mol.GetAtomWithIdx(new_idx)
        charge: float = 0.0
        if atom.HasProp("_GasteigerCharge"):
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                charge = 0.0

        if not np.isfinite(charge):
            charge = 0.0
        all_charges[old_idx] = charge

    return all_charges


def compute_contact_features(
    structure_rdkit: Chem.rdchem.Mol,
    docked_mol: Chem.rdchem.Mol,
    metals: list,
    pose_idx: int,
    n_poses: int,
    receptor_charges: np.ndarray | None = None,
) -> pd.Series:
    """
    Compute all geometric and physicochemical contact features for one docked pose.

    Features include distance-based contact counts and statistics (from ``CUTOFFS``),
    Gasteiger charge statistics near metal atoms, coordination-number decomposition,
    donor-metal-donor angles, and pose rank.

    Parameters
    ----------
    structure_rdkit : Chem.rdchem.Mol
        Parsed receptor structure (PDB → RDKit molecule).
    docked_mol : Chem.rdchem.Mol
        A single docked ligand pose as an RDKit molecule.
    metals : list[tuple[str, np.ndarray]]
        Metal atoms extracted from the receptor via
        :func:`chelate.receptor.extract_metals`.
    pose_idx : int
        1-based rank of this pose within the docked ensemble.
    n_poses : int
        Total number of poses in the docked ensemble (used to normalise ``pose_rank_norm``).
    receptor_charges : np.ndarray, optional
        Output of :func:`compute_gasteiger_charges` for ``structure_rdkit``. The receptor
        charges do not depend on the pose, so pass them when featurising many poses
        against the same receptor; computed here if omitted.

    Returns
    -------
    pd.Series
        Named feature series.
    """
    if n_poses < 1:
        raise ValueError(f"n_poses must be >= 1, got {n_poses}")
    if not 1 <= pose_idx <= n_poses:
        raise ValueError(f"pose_idx {pose_idx} out of range [1, {n_poses}]")

    all_conf_coords: np.ndarray = _coords_from_rdkit(docked_mol)

    lig_idxs: list[int] = [
        a.GetIdx() for a in docked_mol.GetAtoms() if a.GetSymbol() != "H"
    ]
    if len(lig_idxs) == 0:
        warnings.warn(
            f"Docked pose {pose_idx}/{n_poses} has no heavy atoms "
            f"({docked_mol.GetNumAtoms()} atoms total); returning zero-valued features.",
            stacklevel=2,
        )
        return _empty_feature_series()

    coords_all: np.ndarray = all_conf_coords[lig_idxs]

    ons_idxs: list[int] = [
        i
        for i in lig_idxs
        if docked_mol.GetAtomWithIdx(i).GetSymbol() in {"O", "N", "S"}
    ]
    non_ons: list[int] = list(set(lig_idxs) - set(ons_idxs))
    coords_ons: np.ndarray = (
        all_conf_coords[ons_idxs] if ons_idxs else np.empty((0, 3), dtype=np.float32)
    )
    coords_non: np.ndarray = (
        all_conf_coords[non_ons] if non_ons else np.empty((0, 3), dtype=np.float32)
    )

    lig_met_idxs: list[int] = [
        a.GetIdx()
        for a in docked_mol.GetAtoms()
        if a.GetSymbol() in METAL_ELEMENT_SYMBOLS
    ]
    coords_lig_met: np.ndarray = (
        all_conf_coords[lig_met_idxs]
        if lig_met_idxs
        else np.empty((0, 3), dtype=np.float32)
    )

    prot_all_coords: np.ndarray = _coords_from_rdkit(structure_rdkit)
    prot_heavy_idxs: list[int] = [
        a.GetIdx()
        for a in structure_rdkit.GetAtoms()
        if a.GetSymbol() != "H"
        and a.GetSymbol().capitalize() not in METAL_ELEMENT_SYMBOLS
    ]
    prot_coords: np.ndarray = (
        prot_all_coords[prot_heavy_idxs]
        if prot_heavy_idxs
        else np.empty((0, 3), dtype=np.float32)
    )

    prot_ons_idxs: list[int] = [
        a.GetIdx()
        for a in structure_rdkit.GetAtoms()
        if a.GetSymbol() in {"O", "N", "S"}
        and a.GetSymbol().capitalize() not in METAL_ELEMENT_SYMBOLS
    ]
    coords_prot_ons: np.ndarray = (
        prot_all_coords[prot_ons_idxs]
        if prot_ons_idxs
        else np.empty((0, 3), dtype=np.float32)
    )

    prot_non_idxs: list[int] = [
        a.GetIdx()
        for a in structure_rdkit.GetAtoms()
        if a.GetSymbol() != "H"
        and a.GetSymbol().capitalize() not in METAL_ELEMENT_SYMBOLS
        and a.GetSymbol() not in {"O", "N", "S"}
    ]
    coords_prot_non: np.ndarray = (
        prot_all_coords[prot_non_idxs]
        if prot_non_idxs
        else np.empty((0, 3), dtype=np.float32)
    )

    coords_prot_met: np.ndarray = (
        np.asarray([coord for _, coord in metals], dtype=np.float32).reshape(-1, 3)
        if metals
        else np.empty((0, 3), dtype=np.float32)
    )

    kd: dict[str, cKDTree] = {
        "a": cKDTree(coords_all),
        "c": cKDTree(coords_ons if len(coords_ons) else np.empty((0, 3))),
        "e": cKDTree(coords_non if len(coords_non) else np.empty((0, 3))),
        "g": cKDTree(coords_lig_met if len(coords_lig_met) else np.empty((0, 3))),
        "b": cKDTree(prot_coords if len(prot_coords) else np.empty((0, 3))),
        "d": cKDTree(coords_prot_ons if len(coords_prot_ons) else np.empty((0, 3))),
        "f": cKDTree(coords_prot_non if len(coords_prot_non) else np.empty((0, 3))),
        "h": cKDTree(coords_prot_met if len(coords_prot_met) else np.empty((0, 3))),
    }
    coords: dict[str, np.ndarray] = {
        "d": coords_prot_ons,
        "f": coords_prot_non,
    }

    all_charges: np.ndarray = compute_gasteiger_charges(docked_mol)
    lig_charges: np.ndarray = (
        all_charges[lig_idxs] if lig_idxs else np.empty(0, dtype=np.float32)
    )

    feats: dict[str, float] = {}
    for key, cut in CUTOFFS.items():
        s1, s2 = key.split("_")[0]
        tree1 = kd[s1]
        tree2 = kd[s2]
        dists_dict = tree1.sparse_distance_matrix(tree2, cut)

        if len(dists_dict) == 0:
            feats[key] = 0
            feats[f"{key}_mean"] = 0.0
            feats[f"{key}_min"] = 0.0
            feats[f"{key}_max"] = 0.0
            feats[f"{key}_std"] = 0.0
            continue

        all_dists = np.fromiter(dists_dict.values(), dtype=np.float32)
        feats[key] = int(all_dists.size)
        feats[f"{key}_mean"] = float(all_dists.mean())
        feats[f"{key}_min"] = float(all_dists.min())
        feats[f"{key}_max"] = float(all_dists.max())
        feats[f"{key}_std"] = float(all_dists.std())

    prot_all_charges: np.ndarray = (
        compute_gasteiger_charges(structure_rdkit)
        if receptor_charges is None
        else receptor_charges
    )
    prot_charges: np.ndarray = (
        prot_all_charges[prot_heavy_idxs]
        if prot_heavy_idxs
        else np.empty(0, dtype=np.float32)
    )

    metal_tree: cKDTree = kd["h"]
    for cut in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        lig_idx_groups = metal_tree.query_ball_point(coords_all, r=cut)
        near_lig_idxs = [
            i for i, metal_hits in enumerate(lig_idx_groups) if len(metal_hits) > 0
        ]

        prot_idx_groups = metal_tree.query_ball_point(prot_coords, r=cut)
        near_prot_idxs = [
            i for i, metal_hits in enumerate(prot_idx_groups) if len(metal_hits) > 0
        ]

        combined_charges = []
        if near_lig_idxs:
            combined_charges.extend(lig_charges[near_lig_idxs])
        if near_prot_idxs:
            combined_charges.extend(prot_charges[near_prot_idxs])

        if not combined_charges:
            feats[f"mbg_charge_max_{cut}"] = 0.0
            feats[f"mbg_charge_mean_{cut}"] = 0.0
            feats[f"mbg_charge_std_{cut}"] = 0.0
        else:
            chs = np.asarray(combined_charges, dtype=np.float32)
            chs = chs[np.isfinite(chs)]
            if chs.size == 0:
                feats[f"mbg_charge_max_{cut}"] = 0.0
                feats[f"mbg_charge_mean_{cut}"] = 0.0
                feats[f"mbg_charge_std_{cut}"] = 0.0
            else:
                feats[f"mbg_charge_max_{cut}"] = float(chs.max())
                feats[f"mbg_charge_mean_{cut}"] = float(chs.mean())
                feats[f"mbg_charge_std_{cut}"] = float(chs.std())

    feats["mindist_cd"] = _min_distance(kd["c"], coords["d"])
    feats["mindist_cf"] = _min_distance(kd["c"], coords["f"])
    feats["mindist_ed"] = _min_distance(kd["e"], coords["d"])
    feats["mindist_ef"] = _min_distance(kd["e"], coords["f"])

    if len(coords_ons):
        lig_hit_lists = metal_tree.query_ball_point(coords_ons, r=3.5)
        cn_lig = int(sum(1 for hits in lig_hit_lists if len(hits) > 0))
    else:
        cn_lig = 0

    if len(coords_prot_ons):
        prot_hit_lists = metal_tree.query_ball_point(coords_prot_ons, r=3.5)
        cn_prot = int(sum(1 for hits in prot_hit_lists if len(hits) > 0))
    else:
        cn_prot = 0

    cn_total = cn_lig + cn_prot
    feats["cn_lig_3.5"] = float(cn_lig)
    feats["cn_prot_3.5"] = float(cn_prot)
    feats["cn_total_3.5"] = float(cn_total)
    feats["cn_lig_fraction_3.5"] = float(cn_lig / max(cn_total, 1))

    for elem in ["O", "N", "S"]:
        idxs = [a.GetIdx() for a in docked_mol.GetAtoms() if a.GetSymbol() == elem]
        coords_elem = (
            all_conf_coords[idxs] if idxs else np.empty((0, 3), dtype=np.float32)
        )
        if len(coords_elem):
            hits = metal_tree.query_ball_point(coords_elem, r=3.5)
            feats[f"cn_lig_{elem}_3.5"] = float(sum(1 for h in hits if len(h) > 0))
        else:
            feats[f"cn_lig_{elem}_3.5"] = 0.0

    angles: np.ndarray = _coordination_angle_features(
        coords_prot_met,
        [("lig_ons", coords_ons), ("prot_ons", coords_prot_ons)],
        cutoff=3.5,
    )
    feats["angle_count"] = float(len(angles))
    feats["angle_mean"] = float(angles.mean()) if len(angles) else 0.0
    feats["angle_std"] = float(angles.std()) if len(angles) else 0.0
    feats["angle_min"] = float(angles.min()) if len(angles) else 0.0
    feats["angle_max"] = float(angles.max()) if len(angles) else 0.0
    feats["angle_dev_tetrahedral"] = (
        float(np.mean(np.abs(angles - 109.5))) if len(angles) else 0.0
    )
    feats["angle_dev_octahedral"] = (
        float(
            np.mean(np.abs(np.minimum(np.abs(angles - 90.0), np.abs(angles - 180.0))))
        )
        if len(angles)
        else 0.0
    )

    feats["pose_rank"] = float(pose_idx)
    feats["pose_rank_norm"] = float(pose_idx / max(n_poses, 1))

    return pd.Series(feats)
