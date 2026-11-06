"""Check receptor heavy atoms against the original, pair-specific structure.

These are assembled docking structures, not periodic simulation cells. Shared
sanitized intermediates are not reference structures: another ligand's
preparation may have replaced them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from chelate.constants import METAL_ELEMENT_SYMBOLS
from docking.cif_utils import read_atom_site_document

WATERS = frozenset({"HOH", "WAT", "H2O", "TIP", "SOL"})
# Exact keep set in the pinned PDBFixer removeHeterogens implementation.
PDBFIXER_RETAINED_RESIDUES = frozenset(
    [
        "ALA",
        "ASN",
        "CYS",
        "GLU",
        "HIS",
        "LEU",
        "MET",
        "PRO",
        "THR",
        "TYR",
        "ARG",
        "ASP",
        "GLN",
        "GLY",
        "ILE",
        "LYS",
        "PHE",
        "SER",
        "TRP",
        "VAL",
        "DA",
        "DG",
        "DC",
        "DT",
        "DI",
        "A",
        "G",
        "C",
        "U",
        "I",
        "N",
        "UNK",
    ]
)
AUTODOCK_ELEMENTS = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "S": "S",
    "SA": "S",
    "H": "H",
    "HD": "H",
    "HS": "H",
}


def read_source_atoms(path: Path) -> list[dict]:
    """Read original atom identity, including polymer and alternate-site data."""
    block = read_atom_site_document(str(path)).sole_block()
    table = block.find_mmcif_category("_atom_site.")
    indices = {tag.split(".", 1)[1]: i for i, tag in enumerate(table.tags)}
    required = {"type_symbol", "label_atom_id", "label_comp_id", "label_asym_id"}
    required.update("Cartn_" + axis for axis in "xyz")
    if not required <= indices.keys():
        raise ValueError("Original structure lacks atom identity or coordinates")
    entity_types = {}
    entity = block.find_mmcif_category("_entity.")
    if entity:
        columns = {tag.split(".", 1)[1]: i for i, tag in enumerate(entity.tags)}
        if {"id", "type"} <= columns.keys():
            entity_types = {
                row[columns["id"]]: row[columns["type"]].strip("\"'") for row in entity
            }
    atoms = []
    for row in table:

        def value(name, default="", atom_row=row):
            return (
                str(atom_row[indices[name]]).strip("\"'")
                if name in indices
                else default
            )

        seq = value("label_seq_id")
        atoms.append(
            {
                "element": value("type_symbol").capitalize(),
                "name": value("label_atom_id"),
                "residue": value("label_comp_id"),
                "chain": value("label_asym_id"),
                "auth_chain": value("auth_asym_id"),
                "number": value("auth_seq_id", seq),
                "insertion": value("pdbx_PDB_ins_code"),
                "alt": value("label_alt_id"),
                "model": value("pdbx_PDB_model_num", "1"),
                "polymer": (
                    entity_types.get(value("label_entity_id")) == "polymer"
                    or seq not in {"", ".", "?"}
                    or value("group_PDB") == "ATOM"
                ),
                "xyz": [float(value("Cartn_" + axis)) for axis in "xyz"],
            }
        )
    if not atoms:
        raise ValueError("Original structure has no atoms")
    return atoms


def read_receptor_atoms(path: Path) -> list[dict]:
    """Decode AutoDock types without confusing acceptor nitrogen NA with Na."""
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        kind = line.split()[-1]
        element = AUTODOCK_ELEMENTS.get(kind, kind.capitalize())
        xyz = [float(line[i : i + 8]) for i in (30, 38, 46)]
        if not np.isfinite(xyz).all():
            raise ValueError("Receptor has non-finite coordinates")
        if element not in {"H", "D"}:
            atoms.append({"element": element, "xyz": xyz})
    if not atoms:
        raise ValueError("Receptor has no heavy atoms")
    return atoms


@lru_cache(maxsize=1)
def pdbfixer_replacement_rules() -> dict[str, set[str]]:
    """Read allowed atom deletions from the installed PDBFixer templates.

    Only nonstandard-residue atoms absent from their replacement template may
    be ignored. Original atoms of standard polymer residues remain required.
    """
    import pdbfixer
    from pdbfixer.pdbfixer import substitutions

    templates = Path(pdbfixer.__file__).parent / "templates"
    result = {}
    for old, new in substitutions.items():
        path = templates / f"{new}.pdb"
        if path.is_file():
            result[old] = {
                line[12:16].strip()
                for line in path.read_text().splitlines()
                if line.startswith(("ATOM", "HETATM"))
                and line[76:78].strip() not in {"H", "D"}
            }
    return result


def distances_to_docking_box(points: list[list[float]], box: dict) -> np.ndarray:
    """Return Euclidean distances to the closed box, zero for points inside.

    The benchmark uses isolated structures, so periodic images do not apply.
    Distances are to the box surface, not its centre or an expanded cuboid.
    """
    center = np.asarray([box["center_" + axis] for axis in "xyz"], dtype=float)
    size = np.asarray([box["size_" + axis] for axis in "xyz"], dtype=float)
    coordinates = np.asarray(points, dtype=float).reshape(-1, 3)
    if (
        not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError(
            "Docking box and atom coordinates must be finite; sizes positive"
        )
    outside = np.maximum(np.abs(coordinates - center) - size / 2, 0)
    return np.linalg.norm(outside, axis=1)


def compare_source_atoms(
    source: list[dict],
    receptor: list[dict],
    *,
    target_ligand: str | None,
    drop_nonpolymer: bool,
    replacements: dict[str, set[str]] | None = None,
    tolerance: float = 0.01,
    docking_box: dict | None = None,
) -> dict:
    """Find lost source atoms and target-ligand contamination in one pair.

    PDBFixer's explicit heterogen-removal and replacement rules are supplied
    for fallback inputs only. Hydrogens, waters, additional models and added
    atoms are not missing-source defects. Alternate sites of the same original
    atom require one retained conformer. Chain labels and atom renaming do not
    affect the coordinate check. Reduce's amide flips exchange N/O identities,
    so polymer positions are compared geometrically; metals also match element.

    With docking_box supplied, only missing atoms at most 8 A from that box
    exclude a pair. Ligand contamination always excludes it. All missing atoms
    and the strict whole-protein diagnostic are retained in the result. The
    8 A interaction cutoff is part of the protocol and cannot be tuned here.
    """
    if tolerance != 0.01:
        raise ValueError("The source-coordinate tolerance is fixed at 0.01 A")
    if docking_box is not None:
        distances_to_docking_box([], docking_box)
    replacements = replacements or {}
    first_model = source[0]["model"]
    current = [a for a in source if a["model"] == first_model]
    ignored = Counter()
    grouped = defaultdict(list)
    residue_heavy = defaultdict(set)
    for atom in current:
        if atom["element"] not in {"H", "D"}:
            residue_heavy[(atom["chain"], atom["number"], atom["residue"])].add(
                atom["name"]
            )
    for atom in current:
        if atom["element"] in {"H", "D"}:
            ignored["hydrogen"] += 1
            continue
        if atom["residue"] in WATERS:
            ignored["water"] += 1
            continue
        if target_ligand is not None and atom["chain"] == target_ligand:
            ignored["target_ligand"] += 1
            continue
        residue = atom["chain"], atom["number"], atom["residue"]
        metal = (
            atom["element"] in METAL_ELEMENT_SYMBOLS
            and len(residue_heavy[residue]) == 1
        )
        retained_residue = (
            atom["residue"] in PDBFIXER_RETAINED_RESIDUES
            or atom["residue"] in replacements
        )
        if (
            drop_nonpolymer
            and not atom["polymer"]
            and not metal
            and not retained_residue
        ):
            ignored["removed_nonpolymer"] += 1
            continue
        allowed = replacements.get(atom["residue"])
        if allowed is not None and atom["name"] not in allowed:
            ignored["replacement_atom"] += 1
            continue
        key = (*residue, atom["insertion"], atom["name"])
        grouped[key].append(dict(atom, metal=metal))
    xyz = np.asarray([a["xyz"] for a in receptor], dtype=float)
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("Receptor has no finite heavy-atom coordinates")
    tree = cKDTree(xyz)
    assigned = set()
    missing = []
    for alternatives in grouped.values():
        found = None
        for atom in alternatives:
            neighbors = tree.query_ball_point(atom["xyz"], tolerance + 1e-12)
            for i in neighbors:
                if i not in assigned and (
                    not atom["metal"] or receptor[i]["element"] == atom["element"]
                ):
                    found = i
                    break
            if found is not None:
                break
        if found is None:
            atom = dict(alternatives[0])
            if docking_box is not None:
                atom["distance_to_docking_box"] = float(
                    distances_to_docking_box(
                        [alternative["xyz"] for alternative in alternatives],
                        docking_box,
                    ).min()
                )
            missing.append(atom)
        else:
            assigned.add(found)
    contamination = []
    if target_ligand is not None:
        # A coincident atom already required by the receptor is not evidence of
        # a ligand contaminant, for example a duplicated covalent attachment.
        expected_points = [a["xyz"] for v in grouped.values() for a in v]
        expected_tree = cKDTree(expected_points) if expected_points else None
        for atom in current:
            if atom["chain"] != target_ligand or atom["element"] in {"H", "D"}:
                continue
            if expected_tree is not None and expected_tree.query_ball_point(
                atom["xyz"], tolerance + 1e-12
            ):
                continue
            neighbors = tree.query_ball_point(atom["xyz"], tolerance + 1e-12)
            if any(receptor[i]["element"] == atom["element"] for i in neighbors):
                contamination.append(atom)
    relevant = (
        missing
        if docking_box is None
        else [atom for atom in missing if atom["distance_to_docking_box"] <= 8.0]
    )
    result = {
        "defective": bool(relevant or contamination),
        "source_atom_groups_expected": len(grouped),
        "source_atom_groups_retained": len(grouped) - len(missing),
        "missing_protein_atoms": sum(a["polymer"] for a in missing),
        "missing_metal_atoms": sum(a["metal"] for a in missing),
        "missing_other_atoms": sum(
            not a["polymer"] and not a["metal"] for a in missing
        ),
        "target_ligand_atoms_present": len(contamination),
        "missing_atoms": missing,
        "ligand_atoms_in_receptor": contamination,
        "ignored_source_atoms": dict(ignored),
        "coordinate_tolerance": tolerance,
    }
    if docking_box is not None:
        outside = [atom for atom in missing if atom["distance_to_docking_box"] > 8.0]
        result.update(
            strict_whole_protein_defective=bool(missing or contamination),
            docking_box=dict(docking_box),
            binding_site_interaction_cutoff=8.0,
            binding_site_missing_atoms=len(relevant),
            binding_site_missing_protein_atoms=sum(
                atom["polymer"] for atom in relevant
            ),
            binding_site_missing_metal_atoms=sum(atom["metal"] for atom in relevant),
            binding_site_missing_other_atoms=sum(
                not atom["polymer"] and not atom["metal"] for atom in relevant
            ),
            missing_atoms_outside_binding_site=len(outside),
            nearest_missing_atom_distance_to_box=min(
                (atom["distance_to_docking_box"] for atom in missing), default=None
            ),
            nearest_outside_missing_atom_distance_to_box=min(
                (atom["distance_to_docking_box"] for atom in outside), default=None
            ),
        )
    return result


def audit_original_receptor(
    original: Path,
    merged: Path,
    *,
    target_ligand: str | None,
    pdbfixer_used: bool,
    docking_box: dict | None = None,
) -> dict:
    """Audit the original pair-specific source rather than shared intermediates."""
    return compare_source_atoms(
        read_source_atoms(original),
        read_receptor_atoms(merged),
        target_ligand=target_ligand,
        drop_nonpolymer=pdbfixer_used,
        replacements=pdbfixer_replacement_rules() if pdbfixer_used else None,
        docking_box=docking_box,
    )
