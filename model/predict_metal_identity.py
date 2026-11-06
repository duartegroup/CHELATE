"""
Predict metal identity by sweeping metal one-hot encodings. Only supports single-metal systems.

Workflow:
1) Load test systems and filter to single-metal systems with explicit metal labels.
2) Score the crystal pose with VINA for each candidate metal and a zero-metal control.
3) Compute features for the crystal pose.
4) Substitute each candidate's metal one-hot columns and VINA score; keep geometry fixed.
5) Predict MBG RMSD with a pre-trained CatBoost model and evaluate recovery.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from rdkit import Chem

from chelate.constants import EXPLICIT_METALS, METAL_ATOMS_SET
from chelate.features import compute_contact_features
from chelate.receptor import load_receptor_pdb
from docking.docking_params import CHARGE_DICT, VINA_PATH
from docking.receptor_pdb_cache import ensure_receptor_pdb
from model.model_params import (
    DATASET_CSV,
    DOCKING_RESULTS_BASE,
    METAL_COORD_MATCH_TOL,
    METAL_LIGAND_DATA_PATH,
    MODEL_DIR,
)
from model.model_utils import SEED

EXPLICIT_METAL_SET = set(EXPLICIT_METALS)


def candidate_protocol() -> dict:
    """Describe the unchanged explicit-metal ranking and its uniform chance rate."""
    return {
        "ranked_metals": list(EXPLICIT_METALS),
        "ranked_candidate_count": len(EXPLICIT_METALS),
        "uniform_top1_chance": 1.0 / len(EXPLICIT_METALS),
        "ALL_ZERO_is_ranked_in_primary_metric": False,
        "ALL_ZERO_role": "separate zero-metal control",
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the metal identity probe script.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace with fields: ``dataset_path``, ``metals_json``,
        ``systems_dir``, ``output_dir``, ``model``, ``vina_bin``, ``max_systems``,
        ``n_poses``, ``n_bootstrap``, ``bootstrap_seed``, ``bootstrap_ci``, ``verbose``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a trained MBG-RMSD re-ranking model has learned metal "
            "identity. For every eligible single-metal test system, the crystal pose "
            "is scored once with Vina and featurised, then the metal one-hot columns "
            "are overwritten with each metal in EXPLICIT_METALS in turn, holding all "
            "other features fixed. The model predicts an MBG RMSD per candidate, and "
            "the candidate with the lowest prediction counts as the model's choice. "
            "Agreement between that choice and the crystallographic metal is reported "
            "overall and per true metal, with optional bootstrap confidence intervals."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_CSV,
        help="Dataset CSV containing split and ligand columns.",
    )
    parser.add_argument(
        "--metals-json",
        type=str,
        default=METAL_LIGAND_DATA_PATH,
        help="System->metals mapping JSON.",
    )
    parser.add_argument(
        "--systems-dir",
        type=str,
        default=DOCKING_RESULTS_BASE,
        help="Directory containing per-system crystal pose files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODEL_DIR / "model_out"),
        help="Output directory for the result CSV and JSON files.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Explicit path to CatBoost .cbm model file.",
    )

    parser.add_argument(
        "--vina-bin",
        type=str,
        default=VINA_PATH,
        help="Path to VINA binary for score-only single-point calculation.",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=None,
        help=(
            "Process at most this many systems. Since a full run scores every system "
            "with Vina, this is intended for quick trial runs that check the pipeline "
            "end to end without waiting for all systems. The first N systems in "
            "dataset order are kept, so results are not representative."
        ),
    )
    parser.add_argument(
        "--metal-coord-match-tol",
        type=float,
        default=METAL_COORD_MATCH_TOL,
        help=(
            "Tolerance in Angstrom for matching a metal position from the curated "
            "JSON to a metal atom in the receptor PDBQT."
        ),
    )
    parser.add_argument(
        "--n-poses",
        type=int,
        default=100,
        help=(
            "Ensemble size used to normalise the pose_rank_norm feature of the "
            "crystal pose, which is featurised as pose 1 of n_poses, so "
            "pose_rank_norm = 1 / n_poses. The default of 100 matches a top pose "
            "of a 100-pose docking run (pose_rank_norm = 0.01), as seen in "
            "training. Pass 1 to featurise it as the only pose of a single-pose "
            "run (pose_rank_norm = 1.0)."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI estimation (0 to skip).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=SEED,
        help="RNG seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=95.0,
        help="Confidence interval width in percent (e.g. 95 for 95%% CI).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-system progress details.",
    )
    return parser.parse_args()


def _canonical_metal_symbol(raw: str) -> str:
    """
    Normalise a raw metal token to canonical element symbol capitalisation.

    Strips non-alphabetic characters, then returns the first letter uppercased
    and remaining letters lowercased (e.g. ``'ZN'`` → ``'Zn'``).

    Parameters
    ----------
    raw : str
        Raw metal string from a JSON or CSV field.

    Returns
    -------
    str
        Canonical symbol (e.g. ``'Zn'``), or empty string if no alphabetic
        characters are present.
    """
    letters = "".join(ch for ch in str(raw) if ch.isalpha())
    if not letters:
        return ""
    if len(letters) == 1:
        return letters.upper()
    return letters[0].upper() + letters[1:].lower()


def _load_model(model_path: Path) -> tuple[CatBoostRegressor, list[str]]:
    """
    Load a CatBoost model and return it together with its feature name list.

    Parameters
    ----------
    model_path : Path
        Path to the ``.cbm`` model file.

    Returns
    -------
    tuple of (CatBoostRegressor, list of str)
        ``(model, feature_names)``

    Raises
    ------
    RuntimeError
        If the loaded model carries no feature names.
    """
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    feature_names = list(getattr(model, "feature_names_", []) or [])
    if not feature_names:
        raise RuntimeError(
            "Loaded model has no feature names. This script requires named features."
        )
    if "docking_score" not in feature_names:
        raise RuntimeError(
            "The probe rescores the crystal pose with Vina, so it needs a model that "
            "uses the Vina/Smina docking_score feature (e.g. the VINA model), not "
            "GNINA's CNNscore."
        )
    return model, feature_names


def _resolve_system_files(
    system_dir: Path, ligand_id: str
) -> tuple[dict[str, Path | None], list[str]]:
    """
    Locate all required input files for a given system and ligand.

    Searches ``system_dir`` for the ligand SDF, receptor PDB, receptor PDBQT,
    and optional crystal ligand PDBQT using fixed naming conventions.

    Parameters
    ----------
    system_dir : Path
        Directory containing the system's docking input files.
    ligand_id : str
        Ligand instance identifier used to construct file names.

    Returns
    -------
    tuple of (dict of str to Path or None, list of str)
        ``(resolved, missing)``. ``resolved`` always contains all four keys
        (``'ligand_sdf'``, ``'ligand_pdbqt'``, ``'receptor_pdb'``,
        ``'receptor_pdbqt'``), mapping to a ``Path`` when the file was found and to
        ``None`` otherwise, so callers must handle ``None`` for every key.
        ``missing`` lists the keys that are required and were not found, i.e.
        ``'ligand_sdf'`` and ``'receptor_pdbqt'``; a ``None`` entry for one of the two
        optional files does not appear there.
    """
    missing: list[str] = []
    resolved: dict[str, Path | None]

    _p = system_dir / f"{ligand_id}.sdf"
    ligand_sdf = _p if _p.exists() else None
    if ligand_sdf is None:
        missing.append("ligand_sdf")

    ligand_pdbqt = next(
        (
            system_dir / n
            for n in [
                f"{ligand_id}_crystal.pdbqt",
                f"{ligand_id}.pdbqt",
                f"{ligand_id}_test.pdbqt",
            ]
            if (system_dir / n).exists()
        ),
        None,
    )

    _p = system_dir / "receptor_merged.pdb"
    receptor_pdb = _p if _p.exists() else None

    _p = system_dir / f"receptor_merged_{ligand_id}.pdbqt"
    receptor_pdbqt = _p if _p.exists() else None
    if receptor_pdbqt is None:
        missing.append("receptor_pdbqt")

    resolved = {
        "ligand_sdf": ligand_sdf,
        "ligand_pdbqt": ligand_pdbqt,
        "receptor_pdb": receptor_pdb,
        "receptor_pdbqt": receptor_pdbqt,
    }
    return resolved, missing


def _ensure_receptor_pdb(receptor_pdb: Path | None, receptor_pdbqt: Path) -> Path:
    """Use the validated cache for this exact ligand-specific PDBQT.

    ``receptor_pdb`` is the generic ``receptor_merged.pdb``. It cannot identify
    which target ligand was removed, so it is not used as an unverified source.
    """
    return ensure_receptor_pdb(receptor_pdbqt, receptor_pdbqt.with_suffix(".pdb"))


def _ensure_ligand_crystal_pdbqt(ligand_sdf: Path, ligand_id: str) -> Path:
    """
    Ensure a PDBQT file for the crystal ligand pose exists, creating it via OpenBabel if needed.

    Parameters
    ----------
    ligand_sdf : Path
        Source SDF file for the crystal ligand.
    ligand_id : str
        Ligand instance identifier used to name the output PDBQT.

    Returns
    -------
    Path
        Path to the ``{ligand_id}_crystal.pdbqt`` file.

    Raises
    ------
    RuntimeError
        If the conversion fails or the output is not created.
    """
    crystal_pdbqt = ligand_sdf.with_name(f"{ligand_id}_crystal.pdbqt")

    def require_atoms():
        if not any(
            line.startswith(("ATOM  ", "HETATM"))
            for line in crystal_pdbqt.read_text().splitlines()
        ):
            raise RuntimeError(
                "Crystal ligand PDBQT contains no atoms; ligand conversion failed: "
                f"{crystal_pdbqt}"
            )

    if crystal_pdbqt.exists():
        require_atoms()
        return crystal_pdbqt

    cmd = [
        "obabel",
        str(ligand_sdf),
        "-O",
        str(crystal_pdbqt),
        "-h",
        "--partialcharge",
        "gasteiger",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not crystal_pdbqt.exists():
        raise RuntimeError(
            f"Could not prepare ligand crystal PDBQT from SDF. Tried: {' '.join(cmd)}"
        )
    require_atoms()
    return crystal_pdbqt


def _parse_vina_score(vina_output: str) -> float | None:
    """
    Parse the affinity score from VINA stdout/stderr output.

    Tries two regex patterns in order: the VINA 1.2 ``Estimated Free Energy``
    format and the older ``Affinity`` format.

    Parameters
    ----------
    vina_output : str
        Combined stdout and stderr string from a VINA invocation.

    Returns
    -------
    float or None
        Parsed score in kcal/mol, or ``None`` if no match is found.
    """
    patterns = [
        r"Estimated Free Energy of Binding\s*:\s*([-+]?\d+(?:\.\d+)?)",
        r"Affinity\s*:\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for patt in patterns:
        match = re.search(patt, vina_output)
        if match:
            return float(match.group(1))
    return None


def _run_vina_score_only(
    vina_bin: Path, receptor_pdbqt: Path, ligand_pdbqt: Path
) -> tuple[float, str]:
    """
    Run VINA in score-only mode and return the affinity and combined output.

    Parameters
    ----------
    vina_bin : Path
        Path to the VINA binary.
    receptor_pdbqt : Path
        Receptor PDBQT file (must be already prepared).
    ligand_pdbqt : Path
        Ligand PDBQT file at the pose to score.

    Returns
    -------
    tuple of (float, str)
        ``(score, combined_output)`` where ``score`` is in kcal/mol and
        ``combined_output`` is the merged stdout+stderr string.

    Raises
    ------
    RuntimeError
        If VINA returns a non-zero exit code or the score cannot be parsed.
    """
    cmd = [
        str(vina_bin),
        "--score_only",
        "--autobox",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(f"VINA score-only failed (rc={proc.returncode})")

    score = _parse_vina_score(combined)
    if score is None:
        raise RuntimeError("Could not parse score from VINA output")
    return score, combined


def _parse_pdbqt_charge_and_type(line: str) -> tuple[float, str]:
    """
    Parse the charge and atom type from a PDBQT ATOM/HETATM record.

    PDBQT files store the charge in columns 71-76 and the atom type starting at column
    78 (1-indexed). What that field holds depends on how the atom was prepared: Meeko
    writes Gasteiger partial charges for the ordinary receptor atoms, whereas the metals
    it does not support are written out separately by the docking pipeline with their
    common oxidation state from ``CHARGE_DICT``, i.e. a formal charge.

    Parameters
    ----------
    line : str
        A single PDBQT ATOM or HETATM record line.

    Returns
    -------
    tuple of (float, str)
        ``(charge, atom_type)``; defaults to ``(0.0, '')`` if parsing fails.
    """
    charge = 0.0
    atom_type = ""
    try:
        charge = float(line[70:76].strip()) if line[70:76].strip() else 0.0
    except ValueError:
        charge = 0.0
    if len(line) >= 79:
        atom_type = line[77:].strip().split()[0] if line[77:].strip() else ""
    return charge, atom_type


def _format_pdbqt_atom_line(
    line: str,
    charge: float,
    atom_type: str,
    atom_name: str | None = None,
    residue_name: str | None = None,
) -> str:
    """
    Rewrite a PDBQT ATOM/HETATM line with updated charge and atom type fields.

    Optionally also overwrites the atom name (columns 13-16) and residue name
    (columns 18-20).  All other fields are preserved from the original line.

    Parameters
    ----------
    line : str
        Original PDBQT record line.
    charge : float
        New value for the PDBQT charge column. For the metal records rewritten by this
        script this is a formal charge from ``CHARGE_DICT``.
    atom_type : str
        New PDBQT atom type (up to 2 characters).
    atom_name : str or None, optional
        Replacement atom name; uses original if ``None``.
    residue_name : str or None, optional
        Replacement residue name; uses original if ``None``.

    Returns
    -------
    str
        Updated PDBQT record line terminated with a newline.
    """
    # Keep all original fields except selected metal identifiers and PDBQT charge/type columns.
    base = line.rstrip("\n")
    if len(base) < 79:
        base = base.ljust(79)

    if atom_name is not None:
        base = f"{base[:12]}{atom_name:>4}{base[16:]}"
    if residue_name is not None:
        base = f"{base[:17]}{residue_name:>3}{base[20:]}"

    updated = f"{base[:70]}{charge:6.3f} {atom_type:<2}"
    return updated.rstrip() + "\n"


def _metal_charge_for_candidate(candidate_symbol: str, fallback_charge: float) -> float:
    """
    Return the formal charge for a metal candidate from ``CHARGE_DICT``.

    ``CHARGE_DICT`` maps each metal to its common oxidation state in solution, so the
    returned value is a formal charge (an integer such as 2.0 for Zn(II)), not a
    Gasteiger partial charge. It is written into the PDBQT charge column of the
    substituted metal atom, matching how the docking pipeline prepares metals that
    Meeko does not support.

    Falls back to the Fe(II) entry for iron if the plain ``'FE'`` key is absent, and to
    ``fallback_charge`` if no entry is found at all.

    Parameters
    ----------
    candidate_symbol : str
        Metal element symbol (e.g. ``'Zn'``, ``'Fe'``).
    fallback_charge : float
        Charge to use when the symbol is not in ``CHARGE_DICT``. In practice this is
        the charge read from the original receptor record.

    Returns
    -------
    float
        Formal charge for the given candidate.
    """
    key = candidate_symbol.upper()
    if key in CHARGE_DICT:
        return float(CHARGE_DICT[key])
    if key == "FE" and "FE2" in CHARGE_DICT:
        return float(CHARGE_DICT["FE2"])
    return float(fallback_charge)


def _find_target_metal_atom_line_idx(
    lines: list[str],
    metal_coord: tuple[float, float, float],
    match_tol: float = METAL_COORD_MATCH_TOL,
) -> int:
    """
    Find the index of the receptor PDBQT line containing the target metal atom.

    Scans ATOM/HETATM lines whose PDBQT atom type is a metal, and returns the
    index of the line whose coordinates are closest to ``metal_coord``.

    A tolerance is needed because the reference coordinate comes from the curated
    metal/ligand JSON while the receptor has been through structure preparation, so the
    two positions agree closely but not exactly. The tolerance is deliberately far
    smaller than the spacing between distinct metal sites, so the nearest match within
    it is unambiguous.

    Parameters
    ----------
    lines : list of str
        Lines from the receptor PDBQT file.
    metal_coord : tuple of (float, float, float)
        Reference metal coordinates in Angstrom.
    match_tol : float, optional
        Maximum accepted distance in Angstrom between ``metal_coord`` and the closest
        receptor metal atom. Defaults to ``METAL_COORD_MATCH_TOL``; the CLI exposes it
        as ``--metal-coord-match-tol``.

    Returns
    -------
    int
        Zero-based index of the matching line in ``lines``.

    Raises
    ------
    RuntimeError
        If no metal atom line is found within ``match_tol`` Angstrom of
        ``metal_coord``.
    """
    target = np.asarray(metal_coord, dtype=np.float32)
    best_idx = -1
    best_dist = float("inf")

    for i, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            _, atom_type = _parse_pdbqt_charge_and_type(line)
            if atom_type.upper() not in METAL_ATOMS_SET:
                continue
            xyz = np.asarray(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                dtype=np.float32,
            )
        # Malformed coordinate columns mean this is not a usable atom record; the
        # caller raises if no candidate is found at all.
        except Exception:
            continue

        dist = float(np.linalg.norm(xyz - target))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx < 0 or best_dist > match_tol:
        raise RuntimeError(
            "Could not locate receptor metal atom near provided coordinate "
            f"(best distance={best_dist:.3f} A, tolerance={match_tol:.3f} A)."
        )
    return best_idx


def _write_candidate_receptor_pdbqts(
    receptor_pdbqt: Path,
    out_dir: Path,
    metal_coord: tuple[float, float, float],
    match_tol: float = METAL_COORD_MATCH_TOL,
) -> dict[str, Path]:
    """
    Write one modified receptor PDBQT per candidate metal (plus an ALL_ZERO variant).

    For each entry in ``EXPLICIT_METALS`` a copy of the receptor is written where
    the target metal atom has its charge and atom type replaced with the
    candidate's.  An additional ``ALL_ZERO`` variant removes the metal atom
    entirely.

    Parameters
    ----------
    receptor_pdbqt : Path
        Original receptor PDBQT file.
    out_dir : Path
        Output directory; created if absent.
    metal_coord : tuple of (float, float, float)
        Coordinates used to locate the target metal atom in the PDBQT.
    match_tol : float, optional
        Tolerance in Angstrom passed to ``_find_target_metal_atom_line_idx``.

    Returns
    -------
    dict of str to Path
        Mapping from candidate label (``'Zn'``, …, ``'ALL_ZERO'``) to the
        corresponding output PDBQT path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(receptor_pdbqt) as fh:
        orig_lines = fh.readlines()

    metal_idx = _find_target_metal_atom_line_idx(orig_lines, metal_coord, match_tol)
    original_charge, _ = _parse_pdbqt_charge_and_type(orig_lines[metal_idx])
    candidate_paths: dict[str, Path] = {}

    for candidate in [*EXPLICIT_METALS, "ALL_ZERO"]:
        lines = list(orig_lines)
        if candidate == "ALL_ZERO":
            del lines[metal_idx]
            out_path = out_dir / "receptor_ALL_ZERO.pdbqt"
        else:
            charge = _metal_charge_for_candidate(
                candidate, fallback_charge=original_charge
            )
            lines[metal_idx] = _format_pdbqt_atom_line(
                lines[metal_idx],
                charge=charge,
                atom_type=candidate,
                atom_name=candidate.upper(),
                residue_name=candidate.upper(),
            )
            out_path = out_dir / f"receptor_{candidate}.pdbqt"

        with open(out_path, "w") as fh:
            fh.writelines(lines)
        candidate_paths[candidate] = out_path

    return candidate_paths


def _score_candidates_with_vina(
    vina_bin: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    metal_coord: tuple[float, float, float],
    candidate_out_dir: Path,
    match_tol: float = METAL_COORD_MATCH_TOL,
) -> dict[str, float]:
    """
    Score all candidate metal variants of the receptor with VINA score-only mode.

    Writes per-candidate receptor PDBQT files (via ``_write_candidate_receptor_pdbqts``),
    then runs VINA for each one and returns the resulting scores.

    Parameters
    ----------
    vina_bin : Path
        Path to the VINA binary.
    receptor_pdbqt : Path
        Original receptor PDBQT.
    ligand_pdbqt : Path
        Crystal ligand PDBQT.
    metal_coord : tuple of (float, float, float)
        Metal coordinates used to locate the target atom.
    candidate_out_dir : Path
        Directory for intermediate per-candidate PDBQT files.
    match_tol : float, optional
        Tolerance in Angstrom passed to ``_write_candidate_receptor_pdbqts``.

    Returns
    -------
    dict of str to float
        Mapping from candidate label to VINA affinity score (kcal/mol).
    """
    receptor_paths = _write_candidate_receptor_pdbqts(
        receptor_pdbqt=receptor_pdbqt,
        out_dir=candidate_out_dir,
        metal_coord=metal_coord,
        match_tol=match_tol,
    )
    scores: dict[str, float] = {}
    for candidate in [*EXPLICIT_METALS, "ALL_ZERO"]:
        score, _ = _run_vina_score_only(
            vina_bin=vina_bin,
            receptor_pdbqt=receptor_paths[candidate],
            ligand_pdbqt=ligand_pdbqt,
        )
        scores[candidate] = float(score)
    return scores


def _build_eligible_systems(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Build the list of test-set system/ligand pairs eligible for the metal identity probe.

    Reads the dataset CSV and metals JSON, then filters for systems that have
    exactly one annotated metal from ``EXPLICIT_METALS``, a corresponding system
    directory, and a receptor PDBQT on disk.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with ``dataset_path``, ``metals_json``,
        ``systems_dir``, and optional ``max_systems`` cap.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, dict of str to int)
        ``(eligible_df, excluded_df, summary_counts)`` where ``eligible_df``
        contains the rows that will be processed, ``excluded_df`` records why
        each row was skipped, and ``summary_counts`` is a counter dict.
    """
    # Text identifiers: a ligand chain "NA" must not be read as missing.
    dataset_df = pd.read_csv(args.dataset_path, keep_default_na=False, na_values=[""])
    dataset_df = dataset_df[dataset_df["split"] == "test"].copy()
    dataset_df["ligand_id"] = (
        dataset_df["ligand_instance"].astype(str)
        + "."
        + dataset_df["ligand_asym_id"].astype(str)
    )

    with open(args.metals_json) as fh:
        metals_map = json.load(fh)

    systems_dir = Path(args.systems_dir)
    system_dirs = {p.name for p in systems_dir.iterdir() if p.is_dir()}

    eligible: list[dict] = []
    excluded: list[dict] = []
    reason_counter: Counter[str] = Counter()

    unique_rows = dataset_df[
        ["system_id", "ligand_id", "entry_pdb_id"]
    ].drop_duplicates()

    for row in unique_rows.itertuples(index=False):
        system_id = row.system_id
        ligand_id = row.ligand_id

        if system_id not in system_dirs:
            reason = "missing_system_directory"
            excluded.append(
                {
                    "system_id": system_id,
                    "ligand_id": ligand_id,
                    "reason": reason,
                }
            )
            reason_counter[reason] += 1
            continue

        metals_raw = metals_map.get(system_id, {}).get("metals", [])
        if len(metals_raw) == 0:
            reason = "no_metal_annotation"
            excluded.append(
                {"system_id": system_id, "ligand_id": ligand_id, "reason": reason}
            )
            reason_counter[reason] += 1
            continue

        if len(metals_raw) != 1:
            reason = "multi_metal_system"
            excluded.append(
                {"system_id": system_id, "ligand_id": ligand_id, "reason": reason}
            )
            reason_counter[reason] += 1
            continue

        true_metal = _canonical_metal_symbol(metals_raw[0][0])
        if true_metal not in EXPLICIT_METAL_SET:
            reason = "true_metal_not_explicit_onehot"
            excluded.append(
                {
                    "system_id": system_id,
                    "ligand_id": ligand_id,
                    "true_metal": true_metal,
                    "reason": reason,
                }
            )
            reason_counter[reason] += 1
            continue

        system_dir = Path(args.systems_dir) / system_id
        paths, missing = _resolve_system_files(system_dir, ligand_id)
        if missing:
            reason = "missing_required_files: " + ",".join(sorted(missing))
            excluded.append(
                {
                    "system_id": system_id,
                    "ligand_id": ligand_id,
                    "true_metal": true_metal,
                    "reason": reason,
                }
            )
            reason_counter["missing_required_files"] += 1
            continue

        coord = tuple(float(v) for v in metals_raw[0][1])
        eligible.append(
            {
                "system_id": system_id,
                "ligand_id": ligand_id,
                "true_metal": true_metal,
                "metal_coord_x": coord[0],
                "metal_coord_y": coord[1],
                "metal_coord_z": coord[2],
                "system_dir": str(system_dir),
                "ligand_sdf": str(paths["ligand_sdf"]),
                "ligand_pdbqt": str(paths["ligand_pdbqt"]),
                "receptor_pdb": str(paths["receptor_pdb"])
                if paths["receptor_pdb"] is not None
                else "",
                "receptor_pdbqt": str(paths["receptor_pdbqt"]),
            }
        )

    eligible_df = pd.DataFrame(eligible)
    excluded_df = pd.DataFrame(excluded)

    eligible_before_cap = len(eligible_df)
    cap_applied = False
    if args.max_systems is not None and len(eligible_df) > args.max_systems:
        eligible_df = eligible_df.head(args.max_systems).copy()
        cap_applied = True

    summary_counts = {
        "test_unique_system_ligands": len(unique_rows),
        "eligible_system_ligands_before_cap": eligible_before_cap,
        "eligible_system_ligands": len(eligible_df),
        "excluded_system_ligands": len(excluded_df),
        "max_systems_cap_applied": int(cap_applied),
    }
    summary_counts.update(
        {f"excluded_{k}": int(v) for k, v in sorted(reason_counter.items())}
    )

    return eligible_df, excluded_df, summary_counts


def _predict_for_candidates(
    model: CatBoostRegressor,
    feature_names: list[str],
    base_features: dict[str, float],
    candidate_vina_scores: dict[str, float],
    true_metal: str,
    system_id: str,
    ligand_id: str,
) -> pd.DataFrame:
    """
    Predict MBG RMSD for every candidate metal and return a result DataFrame.

    For each candidate in ``EXPLICIT_METALS`` and ``'ALL_ZERO'`` the base
    contact features are combined with the candidate's VINA score and one-hot
    metal encoding, then the model predicts the MBG RMSD.

    Parameters
    ----------
    model : CatBoostRegressor
        Trained model.
    feature_names : list of str
        Ordered list of feature column names expected by the model.
    base_features : dict of str to float
        Pre-computed contact features (without metal encoding or docking score), as
        produced by ``compute_contact_features`` and converted with ``Series.to_dict``.
    candidate_vina_scores : dict of str to float
        VINA score for each candidate label.
    true_metal : str
        True metal symbol for the system (used as a label column only).
    system_id : str
        System identifier (label column).
    ligand_id : str
        Ligand identifier (label column).

    Returns
    -------
    pd.DataFrame
        Columns: ``system_id``, ``ligand_id``, ``true_metal``,
        ``candidate_metal``, ``pred_mbg_rmsd``.
    """
    rows: list[dict] = []

    for candidate in [*EXPLICIT_METALS, "ALL_ZERO"]:
        row = dict(base_features)
        if candidate not in candidate_vina_scores:
            raise KeyError(f"Missing candidate Vina score for {candidate}")
        row["docking_score"] = float(candidate_vina_scores[candidate])

        if candidate == "ALL_ZERO":
            for metal in EXPLICIT_METALS:
                row[f"has_{metal}"] = 0
        else:
            for metal in EXPLICIT_METALS:
                row[f"has_{metal}"] = 1 if metal == candidate else 0
        row["has_GenericMetal"] = 0

        feature_row: dict[str, float] = {}
        for feat in feature_names:
            if feat in row:
                feature_row[feat] = row[feat]
            elif feat.startswith("has_"):
                feature_row[feat] = 0
            else:
                feature_row[feat] = np.nan

        rows.append(
            {
                "system_id": system_id,
                "ligand_id": ligand_id,
                "true_metal": true_metal,
                "candidate_metal": candidate,
                **feature_row,
            }
        )

    pred_df = pd.DataFrame(rows)
    x = pred_df[feature_names]
    pred_df["pred_mbg_rmsd"] = model.predict(x)
    return pred_df[
        ["system_id", "ligand_id", "true_metal", "candidate_metal", "pred_mbg_rmsd"]
    ]


def _summarize_system_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise per-candidate predictions into one row per system/ligand pair.

    Determines the winning candidate (minimum predicted MBG RMSD), the true
    metal's predicted score and rank, and correctness flags.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of ``_predict_for_candidates`` with columns ``system_id``,
        ``ligand_id``, ``true_metal``, ``candidate_metal``, ``pred_mbg_rmsd``.

    Returns
    -------
    pd.DataFrame
        One row per system/ligand pair with columns: ``system_id``,
        ``ligand_id``, ``true_metal``, ``winner_metal``,
        ``true_pred_mbg_rmsd``, ``best_pred_mbg_rmsd``, ``best_alt_pred_mbg_rmsd``,
        ``true_minus_best_alt``, ``true_rank``, ``correct``,
        ``top3_correct``, ``all_zero_pred_mbg_rmsd``, ``winner_with_all_zero``.
    """
    summary_rows: list[dict] = []

    for (system_id, ligand_id), grp in predictions_df.groupby(
        ["system_id", "ligand_id"]
    ):
        grp = grp.copy()
        true_metal = grp["true_metal"].iloc[0]

        grp_explicit = grp[grp["candidate_metal"].isin(EXPLICIT_METALS)].copy()
        if grp_explicit.empty:
            continue

        min_pred = float(grp_explicit["pred_mbg_rmsd"].min())
        winner_metal = str(
            grp_explicit.loc[grp_explicit["pred_mbg_rmsd"].idxmin(), "candidate_metal"]
        )
        true_row = grp_explicit[grp_explicit["candidate_metal"] == true_metal]
        true_pred = float(true_row["pred_mbg_rmsd"].iloc[0])

        # Stable: exact ties keep candidate order, as idxmin does above.
        grp_sorted = grp_explicit.sort_values(
            "pred_mbg_rmsd", ascending=True, kind="stable"
        ).reset_index(drop=True)
        true_rank = (
            int(grp_sorted.index[grp_sorted["candidate_metal"] == true_metal][0]) + 1
        )

        best_alt = float(
            grp_explicit.loc[
                grp_explicit["candidate_metal"] != true_metal, "pred_mbg_rmsd"
            ].min()
        )

        all_zero_row = grp[grp["candidate_metal"] == "ALL_ZERO"]
        all_zero_pred = (
            float(all_zero_row["pred_mbg_rmsd"].iloc[0])
            if not all_zero_row.empty
            else float("nan")
        )
        grp_full_sorted = grp.sort_values(
            "pred_mbg_rmsd", ascending=True, kind="stable"
        ).reset_index(drop=True)
        winner_with_all_zero = str(grp_full_sorted["candidate_metal"].iloc[0])

        correct = int(winner_metal == true_metal)
        top3_correct = int(true_rank <= 3)

        summary_rows.append(
            {
                "system_id": system_id,
                "ligand_id": ligand_id,
                "true_metal": true_metal,
                "winner_metal": winner_metal,
                "true_pred_mbg_rmsd": true_pred,
                "best_pred_mbg_rmsd": min_pred,
                "best_alt_pred_mbg_rmsd": best_alt,
                "true_minus_best_alt": true_pred - best_alt,
                "true_rank": true_rank,
                "correct": correct,
                "top3_correct": top3_correct,
                "all_zero_pred_mbg_rmsd": all_zero_pred,
                "winner_with_all_zero": winner_with_all_zero,
            }
        )

    return pd.DataFrame(summary_rows)


def _aggregate_metrics(
    system_summary_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute global and per-metal accuracy metrics from the system summary DataFrame.

    Parameters
    ----------
    system_summary_df : pd.DataFrame
        Output of ``_summarize_system_predictions``.

    Returns
    -------
    tuple of (dict of str to float, pd.DataFrame)
        ``(global_metrics, per_metal_df)`` where ``global_metrics`` contains
        ``n_systems``, ``accuracy``, ``mean_true_rank``, and ``top3_accuracy``;
        and ``per_metal_df`` contains the same metrics broken down by ``true_metal``.
    """
    if system_summary_df.empty:
        empty_per_metal = pd.DataFrame(
            columns=[
                "true_metal",
                "n_systems",
                "accuracy",
                "mean_true_rank",
                "mean_true_minus_best_alt",
            ]
        )
        return {
            "n_systems": 0,
            "accuracy": float("nan"),
            "mean_true_rank": float("nan"),
            "top3_accuracy": float("nan"),
        }, empty_per_metal

    global_metrics = {
        "n_systems": len(system_summary_df),
        "accuracy": float(system_summary_df["correct"].mean()),
        "mean_true_rank": float(system_summary_df["true_rank"].mean()),
        "top3_accuracy": float(system_summary_df["top3_correct"].mean()),
    }

    per_metal_df = (
        system_summary_df.groupby("true_metal", as_index=False)
        .agg(
            n_systems=("system_id", "count"),
            accuracy=("correct", "mean"),
            top3_accuracy=("top3_correct", "mean"),
            mean_true_rank=("true_rank", "mean"),
            mean_true_minus_best_alt=("true_minus_best_alt", "mean"),
        )
        .sort_values("true_metal")
    )

    return global_metrics, per_metal_df


def _bootstrap_metrics(
    system_summary_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = SEED,
    ci: float = 95.0,
) -> dict[str, dict[str, float]]:
    """
    Estimate bootstrap confidence intervals for key accuracy metrics.

    Parameters
    ----------
    system_summary_df : pd.DataFrame
        Output of ``_summarize_system_predictions``.
    n_bootstrap : int, optional
        Number of bootstrap resamples.  Pass 0 to skip bootstrapping.
    seed : int, optional
        Seed for the NumPy default RNG.
    ci : float, optional
        Confidence interval width in percent (e.g. ``95.0`` for 95% CI).

    Returns
    -------
    dict of str to dict of str to float
        For each metric key, a dict with ``lower``, ``upper``, and ``mean``
        entries.  Returns an empty dict if ``system_summary_df`` is empty or
        ``n_bootstrap <= 0``.
    """
    if system_summary_df.empty or n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(system_summary_df)
    alpha = (100.0 - ci) / 2.0

    accumulators: dict[str, list[float]] = {
        "accuracy": [],
        "mean_true_rank": [],
        "top3_accuracy": [],
        "mean_true_pred_mbg_rmsd": [],
        "mean_all_zero_pred_mbg_rmsd": [],
        "mean_true_vs_all_zero_diff": [],
        "frac_true_better_than_all_zero": [],
    }

    has_all_zero = "all_zero_pred_mbg_rmsd" in system_summary_df.columns

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = system_summary_df.iloc[idx]

        accumulators["accuracy"].append(float(sample["correct"].mean()))
        accumulators["mean_true_rank"].append(float(sample["true_rank"].mean()))
        accumulators["top3_accuracy"].append(float(sample["top3_correct"].mean()))
        accumulators["mean_true_pred_mbg_rmsd"].append(
            float(sample["true_pred_mbg_rmsd"].mean())
        )

        if has_all_zero:
            valid = sample[sample["all_zero_pred_mbg_rmsd"].notna()]
            if len(valid):
                accumulators["mean_all_zero_pred_mbg_rmsd"].append(
                    float(valid["all_zero_pred_mbg_rmsd"].mean())
                )
                diff = valid["true_pred_mbg_rmsd"] - valid["all_zero_pred_mbg_rmsd"]
                accumulators["mean_true_vs_all_zero_diff"].append(float(diff.mean()))
                accumulators["frac_true_better_than_all_zero"].append(
                    float((diff < 0).mean())
                )
            else:
                for key in (
                    "mean_all_zero_pred_mbg_rmsd",
                    "mean_true_vs_all_zero_diff",
                    "frac_true_better_than_all_zero",
                ):
                    accumulators[key].append(float("nan"))
        else:
            for key in (
                "mean_all_zero_pred_mbg_rmsd",
                "mean_true_vs_all_zero_diff",
                "frac_true_better_than_all_zero",
            ):
                accumulators[key].append(float("nan"))

    result: dict[str, dict[str, float]] = {}
    for key, values in accumulators.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            result[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
        else:
            result[key] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "ci_lower": float(np.percentile(finite, alpha)),
                "ci_upper": float(np.percentile(finite, 100.0 - alpha)),
            }
    return result


def main() -> None:
    """
    CLI entry point: run the metal identity probe on the test set.

    Loads the CatBoost model, builds the list of eligible systems, then for each
    system:
    1. Resolves receptor and ligand files.
    2. Runs VINA score-only for each candidate metal receptor.
    3. Computes CHELATE contact features on the crystal pose.
    4. Sweeps all candidate metal one-hot encodings and predicts MBG RMSD.
    5. Determines which candidate has the lowest predicted RMSD.

    Aggregates per-system results and writes CSVs and a summary JSON to
    ``--output-dir``.
    """
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    model, feature_names = _load_model(model_path)

    if args.n_poses < 1:
        raise ValueError(f"--n-poses must be >= 1, got {args.n_poses}")

    vina_bin = Path(args.vina_bin)
    if not vina_bin.exists():
        raise FileNotFoundError(f"VINA binary path does not exist: {vina_bin}")

    eligible_df, excluded_df, summary_counts = _build_eligible_systems(args)

    excluded_path = output_dir / "excluded_systems.csv"
    excluded_df.to_csv(excluded_path, index=False)

    eligible_path = output_dir / "eligible_systems.csv"
    eligible_df.to_csv(eligible_path, index=False)

    prediction_rows: list[pd.DataFrame] = []
    scoring_errors: list[dict] = []

    total = len(eligible_df)
    for idx, row in enumerate(eligible_df.itertuples(index=False), start=1):
        system_id = row.system_id
        ligand_id = row.ligand_id
        true_metal = row.true_metal

        if args.verbose or idx % 25 == 0:
            print(f"[{idx}/{total}] Processing {system_id} {ligand_id} ({true_metal})")

        try:
            receptor_pdb_path = _ensure_receptor_pdb(
                Path(row.receptor_pdb) if str(row.receptor_pdb) else None,
                Path(row.receptor_pdbqt),
            )
            ligand_crystal_pdbqt = _ensure_ligand_crystal_pdbqt(
                Path(row.ligand_sdf),
                ligand_id,
            )
            # Parse both hydrogen-free, as in feature generation for training; explicit
            # hydrogens change the Gasteiger charge features.
            receptor_mol = load_receptor_pdb(receptor_pdb_path)
            if receptor_mol is None:
                raise RuntimeError("RDKit failed to parse receptor PDB")

            lig_supplier = Chem.SDMolSupplier(str(row.ligand_sdf))
            ligand_mol = (
                lig_supplier[0] if lig_supplier and len(lig_supplier) > 0 else None
            )
            if ligand_mol is None:
                raise RuntimeError("RDKit failed to parse ligand SDF")
            ligand_mol = Chem.RemoveAllHs(ligand_mol)

            metals = [
                (
                    true_metal,
                    (
                        float(row.metal_coord_x),
                        float(row.metal_coord_y),
                        float(row.metal_coord_z),
                    ),
                )
            ]
            feat_series = compute_contact_features(
                receptor_mol,
                ligand_mol,
                metals,
                pose_idx=1,
                n_poses=args.n_poses,
            )

            base_features = feat_series.to_dict()
            metal_coord = (
                float(row.metal_coord_x),
                float(row.metal_coord_y),
                float(row.metal_coord_z),
            )
            candidate_pdbqt_dir = (
                output_dir
                / "candidate_receptor_pdbqts"
                / str(system_id)
                / str(ligand_id)
            )
            candidate_vina_scores = _score_candidates_with_vina(
                vina_bin=vina_bin,
                receptor_pdbqt=Path(row.receptor_pdbqt),
                ligand_pdbqt=ligand_crystal_pdbqt,
                metal_coord=metal_coord,
                candidate_out_dir=candidate_pdbqt_dir,
                match_tol=args.metal_coord_match_tol,
            )

            pred_df = _predict_for_candidates(
                model=model,
                feature_names=feature_names,
                base_features=base_features,
                candidate_vina_scores=candidate_vina_scores,
                true_metal=true_metal,
                system_id=system_id,
                ligand_id=ligand_id,
            )
            prediction_rows.append(pred_df)

            display_rows = pred_df[["candidate_metal", "pred_mbg_rmsd"]].copy()
            display_rows["vina_score"] = display_rows["candidate_metal"].map(
                candidate_vina_scores
            )
            for cand in [*EXPLICIT_METALS, "ALL_ZERO"]:
                sub = display_rows[display_rows["candidate_metal"] == cand]
                if sub.empty:
                    continue
                vina_val = float(sub["vina_score"].iloc[0])
                pred_val = float(sub["pred_mbg_rmsd"].iloc[0])
                star = "*" if cand == true_metal else ""
                print(
                    f"  {cand:>8}{star}  vina={vina_val:8.3f}  pred_mbg_rmsd={pred_val:8.4f}"
                )

        except Exception as exc:
            print(f"  FAILED: {exc}")
            scoring_errors.append(
                {
                    "system_id": system_id,
                    "ligand_id": ligand_id,
                    "true_metal": true_metal,
                    "error": str(exc),
                }
            )

    predictions_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(
            columns=[
                "system_id",
                "ligand_id",
                "true_metal",
                "candidate_metal",
                "pred_mbg_rmsd",
            ]
        )
    )

    predictions_path = output_dir / "metal_identity_predictions_long.csv"
    predictions_df.to_csv(predictions_path, index=False)

    errors_df = pd.DataFrame(scoring_errors)
    errors_path = output_dir / "system_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    system_summary_df = _summarize_system_predictions(predictions_df)

    system_summary_path = output_dir / "system_prediction_summary.csv"
    system_summary_df.to_csv(system_summary_path, index=False)

    global_metrics, per_metal_df = _aggregate_metrics(system_summary_df)
    per_metal_path = output_dir / "per_true_metal_accuracy.csv"
    per_metal_df.to_csv(per_metal_path, index=False)

    bootstrap_results = _bootstrap_metrics(
        system_summary_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
        ci=args.bootstrap_ci,
    )
    if bootstrap_results:
        bootstrap_path = output_dir / "bootstrap_metrics.json"
        with open(bootstrap_path, "w") as fh:
            json.dump(
                {
                    "ci_pct": args.bootstrap_ci,
                    "n_bootstrap": args.n_bootstrap,
                    "metrics": bootstrap_results,
                },
                fh,
                indent=2,
            )

    all_zero_report_df = pd.DataFrame()
    if not predictions_df.empty:
        all_zero_df = predictions_df[predictions_df["candidate_metal"] == "ALL_ZERO"][
            ["system_id", "ligand_id", "pred_mbg_rmsd"]
        ].rename(columns={"pred_mbg_rmsd": "all_zero_pred_mbg_rmsd"})
        explicit_winners = (
            predictions_df[predictions_df["candidate_metal"].isin(EXPLICIT_METALS)]
            .sort_values(
                ["system_id", "ligand_id", "pred_mbg_rmsd", "candidate_metal"],
                ascending=[True, True, True, True],
            )
            .groupby(["system_id", "ligand_id"], as_index=False)
            .first()[["system_id", "ligand_id", "candidate_metal"]]
            .rename(columns={"candidate_metal": "winner_explicit_metal"})
        )
        all_zero_report_df = explicit_winners.merge(
            all_zero_df,
            on=["system_id", "ligand_id"],
            how="left",
        )
    all_zero_report_path = output_dir / "all_zero_probe_report.csv"
    all_zero_report_df.to_csv(all_zero_report_path, index=False)

    run_summary = {
        "candidate_protocol": candidate_protocol(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "feature_count": len(feature_names),
        "pose_idx": 1,
        "n_poses": args.n_poses,
        "pose_rank_norm": 1.0 / args.n_poses,
        "counts": summary_counts,
        "successful_systems": int(system_summary_df.shape[0]),
        "failed_systems_after_filtering": int(errors_df.shape[0]),
        "global_metrics": global_metrics,
        "bootstrap": (
            {
                "ci_pct": args.bootstrap_ci,
                "n_bootstrap": args.n_bootstrap,
                "metrics": bootstrap_results,
            }
            if bootstrap_results
            else None
        ),
        "outputs": {
            "eligible_csv": str(eligible_path),
            "excluded_csv": str(excluded_path),
            "predictions_csv": str(predictions_path),
            "system_summary_csv": str(system_summary_path),
            "per_true_metal_csv": str(per_metal_path),
            "errors_csv": str(errors_path),
            "all_zero_report_csv": str(all_zero_report_path),
        },
    }

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(run_summary, fh, indent=2)

    print("=" * 80)
    print("Metal identity probe finished")
    print(f"Model: {model_path}")
    print(f"Eligible systems: {summary_counts['eligible_system_ligands']}")
    print(f"Successful systems: {system_summary_df.shape[0]}")
    print(f"Failed systems after filtering: {errors_df.shape[0]}")
    print(
        f"Global accuracy: {global_metrics['accuracy']:.4f}"
        if system_summary_df.shape[0] > 0
        else "Global accuracy unavailable (no successful systems)."
    )
    if bootstrap_results:
        ci = args.bootstrap_ci
        print(f"Bootstrap {ci:.0f}% CI (n={args.n_bootstrap}):")
        for metric in (
            "accuracy",
            "top3_accuracy",
            "mean_true_rank",
            "mean_true_pred_mbg_rmsd",
            "mean_all_zero_pred_mbg_rmsd",
            "mean_true_vs_all_zero_diff",
            "frac_true_better_than_all_zero",
        ):
            if metric not in bootstrap_results:
                continue
            b = bootstrap_results[metric]
            print(
                f"  {metric}: {b['mean']:.4f} ± {b['std']:.4f}"
                f"  [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]"
            )
        print(f"  Bootstrap details: {bootstrap_path}")
    print(f"Summary JSON: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
