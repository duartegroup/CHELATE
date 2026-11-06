"""
Description
-----------
Read docked poses and their docking scores from the output of Vina, Smina or GNINA.

Vina writes PDBQT natively; an SDF exported from it with Meeko's ``mk_export.py`` is
also accepted. Smina writes PDBQT and GNINA writes SDF (optionally gzipped). Reading
PDBQT relies on the ``REMARK SMILES`` records that Meeko adds when preparing the
ligand, so the ligand must have been prepared with Meeko.

Scores keep each program's own convention and are **not** made comparable across
programs: Vina and Smina report a binding free energy in kcal/mol where lower is
better, GNINA reports ``CNNscore`` where higher is better.
"""

import contextlib
import gzip
import json
import warnings

from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem

_SUPPORTED_METHODS = ("VINA", "SMINA", "GNINA")


def _vina_score(mol: Chem.rdchem.Mol) -> float:
    """
    Read the Vina binding free energy from a pose's ``meeko`` property.

    Meeko stores a small JSON blob on each pose it writes; ``free_energy`` is the
    docking score in kcal/mol.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        A single docked pose.

    Returns
    -------
    float
        Binding free energy in kcal/mol, or ``nan`` if the property is absent or
        unparseable.
    """
    if not mol.HasProp("meeko"):
        return float("nan")
    try:
        return float(json.loads(mol.GetProp("meeko"))["free_energy"])
    except (ValueError, KeyError, TypeError):
        return float("nan")


def _gnina_score(mol: Chem.rdchem.Mol) -> float:
    """
    Read GNINA's CNN pose score from a pose's properties.

    Unlike the Vina and Smina scores this is a quality score where **higher is
    better**.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        A single docked pose.

    Returns
    -------
    float
        ``CNNscore``, or ``nan`` if absent.
    """
    if not mol.HasProp("CNNscore"):
        return float("nan")
    try:
        return float(mol.GetProp("CNNscore"))
    except ValueError:
        return float("nan")


def _read_sdf(path: str) -> list[Chem.rdchem.Mol]:
    """
    Read every pose from an SDF file.

    All hydrogens are removed, including those RDKit's ``RemoveHs`` keeps to define
    double-bond stereochemistry; otherwise the pose graph would not match the
    hydrogen-free reference ligand and symmetry-corrected RMSDs cannot be computed.

    Parameters
    ----------
    path : str
        Path to a ``.sdf`` or ``.sdf.gz`` file.

    Returns
    -------
    list of Chem.rdchem.Mol
        Poses in file order; entries RDKit could not parse are skipped.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            supplier = Chem.ForwardSDMolSupplier(fh)
            return [Chem.RemoveAllHs(m) for m in supplier if m is not None]

    supplier = Chem.SDMolSupplier(path)
    return [Chem.RemoveAllHs(m) for m in supplier if m is not None]


def read_poses(path: str, method: str) -> tuple[list[Chem.rdchem.Mol], list[float]]:
    """
    Read docked poses and their docking scores.

    Parameters
    ----------
    path : str
        Docking output file: PDBQT for Vina and Smina (a Vina SDF exported with
        Meeko's ``mk_export.py`` also works), SDF (``.sdf`` or ``.sdf.gz``) for GNINA.
    method : str
        Docking program that produced the file: ``'VINA'``, ``'SMINA'`` or ``'GNINA'``.

    Returns
    -------
    tuple of (list of Chem.rdchem.Mol, list of float)
        ``(poses, scores)`` in file order, i.e. the docking program's own ranking, with
        one score per pose. Poses whose score could not be read get ``nan`` rather than
        being dropped, so the two lists always align.

    Raises
    ------
    ValueError
        If ``method`` is not one of the three supported programs.
    """
    method = method.upper()
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown docking method: {method!r}. Expected one of {_SUPPORTED_METHODS}."
        )

    if method == "SMINA":
        return _read_pdbqt(path, "REMARK minimizedAffinity", score_field=-1)
    if method == "VINA" and str(path).lower().endswith(".pdbqt"):
        return _read_pdbqt(path, "REMARK VINA RESULT:", score_field=3)

    mols = _read_sdf(path)
    score_fn = _gnina_score if method == "GNINA" else _vina_score
    return mols, [score_fn(m) for m in mols]


def _read_pdbqt(
    path: str, score_remark: str, score_field: int
) -> tuple[list[Chem.rdchem.Mol], list[float]]:
    """
    Read Vina or Smina poses from a multi-model PDBQT file.

    Parameters
    ----------
    path : str
        Path to the docking output PDBQT file.
    score_remark : str
        Prefix of the remark line holding the score: ``REMARK VINA RESULT:`` for
        Vina, ``REMARK minimizedAffinity`` for Smina.
    score_field : int
        Index of the score in the whitespace-split remark line.

    Returns
    -------
    tuple of (list of Chem.rdchem.Mol, list of float)
        ``(poses, scores)`` in file order.
    """
    mols: list[Chem.rdchem.Mol] = []
    scores: list[float] = []

    def add_pose(lines: list[str]) -> None:
        if not any(line.startswith(("ATOM", "HETATM")) for line in lines):
            warnings.warn(
                f"Ignoring an empty PDBQT model block in {path}", stacklevel=2
            )
            return
        # Meeko converts every pose in its input into conformers and changes its
        # iterator position. Give it one model at a time to retain file order.
        pdbqt_mol = PDBQTMolecule("".join(lines), skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        mol = next((m for m in rdkit_mols if m is not None), None)
        if mol is None:
            return

        score = float("nan")
        for line in lines:
            if line.startswith(score_remark):
                with contextlib.suppress(ValueError, IndexError):
                    score = float(line.split()[score_field])
                break

        mols.append(Chem.RemoveAllHs(mol))
        scores.append(score)

    with open(path) as fh:
        pose_lines: list[str] = []
        for line in fh:
            if line.startswith("MODEL"):
                pose_lines = []
            pose_lines.append(line)
            if line.startswith("ENDMDL"):
                add_pose(pose_lines)
                pose_lines = []

        # Also accept a single pose without MODEL/ENDMDL records.
        if any(line.startswith(("ATOM", "HETATM")) for line in pose_lines):
            add_pose(pose_lines)

    return mols, scores
