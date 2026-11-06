"""
Description
-----------
This module contains functionality to perform various docking utilities and manipulations, including handling PDB and CIF files, preparing ligands and receptors for docking, calculating RMSD, and more.
"""

import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.spatial
from biotite.database.rcsb import fetch
from biotite.structure import array as biotite_array
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from cif_utils import sanitize_asym_id_columns_in_cif as _sanitize_asym_ids
from docking_params import (
    AUTODOCK_LIGAND_TYPES,
    AUTODOCK_RECEPTOR_TYPES,
    CHARGE_DICT,
    GPDOCK_MULTI_METAL_PATH,
    GPDOCK_PATH,
    METAL_ATOMS_SET,
    METALLODOCK_ENV,
    METALLODOCK_PATH,
    mk_prepare_ligand,
    mk_prepare_receptor,
    reduce2,
    scrub,
)
from meeko import PDBQTMolecule, RDKitMolCreate
from meeko.gridbox import get_gpf_string
from meeko.writer import PDBQTWriterLegacy
from openbabel import pybel
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem
from spyrmsd.molecule import Molecule as SpyMolecule
from spyrmsd.rmsd import symmrmsd

from chelate.poses import read_poses
from docking.pose_ranking import rank_by_energy
from docking.prepare_ligand import preparation_command

logger = logging.getLogger(__name__)

# PyMOL's ``cmd`` is a single global session; run_docking.py calls get_box from a
# thread pool, so access to it is serialised.
_PYMOL_LOCK = threading.Lock()


def split_by_meeko_support(pdb_path: str) -> tuple[str, str]:
    """
    Split a PDB file into two separate files: one containing only metal atoms supported by Meeko and the other containing
    unsupported metal atoms by Meeko. The supported atoms are saved as a PDB file, while the unsupported atoms are converted
    and saved as a PDBQT file, which will later be merged.

    The supported atoms are saved to a new file with the suffix "_wo_meekoMetals.pdb" appended to the original file name.
    The unsupported atoms are saved to a new file with the suffix "_meekoMetals.pdbqt" appended to the original file name.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.

    Returns
    -------
    tuple [str, str]
        A tuple containing the paths to the two output files: one containing only supported metal atoms and the other
        containing unsupported metal atoms.
    """
    # Load PDB file with RDKit
    mol: Chem.rdchem.Mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)

    if mol is None:
        raise ValueError(f"Failed to load PDB file: {pdb_path}")

    supported_atoms: list[Chem.rdchem.Atom] = []
    unsupported_atoms: list[Chem.rdchem.Atom] = []
    unsupported_serials: list[int] = []

    # Create a new editable molecule
    mol_editable: Chem.RWMol = Chem.RWMol(mol)

    for atom in mol.GetAtoms():
        element: str = atom.GetSymbol().upper()
        if element in CHARGE_DICT:
            unsupported_atoms.append(atom)
            unsupported_serials.append(atom.GetIdx())
        else:
            supported_atoms.append(atom)

    # Remove unsupported atoms from the new molecule
    for atom_idx in reversed(unsupported_serials):  # Reverse to prevent index shifts
        mol_editable.RemoveAtom(atom_idx)

    # Convert edited molecule back to RDKit format
    final_supported_mol: Chem.rdchem.Mol = mol_editable.GetMol()

    # Save the molecule without unsupported metals
    pdb_wo_meeko_metals_file: str = pdb_path.replace(".pdb", "_wo_meekoMetals.pdb")
    Chem.MolToPDBFile(final_supported_mol, pdb_wo_meeko_metals_file)

    # Write unsupported metals to a PDBQT file
    meeko_metals_file: str = pdb_path.replace(".pdb", "_meekoMetals.pdbqt")

    with open(meeko_metals_file, "w") as file:
        for atom in unsupported_atoms:
            pos: Chem.rdGeometry.Point3D = mol.GetConformer().GetAtomPosition(
                atom.GetIdx()
            )
            serial_num: int = atom.GetIdx() + 1  # Atom serial number
            charge: float = CHARGE_DICT.get(atom.GetSymbol().upper(), 0)
            capitalized_symbol: str = atom.GetSymbol().upper()

            formatted_string: str = (
                f"ATOM  {serial_num:>5} {capitalized_symbol:<3}  {capitalized_symbol:<2}  _   0    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    {charge:6.3f} {atom.GetSymbol():>2}"
            )
            file.write(formatted_string + "\n")

    return meeko_metals_file, pdb_wo_meeko_metals_file


def merge_pdbs(pdb1_path: str, pdb2_path: str, output_path: str) -> bool:
    """
    Merge two PDB files into a single PDB file using the Biotite library.

    Parameters
    ----------
    pdb1_path : str
        Path to the first PDB file.
    pdb2_path : str
        Path to the second PDB file.
    output_path : str
        Path to save the output PDB file.

    Returns
    -------
    bool
        True if the output file was successfully created, False otherwise.
    """
    try:
        structure1: biotite_array = PDBFile.read(pdb1_path).get_structure()[0]
        structure2: biotite_array = PDBFile.read(pdb2_path).get_structure()[0]

        merged: biotite_array = structure1 + structure2

        pdb_file: PDBFile = PDBFile()
        pdb_file.set_structure(merged)
        pdb_file.write(output_path)

        return os.path.exists(output_path)
    except FileNotFoundError as e:
        logger.error(f"Error: File not found - {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred while merging PDB files: {e}")
        return False


def merge_pdbqt_files(output_pdbqt: str, *input_pdbqts: str) -> str:
    """
    Merge multiple PDBQT files into one, preserving all contents without modification.

    Parameters
    ----------
    output_pdbqt : str
        Path to the output merged PDBQT file.
    *input_pdbqts : str
        Paths to the input PDBQT files to be merged.

    Returns
    -------
    bool
        True if the output file was successfully created, False otherwise.
    """
    try:
        with open(output_pdbqt, "w") as outfile:
            for pdbqt_file in input_pdbqts:
                with open(pdbqt_file) as infile:
                    contents = infile.read()
                    outfile.write(contents)

        print(
            f"Successfully merged {len(input_pdbqts)} PDBQT files into '{output_pdbqt}'"
        )
        return os.path.exists(output_pdbqt)

    except Exception as e:
        logger.error(f"Error merging PDBQT files: {e}")
        return False


def remove_unsupported_atoms_cif(
    input_cif: str, cleaned_cif: str, removed_atoms_cif: str
) -> bool:
    """
    Remove atoms unsupported by Meeko from a CIF file.

    Parameters
    ----------
    input_cif : str
        Path to the input CIF file.
    cleaned_cif : str
        Path to save the cleaned CIF file without unsupported atoms.
    removed_atoms_cif : str
        Path to save the CIF file containing only the removed atoms.

    Returns
    -------
    bool
        True if the output files were successfully created, False otherwise.
    """
    try:
        with open(input_cif) as infile:
            lines: list[str] = infile.readlines()

        cleaned_lines: list[str] = []
        removed_atoms: list[str] = []
        removed_cnt: int = 0

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                columns: list[str] = line.split()
                # Match on _atom_site.type_symbol, or on _atom_site.label_comp_id for
                # ions such as FE2. The residue-name match is restricted to metal atoms,
                # since names like U (uridine) or NO (nitric oxide) are also CCD codes.
                if len(columns) > 3 and (
                    columns[1] in CHARGE_DICT
                    or (columns[1] in METAL_ATOMS_SET and columns[3] in CHARGE_DICT)
                ):
                    removed_atoms.append(line)  # Store removed atom
                    removed_cnt += 1
                    continue  # Skip adding to cleaned CIF

            cleaned_lines.append(line)  # Keep supported atoms and all other lines

        # Write cleaned CIF file (without unsupported atoms)
        with open(cleaned_cif, "w") as outfile:
            outfile.writelines(cleaned_lines)

        # Write removed atoms to a separate CIF file (only ATOM/HETATM lines)
        with open(removed_atoms_cif, "w") as outfile:
            outfile.writelines(removed_atoms)

        print(
            f"INFO: Removed {removed_cnt} unsupported atoms which are converted manually"
        )
        return True

    except Exception as e:
        logger.error(f"Error processing CIF file: {e}")
        return False


def remove_ligand_cif(system_cif: str, ligand_id: str, out_path: str) -> bool:
    """
    Remove all lines from a CIF file where _atom_site.label_asym_id matches ligand_id.

    Parameters
    ----------
    system_cif : str
        Path to the input CIF file.
    ligand_id : str
        The ligand chain identifier (_atom_site.label_asym_id) to be removed.
    out_path : str
        Path to save the output CIF file without the specified ligand.

    Returns
    -------
    bool
        True if the output file was successfully created, False otherwise.
    """
    try:
        with open(system_cif) as infile:
            lines: list[str] = infile.readlines()

        cleaned_lines: list[str] = []
        removed_cnt: int = 0

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                columns: list[str] = line.split()
                if (
                    len(columns) > 4 and columns[4] == ligand_id
                ):  # Check _atom_site.label_asym_id
                    removed_cnt += 1
                    continue  # Skip ligand lines

            cleaned_lines.append(line)  # Keep all non-ligand lines

        # Write the cleaned CIF file
        with open(out_path, "w") as outfile:
            outfile.writelines(cleaned_lines)

        print(
            f"INFO: Removed {removed_cnt} atoms belonging to ligand '{ligand_id}' from receptor"
        )
        return True

    except Exception as e:
        logger.error(f"Error processing CIF file: {e}")
        return False


def remove_solvent_cif(system_cif: str, output_path: str) -> bool:
    """
    This function removes solvent molecules from a CIF file. The cleaned CIF file is then saved to the specified
    output path. The following solvent residues are considered and removed:
    - "HOH"
    - "WAT"
    - "H2O"
    - "TIP"
    - "SOL"

    Parameters
    ----------
    system_cif : str
        Path to the input CIF file.
    output_path : str
        Path to save the output CIF file.

    Returns
    -------
    bool
        True if the output file was successfully created, False otherwise.
    """
    solvent_residues: set[str] = {"HOH", "WAT", "H2O", "TIP", "SOL"}

    try:
        with open(system_cif) as infile:
            lines: list[str] = infile.readlines()

        cleaned_lines: list[str] = []
        removed_cnt: int = 0

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                columns: list[str] = line.split()
                if len(columns) > 3 and columns[3] in solvent_residues:
                    removed_cnt += 1
                    continue  # Skip solvent molecule lines

            cleaned_lines.append(line)  # Keep all non-solvent lines

        # Write the cleaned CIF file
        with open(output_path, "w") as outfile:
            outfile.writelines(cleaned_lines)

        print(f"INFO: Removed {removed_cnt} solvent molecules from receptor")
        return True

    except Exception as e:
        logger.error(f"Error processing CIF file: {e}")
        return False


def sanitize_asym_id_columns_in_cif(input_cif: str, output_path: str) -> bool:
    """Map complete atom-site chain IDs to unique one-character identifiers."""
    return _sanitize_asym_ids(input_cif, output_path)


def convert_unsupported_cif_to_pdbqt(
    input_cif: str, output_pdbqt: str, format_type: str = "BIOTITE"
) -> bool:
    """
    Convert a CIF file containing removed atoms into PDBQT format using proper formatting.

    Parameters
    ----------
    input_cif : str
        Path to the input CIF file containing removed atoms.
    output_pdbqt : str
        Path to save the converted PDBQT file.
    format_type : str, default="BIOTITE"
        The format type of the input CIF file. Either "BIOTITE" or "PLINDER".

    Returns
    -------
    bool
        True if the conversion was successful, False otherwise.
    """
    try:
        # Validate format type
        format_type = format_type.upper()
        if format_type not in ["BIOTITE", "PLINDER"]:
            raise ValueError(
                f"Unsupported format type: {format_type}. Must be either 'BIOTITE' or 'PLINDER'"
            )

        with open(input_cif) as infile:
            lines: list[str] = infile.readlines()

        pdbqt_lines: list[str] = []
        atom_count: int = 0  # Start atom numbering from 1

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                columns: list[str] = line.split()

                # Extract values based on format type
                if format_type == "BIOTITE":
                    atom_type: str = columns[
                        1
                    ]  # _atom_site.type_symbol (AutoDock atom type)
                    atom_name: str = columns[2]  # _atom_site.label_atom_id
                    res_name: str = columns[4]  # _atom_site.label_comp_id
                    chain: str = columns[5]  # _atom_site.label_asym_id
                    res_num: int = (
                        int(columns[7]) if columns[7] != "." else 0
                    )  # _atom_site.label_seq_id
                    coord: tuple[float, float, float] = (
                        float(columns[17]),
                        float(columns[18]),
                        float(columns[19]),
                    )  # X, Y, Z coordinates
                else:  # PLINDER
                    atom_type: str = columns[
                        1
                    ]  # _atom_site.type_symbol (AutoDock atom type)
                    atom_name: str = columns[2]  # _atom_site.label_atom_id
                    res_name: str = columns[3]  # _atom_site.label_comp_id
                    chain: str = columns[4][
                        -1
                    ]  # _atom_site.label_asym_id (Only use last char)
                    res_num: int = (
                        int(columns[6]) if columns[6] != "." else 0
                    )  # _atom_site.label_seq_id
                    coord: tuple[float, float, float] = (
                        float(columns[8]),
                        float(columns[9]),
                        float(columns[10]),
                    )  # X, Y, Z coordinates

                atom_type = atom_type.capitalize()  # AutoDock types are case-sensitive

                # Get charge from CHARGE_DICT (default to 0.000 if not found)
                charge: float = CHARGE_DICT.get(res_name.upper(), 0.000)

                # Generate properly formatted PDBQT line
                atom_count += 1
                pdbqt_line: str = PDBQTWriterLegacy._make_pdbqt_line(
                    atom_count,
                    atom_name,
                    res_name,
                    chain,
                    res_num,
                    coord,
                    charge,
                    atom_type,
                )
                pdbqt_lines.append(pdbqt_line)

        # Write the converted PDBQT file
        with open(output_pdbqt, "w") as outfile:
            outfile.writelines("\n".join(pdbqt_lines) + "\n")  # Ensure proper newlines

        print(f"INFO: Successfully converted CIF to PDBQT and saved to {output_pdbqt}")
        return True

    except Exception as e:
        logger.error(f"Error processing CIF file: {e}")
        return False


def process_gnina_poses(docked_sdf_gz_path: str) -> list[Chem.rdchem.Mol]:
    """
    Process docked poses from a GNINA-generated SDF GZ file.
    Hydrogens are removed from the docked molecules.

    Parameters
    ----------
    docked_sdf_gz_path : str
        Path to the docked SDF GZ file.

    Returns
    -------
    list[Chem.rdchem.Mol]
        List of docked molecules.
    """
    return read_poses(docked_sdf_gz_path, "GNINA")[0]


def process_vina_poses(docked_sdf_path: str) -> list[Chem.rdchem.Mol]:
    """
    Process docked poses from a VINA-generated SDF file.
    Hydrogens are removed from the docked molecules.

    Parameters
    ----------
    docked_sdf_path : str
        Path to the docked SDF file.

    Returns
    -------
    list[Chem.rdchem.Mol]
        List of docked molecules.
    """
    return read_poses(docked_sdf_path, "VINA")[0]


def process_vina_pdbqt_poses(docked_pdbqt_path: str) -> list[Chem.rdchem.Mol]:
    """
    Process docked poses from a VINA-generated PDBQT file by converting to SDF.
    Hydrogens are removed from the docked molecules.

    Parameters
    ----------
    docked_pdbqt_path : str
        Path to the docked PDBQT file.

    Returns
    -------
    list[Chem.rdchem.Mol]
        List of docked molecules.
    """
    docked_sdf_path: str = os.path.splitext(docked_pdbqt_path)[0] + ".sdf"
    subprocess.run(["mk_export.py", docked_pdbqt_path, "-s", docked_sdf_path])

    return process_vina_poses(docked_sdf_path)


def process_smina_poses(
    docked_pdbqt_path: str,
) -> tuple[list[Chem.rdchem.Mol], list[float]]:
    """
    Process docked poses from a SMINA-generated PDBQT file.
    Hydrogens are removed from the docked molecules.

    Parameters
    ----------
    docked_pdbqt_path : str
        Path to the docked PDBQT file.

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[float]]
        A tuple containing a list of docked molecules and a list of their affinities.
    """
    return read_poses(docked_pdbqt_path, "SMINA")


def process_autodock_poses(dlg_path: str) -> tuple[list[Chem.rdchem.Mol], list[float]]:
    """
    Process docked poses from an AutoDock DLG file.

    Parameters
    ----------
    dlg_path : str
        Path to the AutoDock DLG file.

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[float]]
        Tuple containing a list of docked molecules and a list of their docking scores.
    """
    with open(dlg_path) as f:
        lines = f.readlines()

    docked_lines: list[str] = [
        line[8:] for line in lines if line.startswith("DOCKED: ")
    ]

    docking_scores: list[float] = []
    for line in docked_lines:
        if "Estimated Free Energy of Binding" in line:
            try:
                parts = line.split()
                idx = parts.index("=")
                docking_score = float(parts[idx + 1])
                docking_scores.append(docking_score)
            except ValueError:
                pass

    # Split by MODEL
    models: list[str] = []
    current_model: list[str] = []
    in_model: bool = False
    for line in docked_lines:
        if line.startswith("MODEL"):
            in_model = True
            current_model = [line]
        elif line.startswith("ENDMDL"):
            if in_model:
                current_model.append(line)
                models.append("".join(current_model))
                in_model = False
        elif in_model:
            current_model.append(line)

    if len(docking_scores) != len(models):
        raise ValueError(
            f"DLG file formatting error: Found {len(docking_scores)} scores but {len(models)} models."
        )

    valid_mols: list[Chem.rdchem.Mol] = []
    valid_scores: list[float] = []

    # Check 2: Iterate through them together using zip()
    for model_str, score in zip(models, docking_scores, strict=False):
        try:
            pdbqt_mol: PDBQTMolecule = PDBQTMolecule(model_str, skip_typing=True)
            rdkit_mols: list[Chem.rdchem.Mol] = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

            if rdkit_mols:
                valid_mols.append(Chem.RemoveAllHs(rdkit_mols[0]))
                valid_scores.append(
                    score
                )  # Only append the score if the molecule was successful
            else:
                print(
                    "Warning: RDKit returned an empty list. Skipping molecule and its score."
                )

        except Exception as e:
            print(
                f"Error parsing AutoDock model: {e}. Skipping molecule and its score."
            )

    if len(valid_mols) != len(valid_scores):
        print("Mismatch between valid molecules and scores!")

    return rank_by_energy(valid_mols, valid_scores)


def process_gpdock_poses(
    pdb_path: str, original_sdf_path: str
) -> tuple[list[Chem.rdchem.Mol], list[int]]:
    """
    Process docked poses from a GPDOCK PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the GPDOCK PDB file.
    original_sdf_path : str
        Path to the original SDF file (used for topology).

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[int]]
        A tuple containing a list of docked molecules and a list of their ranks.
    """
    # Load reference for topology
    ref_mol: Chem.rdchem.Mol = Chem.SDMolSupplier(original_sdf_path, removeHs=False)[0]
    if ref_mol is None:
        ref_mol = Chem.SDMolSupplier(original_sdf_path, removeHs=True)[0]
    if ref_mol is None:
        print(f"Failed to load reference ligand: {original_sdf_path}")
        return [], []

    with open(pdb_path) as f:
        content: str = f.read()
    content = content.replace("ENDMOL", "ENDMDL")
    lines: list[str] = content.splitlines()
    models: list[str] = []
    current_model_lines: list[str] = []
    in_model: bool = False
    if "MODEL" not in content:
        models.append(content)
    else:
        for line in lines:
            if line.startswith("MODEL"):
                in_model = True
                current_model_lines = [line]
            elif line.startswith("ENDMDL"):
                if in_model:
                    current_model_lines.append(line)
                    models.append("\n".join(current_model_lines))
                    in_model = False
            elif in_model:
                current_model_lines.append(line)

    mols: list[Chem.rdchem.Mol] = []
    for i, model_block in enumerate(models):
        mol_pdb = Chem.MolFromPDBBlock(model_block, removeHs=False)

        if mol_pdb is None:
            print(f"Failed to parse GPDOCK model {i + 1}")
            continue

        try:
            # Try to assign bond orders
            new_mol: Chem.rdchem.Mol = AllChem.AssignBondOrdersFromTemplate(
                ref_mol, mol_pdb
            )
            mols.append(new_mol)
        except Exception:
            # Fallback: Copy coordinates to reference topology
            try:
                if mol_pdb.GetNumAtoms() == ref_mol.GetNumAtoms():
                    conf: Chem.Conformer = Chem.Conformer(ref_mol.GetNumAtoms())
                    for atom_idx in range(ref_mol.GetNumAtoms()):
                        pos = mol_pdb.GetConformer().GetAtomPosition(atom_idx)
                        conf.SetAtomPosition(atom_idx, pos)

                    new_mol: Chem.rdchem.Mol = Chem.Mol(ref_mol)
                    new_mol.RemoveAllConformers()
                    new_mol.AddConformer(conf)
                    mols.append(new_mol)
                else:
                    mols.append(mol_pdb)
            except Exception as e:
                print(f"Coordinate transfer failed for model {i + 1}: {e}")
                mols.append(mol_pdb)

    docking_scores: list[int] = list(range(1, len(mols) + 1))
    mols: list[Chem.rdchem.Mol] = [Chem.RemoveAllHs(mol) for mol in mols]
    return mols, docking_scores


def sdf_to_pdb_ligand(sdf_path: str, out_pdb: str) -> bool:
    """
    Convert a ligand SDF file to PDB format using OpenBabel.

    Parameters
    ----------
    sdf_path : str
        Path to the input SDF file.
    out_pdb : str
        Path to write the output PDB file.

    Returns
    -------
    bool
        True if conversion succeeds and the output file exists, False otherwise.
    """
    try:
        mol = next(pybel.readfile("sdf", sdf_path))
        mol.write("pdb", out_pdb, overwrite=True)
        return os.path.exists(out_pdb)
    except Exception as e:
        logger.error(f"Error converting SDF to PDB ({sdf_path}): {e}")
        return False


def add_conect_to_pdb(pdb_path: str) -> bool:
    """
    Add CONECT records for metal coordination bonds to a PDB file in-place.

    MetalloDock's feature extractor calls `atom.bonded_atoms` on each metal atom
    in the pocket to discover coordinating protein residues. MDAnalysis only
    populates `bonded_atoms` when CONECT records are present — it does not
    guess bonds from distances by default.

    This function reads the PDB, finds all metal atoms (by element symbol),
    finds all donor atoms (N, O, S) within a coordination cutoff (3.0 Å), and
    appends the corresponding CONECT records.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file to fix (CONECT records appended in place).

    Returns
    -------
    bool
        True if the file was successfully written, False otherwise.
    """
    DONOR_ELEMENTS = {"N", "O", "S"}
    COORD_CUTOFF = 3.0  # Å — typical metal coordination bond length upper bound

    try:
        with open(pdb_path) as f:
            lines: list[str] = f.readlines()

        # Strip existing CONECT / END lines; we will re-append
        atom_lines: list[str] = [
            line for line in lines if not line.startswith(("CONECT", "END"))
        ]

        serials: list[int] = []
        coords: list[list[float]] = []
        elements: list[str] = []

        for line in atom_lines:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                serial: int = int(line[6:11])
                x: float = float(line[30:38])
                y: float = float(line[38:46])
                z: float = float(line[46:54])
                element: str = (
                    line[76:78].strip().upper()
                    if len(line) > 76
                    else line[12:16].strip().lstrip("0123456789").upper()
                )
            except ValueError:
                print(f"Warning: Skipping malformed line in {pdb_path}: {line.strip()}")
                continue
            serials.append(serial)
            coords.append([x, y, z])
            elements.append(element)

        if not serials:
            return True  # nothing to do

        coords_arr = np.array(coords)
        conect_map: dict[int, list[int]] = {}  # serial → [bonded serials]

        for i, elem in enumerate(elements):
            if elem not in METAL_ATOMS_SET:
                continue
            metal_coord: list[float] = coords_arr[i]
            dists = np.linalg.norm(coords_arr - metal_coord, axis=1)
            for j, (dist, donor_elem) in enumerate(zip(dists, elements, strict=False)):
                if i == j:
                    continue
                if donor_elem in DONOR_ELEMENTS and dist <= COORD_CUTOFF:
                    conect_map.setdefault(serials[i], []).append(serials[j])
                    conect_map.setdefault(serials[j], []).append(serials[i])

        conect_lines: list[str] = []
        for src_serial, bonded in conect_map.items():
            # PDB CONECT records hold max 4 partners per line
            for chunk_start in range(0, len(bonded), 4):
                chunk = bonded[chunk_start : chunk_start + 4]
                partners = "".join(f"{s:5d}" for s in chunk)
                conect_lines.append(f"CONECT{src_serial:5d}{partners}\n")

        with open(pdb_path, "w") as f:
            f.writelines(atom_lines)
            f.writelines(conect_lines)
            f.write("END\n")

        return True

    except Exception as e:
        logger.error(f"Error adding CONECT records to {pdb_path}: {e}")
        return False


def extract_pdb_chain(pdb_path: str, chain_id: str, out_pdb: str) -> bool:
    """
    Extract all atoms belonging to a specific chain from a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    chain_id : str
        Chain identifier to extract (single character after CIF sanitization).
    out_pdb : str
        Path to write the extracted chain as a PDB file.

    Returns
    -------
    bool
        True if the output file was created with at least one atom, False otherwise.
    """
    try:
        structure: PDBFile = PDBFile.read(pdb_path).get_structure()[0]
        chain_atoms: biotite_array = structure[structure.chain_id == chain_id]
        if len(chain_atoms) == 0:
            logger.error(f"No atoms found for chain '{chain_id}' in {pdb_path}")
            return False
        pdb_out: PDBFile = PDBFile()
        pdb_out.set_structure(chain_atoms)
        pdb_out.write(out_pdb)
        return os.path.exists(out_pdb)
    except Exception as e:
        logger.error(f"Error extracting chain '{chain_id}' from {pdb_path}: {e}")
        return False


def run_metallodock_subprocess(script_name: str, args: dict) -> bool:
    """
    Run a MetalloDock utility script inside the MetalloDock conda environment.

    Parameters
    ----------
    script_name : str
        Name of the script to run (e.g. 'pre_process.py'), resolved relative to
        METALLODOCK_PATH/utils/.
    args : dict
        Mapping of CLI flag → value (e.g. {'--complex_path': '/some/dir'}).
        Flags with empty-string values are passed as bare flags.

    Returns
    -------
    bool
        True if the subprocess exits with code 0, False otherwise.
    """
    if not os.path.exists(METALLODOCK_ENV):
        raise OSError(f"MetalloDock conda environment not found at: {METALLODOCK_ENV}.")
    script_path = os.path.join(METALLODOCK_PATH, "utils", script_name)
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        METALLODOCK_ENV,
        "python",
        script_path,
    ]
    for flag, value in args.items():
        cmd.append(flag)
        if value != "":
            cmd.append(str(value))
    try:
        result = subprocess.run(cmd, check=True, timeout=90 * 60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"MetalloDock timed out after 90 minutes ({script_name})")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"MetalloDock subprocess failed ({script_name}): {e}")
        return False


def process_metallodock_poses(
    sdf_path: str,
) -> tuple[list[Chem.rdchem.Mol], list[None]]:
    """
    Load the single pose produced by MetalloDock (MMFF-corrected _ff.sdf).

    Parameters
    ----------
    sdf_path : str
        Path to the MetalloDock output SDF file.

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[None]]
        A one-element list containing the docked molecule and a matching list of
        None scores (the MDN score is not written to the SDF).
    """
    suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols: list[Chem.rdchem.Mol] = [
        Chem.RemoveAllHs(mol) for mol in suppl if mol is not None
    ]
    if not mols:
        raise ValueError(f"No valid molecules found in MetalloDock output: {sdf_path}")
    return mols, [None] * len(mols)


def process_multi_metallodock_poses(
    sdf_path: str,
) -> tuple[list[Chem.rdchem.Mol], list[float]]:
    """
    Load the multiple poses produced by MetalloDock from a single SDF file.

    Parameters
    ----------
    sdf_path : str
        Path to the MetalloDock output SDF file.

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[float]]
        A tuple containing the docked molecules and a matching list of
        MDN scores (parsed from the SDF properties).
    """
    suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mols: list[Chem.rdchem.Mol] = []
    scores: list[float] = []
    for mol in suppl:
        if mol is not None:
            mol = Chem.RemoveAllHs(mol)
            # Try to get the specific min_score or fallback to mdn_score
            if mol.HasProp("min_score"):
                try:
                    score = float(mol.GetProp("min_score"))
                except (ValueError, KeyError, RuntimeError):
                    score = (
                        float(mol.GetProp("mdn_score"))
                        if mol.HasProp("mdn_score")
                        else None
                    )
            else:
                score = (
                    float(mol.GetProp("mdn_score"))
                    if mol.HasProp("mdn_score")
                    else None
                )

            if score is not None:
                scores.append(score)
                mols.append(mol)

    if not mols:
        raise ValueError(f"No valid molecules found in MetalloDock output: {sdf_path}")
    return mols, scores


def get_original_ligand(original_sdf_path: str) -> Chem.rdchem.Mol:
    """
    Load the original ligand from an SDF file.
    Hydrogens are removed from the ligand.

    Parameters
    ----------
    original_sdf_path : str
        Path to the original SDF file.

    Returns
    -------
    Chem.rdchem.Mol
        The original ligand molecule.

    Raises
    ------
    ValueError
        If the ligand cannot be loaded or does not have 3D coordinates.
    """
    original_ligand: Chem.rdchem.Mol = Chem.SDMolSupplier(
        original_sdf_path, removeHs=False
    )[0]
    if original_ligand is None:
        raise ValueError("Failed to load the original ligand from the SDF file.")
    if not original_ligand.GetConformer().Is3D():
        raise ValueError("The original ligand does not have 3D coordinates.")
    return Chem.RemoveAllHs(original_ligand)


def get_docked_mols(
    docked_path: str, original_sdf_path: str, docking_type: str
) -> tuple[list[Chem.rdchem.Mol], list[float], list[float]]:
    """
    Process the docked poses from files generated by various docking programs.

    Parameters
    ----------
    docked_path : str
        Path to the docked file. The format depends on the docking_type.
    original_sdf_path : str
        Path to the original SDF file.
    docking_type : str
        Type of docking software used. Must be one of: 'vina', 'vina_pdbqt', 'smina', 'gnina'.

    Returns
    -------
    tuple[list[Chem.rdchem.Mol], list[float], list[float]]
        A tuple containing:
        - A list of docked molecules
        - A list of docking scores
        - A list of RMSD values

    Raises
    ------
    ValueError
        If an invalid docking_type is provided.
    """
    docked_mols: list[Chem.rdchem.Mol] = []
    docking_scores: list[float] = []
    rmsds: list[float] = []

    if docking_type.lower() == "gnina":
        docked_mols: list[Chem.rdchem.Mol] = process_gnina_poses(docked_path)
        for docked_mol in docked_mols:
            symm_rmsd: float
            symm_rmsd, _ = calculate_symmetry_corrected_rmsd(
                original_sdf_path, docked_mol
            )
            minimizedAffinity: float = float(docked_mol.GetProp("minimizedAffinity"))
            docking_scores.append(minimizedAffinity)
            rmsds.append(symm_rmsd)

    elif docking_type.lower() == "vina":
        docked_mols: list[Chem.rdchem.Mol] = process_vina_poses(docked_path)
        for docked_mol in docked_mols:
            symm_rmsd: float
            symm_rmsd, _ = calculate_symmetry_corrected_rmsd(
                original_sdf_path, docked_mol
            )
            meeko_property: str = docked_mol.GetProp("meeko")
            meeko_data: dict = json.loads(meeko_property)
            docking_score: float = meeko_data["free_energy"]
            docking_scores.append(docking_score)
            rmsds.append(symm_rmsd)

    elif docking_type.lower() == "vina_pdbqt":
        docked_mols: list[Chem.rdchem.Mol] = process_vina_pdbqt_poses(docked_path)
        for docked_mol in docked_mols:
            symm_rmsd: float
            symm_rmsd, _ = calculate_symmetry_corrected_rmsd(
                original_sdf_path, docked_mol
            )
            meeko_property: str = docked_mol.GetProp("meeko")
            meeko_data: dict = json.loads(meeko_property)
            docking_score: float = meeko_data["free_energy"]
            docking_scores.append(docking_score)
            rmsds.append(symm_rmsd)

    elif docking_type.lower() == "smina":
        docked_mols: list[Chem.rdchem.Mol]
        affinities: list[float]
        docked_mols, affinities = process_smina_poses(docked_path)
        for i, docked_mol in enumerate(docked_mols):
            symm_rmsd: float
            symm_rmsd, _ = calculate_symmetry_corrected_rmsd(
                original_sdf_path, docked_mol
            )
            docking_scores.append(affinities[i])
            rmsds.append(symm_rmsd)

    else:
        raise ValueError(
            f"Invalid docking_type: {docking_type}. Must be one of: 'vina', 'vina_pdbqt', 'smina', 'gnina'."
        )

    return docked_mols, docking_scores, rmsds


def get_docking_data(
    docked_path: str,
    original_sdf_path: str,
    metal_coords: list[tuple[float, float, float]],
    docking_type: str,
    *,
    docked_mols: list[Chem.rdchem.Mol] | None = None,
    docking_scores: list[float] | None = None,
) -> pd.DataFrame:
    """
    Process docked poses and generate analysis data for various docking programs.

    Parameters
    ----------
    docked_path : str
        Path to the docked file (format depends on docking_type).
    original_sdf_path : str
        Path to the original SDF file.
    metal_coords : list[tuple[float, float, float]]
        List of metal atom coordinates [x, y, z], one entry per metal in the site.
        Metal-binding-group RMSD selects atoms by distance to the nearest metal,
        so pass every metal in the site.
    docking_type : str
        Type of docking: 'vina', 'vina_pdbqt', 'vina_sdf', 'smina', or 'gnina'.
    docked_mols : list[Chem.rdchem.Mol], optional
        Already parsed Vina, Smina or GNINA poses in file order. When supplied,
        skip reading the docked file again.
    docking_scores : list[float], optional
        Scores aligned with ``docked_mols``. Required for pre-parsed Smina poses;
        Vina and GNINA metadata are read from the molecule properties.

    Returns
    -------
    pd.DataFrame
        Analysis data for each pose.
    """
    if docking_scores is not None and (
        docked_mols is None or len(docking_scores) != len(docked_mols)
    ):
        raise ValueError("docking_scores must align with the supplied docked_mols.")
    if docked_mols is not None and docking_type.lower() not in {
        "vina",
        "vina_pdbqt",
        "vina_sdf",
        "smina",
        "gnina",
    }:
        raise ValueError(
            "Pre-parsed poses are supported only for Vina, Smina and GNINA."
        )

    results = defaultdict(list)
    original_ligand: Chem.rdchem.Mol = get_original_ligand(original_sdf_path)

    # Handle VINA PDBQT by converting to SDF first
    if docking_type.lower() == "vina_pdbqt" and docked_mols is None:
        docked_sdf_path = docked_path.split(".pdbqt")[0] + ".sdf"
        subprocess.run(["mk_export.py", docked_path, "-s", docked_sdf_path])
        return get_docking_data(
            docked_sdf_path, original_sdf_path, metal_coords, "vina"
        )

    # Get docked molecules based on docking type
    if docking_type.lower() == "gnina":
        if docked_mols is None:
            docked_mols = process_gnina_poses(docked_path)
        mols_with_data: list[tuple[Chem.rdchem.Mol, dict]] = [
            (
                mol,
                {
                    "minimizedAffinity": float(mol.GetProp("minimizedAffinity")),
                    "CNNaffinity": float(mol.GetProp("CNNaffinity")),
                    "CNNscore": float(mol.GetProp("CNNscore")),
                },
            )
            for mol in docked_mols
        ]
    elif docking_type.lower() in ["vina", "vina_sdf", "vina_pdbqt"]:
        if docked_mols is None:
            docked_mols = process_vina_poses(docked_path)
        mols_with_data: list[tuple[Chem.rdchem.Mol, dict]] = []
        for mol in docked_mols:
            meeko_data = json.loads(mol.GetProp("meeko"))
            mols_with_data.append(
                (
                    mol,
                    {
                        "docking_score": meeko_data["free_energy"],
                        "intermolecular_energy": meeko_data["intermolecular_energy"],
                        "internal_energy": meeko_data["internal_energy"],
                    },
                )
            )
    elif docking_type.lower() == "smina":
        if docked_mols is None:
            docked_mols, affinities = process_smina_poses(docked_path)
        elif docking_scores is None:
            raise ValueError("docking_scores are required for pre-parsed Smina poses.")
        else:
            affinities = docking_scores
        mols_with_data: list[tuple[Chem.rdchem.Mol, dict]] = [
            (mol, {"docking_score": aff})
            for mol, aff in zip(docked_mols, affinities, strict=True)
        ]
    elif docking_type.lower() == "autodock":
        docked_mols: list[Chem.rdchem.Mol]
        energies: list[float]
        docked_mols, energies = process_autodock_poses(docked_path)
        mols_with_data = [
            (mol, {"docking_score": en})
            for mol, en in zip(docked_mols, energies, strict=False)
        ]
    elif docking_type.lower() == "gpdock":
        docked_mols: list[Chem.rdchem.Mol]
        energies: list[int]
        docked_mols, energies = process_gpdock_poses(docked_path, original_sdf_path)
        mols_with_data = [
            (mol, {"docking_score": en})
            for mol, en in zip(docked_mols, energies, strict=False)
        ]
    elif docking_type.lower() == "metallodock":
        docked_mols: list[Chem.rdchem.Mol]
        scores: list[float | None]
        docked_mols, scores = process_multi_metallodock_poses(docked_path)

        # sort by descending score, i.e., best score first
        sorted_indices: list[int] = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        docked_mols = [docked_mols[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        mols_with_data = [
            (mol, {"mdn_score": s}) for mol, s in zip(docked_mols, scores, strict=False)
        ]
    else:
        raise ValueError(
            f"Invalid docking_type: {docking_type}. Must be one of: "
            "'vina', 'vina_pdbqt', 'vina_sdf', 'smina', 'gnina', 'autodock', 'gpdock', 'metallodock'."
        )

    # Process each molecule
    def r(x):
        return None if x is None else round(x, 5)

    for pose_idx, (docked_mol, data) in enumerate(mols_with_data, 1):
        # Calculate RMSD values
        symm_rmsd: float
        best_isomorphism: tuple[list[int], list[int]]
        symm_rmsd, best_isomorphism = calculate_symmetry_corrected_rmsd(
            original_sdf_path, docked_mol
        )

        # Calculate the metal-binding-group RMSD used as the CHELATE target.
        close_rmsd_35_hetero: float | None
        try:
            close_rmsd_35_hetero = get_rmsd_within_metal_radius(
                original_ligand,
                docked_mol,
                best_isomorphism,
                metal_coords,
                dist_threshold=3.5,
            )
        except Exception:
            close_rmsd_35_hetero = None

        # Print info for first few poses
        if pose_idx < 3:
            print("-" * 50)
            print(f"Pose {pose_idx}:")
            print(
                f"RMSD = {symm_rmsd:.2f} Å"
                if symm_rmsd is not None
                else "RMSD = n/a (no isomorphism)"
            )
            for k, v in data.items():
                print(f"{k}: {v}")
            print("-" * 50)

        # Create result dictionary
        res_vals = {
            "pose": pose_idx,
            "rmsd": r(symm_rmsd),
            "close_rmsd_3.5_hetero": r(close_rmsd_35_hetero),
        }

        # Add program-specific data
        for k, v in data.items():
            res_vals[k] = r(v)

        # Add to results
        for key, value in res_vals.items():
            results[key].append(value)

    return pd.DataFrame(results)


def merge_pdbqt_files_with_fixed_indices(
    pdbqt_file1: str, pdbqt_file2: str, output_pdbqt: str
) -> bool:
    """
    Merge two PDBQT files, adjusting atom indices in the second file to avoid overlap.
    Atom indices from the second structure are adjusted to follow the last index of the first structure.

    Parameters
    ----------
    pdbqt_file1 : str
        Path to the first PDBQT file (indices unchanged).
    pdbqt_file2 : str
        Path to the second PDBQT file (indices adjusted).
    output_pdbqt : str
        Path to save the merged PDBQT file.

    Returns
    -------
    bool
        True if merge succeeds, False otherwise.
    """
    try:
        with open(pdbqt_file1) as file1:
            content1 = file1.readlines()

        with open(pdbqt_file2) as file2:
            content2 = file2.readlines()

        # Find the last serial number in the first file
        last_index = 0
        for line in reversed(content1):
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if parts[1].isdigit():
                    last_index = int(parts[1])
                    break

        # Update the atom indices in the second file
        updated_content2 = []
        for line in content2:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if parts[1].isdigit():
                    new_index = last_index + 1
                    updated_line = (
                        line[:6] + f"{new_index:5d}" + line[11:]
                    )  # Replace index
                    updated_content2.append(updated_line)
                    last_index += 1
                else:
                    updated_content2.append(line)
            else:
                updated_content2.append(line)

        # Merge contents without an extra empty line
        merged_content = content1 + updated_content2

        # Write the merged PDBQT file
        with open(output_pdbqt, "w") as outfile:
            outfile.writelines(merged_content)

        print(f"INFO: Successfully merged PDBQT files into {output_pdbqt}")
        return True

    except Exception as e:
        logger.error(f"Error merging PDBQT files: {e}")
        return False


def convert_cif_to_pdb_pymol(cif_file: str, output_file: str) -> bool:
    """
    Convert a CIF file to a PDB file using PyMOL.

    Parameters
    ----------
    cif_file : str
        Path to the input CIF file.
    output_file : str
        Path to the output PDB file.

    Returns
    -------
    bool
        True if the conversion was successful, False otherwise.
    """
    try:
        with _PYMOL_LOCK:
            cmd.load(cif_file, "myprotein")
            cmd.save(output_file, selection="myprotein")
            cmd.delete("all")
        print(f"CIF file '{cif_file}' converted to PDB file '{output_file}'")
        return os.path.exists(output_file)
    except Exception as e:
        logger.error(f"An error occurred during CIF to PDB conversion: {e}")
        return False


def convert_cif_to_pdb_biotite(cif_file: str, output_file: str) -> bool:
    """
    Convert a CIF file to a PDB file using the biotite library.

    Parameters
    ----------
    cif_file : str
        Path to the input CIF file.
    output_file : str
        Path to the output PDB file.

    Returns
    -------
    bool
        True if the conversion was successful, False otherwise.
    """
    try:
        cif = CIFFile.read(cif_file)
        # Some CIF files lack pdbx_formal_charge, so we try with and without it
        try:
            structure: biotite_array = get_structure(
                cif,
                model=1,
                use_author_fields=False,
                extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
            )
        except KeyError:
            structure: biotite_array = get_structure(
                cif,
                model=1,
                use_author_fields=False,
                extra_fields=["atom_id", "b_factor", "occupancy"],
            )

        if hasattr(structure, "element"):
            structure.element = np.array([e.capitalize() for e in structure.element])

        pdb_file = PDBFile()
        pdb_file.set_structure(structure)
        pdb_file.write(output_file)
        return True
    except Exception as e:
        logger.error(f"An error occurred during CIF to PDB conversion: {e}")
        return False


def convert_pdbqt_to_pdb_pybel(pdbqt_path: str, pdb_path: str) -> bool:
    """
    Convert a PDBQT file to PDB format using OpenBabel/pybel.
    Only the first entry in the PDBQT file is converted.

    Parameters
    ----------
    pdbqt_path : str
        Path to the input PDBQT file.
    pdb_path : str
        Path to the output PDB file.

    Returns
    -------
    bool
        True if the conversion was successful, False otherwise.
    """
    try:
        mol = next(pybel.readfile("pdbqt", pdbqt_path))
        mol.write("pdb", pdb_path, overwrite=True)
        return True
    except StopIteration:
        logger.error(f"Error: Empty PDBQT file or pybel failed to read {pdbqt_path}")
        return False
    except Exception as e:
        logger.error(f"Error converting PDBQT to PDB: {e}")
        return False


def get_box(
    ligand_sdf_path: str, extending: float = 4.0, software: str = "vina"
) -> tuple:
    """
    Calculates the bounding box around a ligand in a given SDF file and formats it for docking software.
    The bounding box is extended by a specified amount in all directions.

    Parameters
    ----------
    ligand_sdf_path : str
        Path to the ligand's SDF file.
    extending : float, default=4.0
        Amount by which to extend the bounding box in all directions.
    software : str, default="vina"
        The docking software format. Can be "vina", "ledock", or "both".

    Returns
    -------
    tuple
        - If `software == "vina"`, returns a tuple with two dictionaries:
            - The center coordinates of the bounding box (center_x, center_y, center_z).
            - The size of the bounding box (size_x, size_y, size_z).
        - If `software == "ledock"`, returns a tuple with three dictionaries:
            - The minimum and maximum x coordinates (minX, maxX).
            - The minimum and maximum y coordinates (minY, maxY).
            - The minimum and maximum z coordinates (minZ, maxZ).
        - If `software == "both"`, returns a tuple containing:
            - Two dictionaries for the Vina format (center and size).
            - Three dictionaries for the LeDock format (min/max for x, y, and z).

    Raises
    ------
    ValueError
        If the `software` parameter is not one of "vina", "ledock", or "both".

    Example Usage
    -------------
    box_vina, size_vina = getbox("ligand.sdf", extending=8.0, software="vina")
    box_ledock, size_ledock = getbox("ligand.sdf", software="ledock")
    """
    if software not in ["vina", "ledock", "both"]:
        raise ValueError("Software must be 'vina', 'ledock' or 'both'")

    with _PYMOL_LOCK:
        cmd.load(filename=ligand_sdf_path, format="sdf", object="lig")
        # get the min and max x, y, and z coordinates of the selected atoms
        extent: tuple[list[float], list[float]] = cmd.get_extent("lig")
        # This clears the pymol data, refreshing the env
        cmd.delete("all")
    [minX, minY, minZ], [maxX, maxY, maxZ] = extent

    # Extending the box by the specified amount
    minX: float = minX - extending
    minY: float = minY - extending
    minZ: float = minZ - extending
    maxX: float = maxX + extending
    maxY: float = maxY + extending
    maxZ: float = maxZ + extending

    # Calculating size & centre
    SizeX: float = maxX - minX
    SizeY: float = maxY - minY
    SizeZ: float = maxZ - minZ
    CenterX: float = (maxX + minX) / 2
    CenterY: float = (maxY + minY) / 2
    CenterZ: float = (maxZ + minZ) / 2

    # Saves data into tuple of dictionaries, formatted to fit the software used.
    # Not a dict lookup: each branch returns a differently shaped tuple built from
    # locals, so the if/elif chain is clearer than a mapping.
    if software == "vina":  # noqa: SIM116
        return {"center_x": CenterX, "center_y": CenterY, "center_z": CenterZ}, {
            "size_x": SizeX,
            "size_y": SizeY,
            "size_z": SizeZ,
        }
    elif software == "ledock":
        return (
            {"minX": minX, "maxX": maxX},
            {"minY": minY, "maxY": maxY},
            {"minZ": minZ, "maxZ": maxZ},
        )
    elif software == "both":
        return (
            {"center_x": CenterX, "center_y": CenterY, "center_z": CenterZ},
            {"size_x": SizeX, "size_y": SizeY, "size_z": SizeZ},
        ), (
            {"minX": minX, "maxX": maxX},
            {"minY": minY, "maxY": maxY},
            {"minZ": minZ, "maxZ": maxZ},
        )


def add_hydrogen_reduce(receptor_path: str, output_path: str) -> tuple[str, dict]:
    """
    Adds hydrogen atoms to a receptor PDB or CIF file using the Reduce tool. It generates new PDB file with
    the added hydrogen atoms and saves it in the same directory as the input file.
    The output file is named by appending "FH" to the original filename, and the suffix of the input file is preserved.

    Parameters
    ----------
    receptor_path : str
        Path to the input receptor PDB/CIF file.
    output_path : str
        Path to save the output PDB/CIF file.

    Returns
    -------
    tuple [str, dict]
        A tuple containing the path to the output PDB/CIF file with added hydrogen atoms and a dictionary of errors.

    Requirements
    ------------
    - The `reduce` tool must be installed and accessible from via the path `reduce2` (as defined in `docking_params.py`).
    """
    print("INFO: Attempting to add hydrogen atoms using Reduce")
    suffix: str = os.path.splitext(receptor_path)[1][1:]
    reduce_path: str = os.path.splitext(receptor_path)[0] + f"FH.{suffix}"
    filename: str = os.path.basename(reduce_path)
    reduce_txt_path: str = os.path.splitext(receptor_path)[0] + "FH.txt"
    filename_txt: str = os.path.splitext(filename)[0] + ".txt"

    cwd: str = os.getcwd()
    result: subprocess.CompletedProcess = subprocess.run(
        ["python", reduce2, receptor_path, "approach=add", "add_flip_movers=True"],
        capture_output=True,
        text=True,
    )

    errors: dict = {}
    if result.stdout:
        pattern: str = r"^\s*([A-Z0-9]{1,4}\s+[A-Z]{3}\s+[A-Z]\s+\d+)\s*$"
        matches: list[str] = re.findall(pattern, result.stdout, re.MULTILINE)
        errors["reduce_residues_failed_parameterization"] = matches

    # Move the output files to the specified paths
    if os.path.exists(f"{cwd}/{filename}"):
        subprocess.run(["mv", f"{cwd}/{filename}", output_path])
    if os.path.exists(f"{cwd}/{filename_txt}"):
        subprocess.run(["mv", f"{cwd}/{filename_txt}", reduce_txt_path])

    return os.path.exists(output_path), errors


def trim_cif_file(path: str) -> bool:
    """
    Cleans a CIF file by removing empty lines and unnecessary indentations, overwriting the original file.

    This is necessary since ProDy (used by Meeko) is otherwise unable to read the CIF file produced by Reduce2.
    Might be resolved in the future: https://github.com/prody/ProDy/discussions/2094

    Parameters
    ----------
    path : str
        Path to the input CIF file.

    Returns
    -------
    bool
        True if the CIF file was successfully cleaned, False otherwise.
    """
    try:
        with open(path) as infile:
            lines: list[str] = infile.readlines()

        # Remove empty lines and trim spaces
        cleaned_lines: list[str] = [
            line.strip() for line in lines if line.strip()
        ]  # List without empty lines

        # Ensure last line is a hashtag (#)
        if cleaned_lines and cleaned_lines[-1] != "#":
            cleaned_lines.append("#")

        # Write cleaned content back to the file
        with open(path, "w") as outfile:
            outfile.write("\n".join(cleaned_lines) + "\n")  # Preserve formatting

        print(f"INFO: Cleaned and overwritten CIF file: {path}")
        return True

    except Exception as e:
        logger.error(f"Error processing CIF file: {e}")
        return False


def prepare_receptor_meeko(
    receptor_path: str, output_path: str, file_type: str | None = None
) -> tuple[str, dict]:
    """
    Prepares the receptor structure for docking using Meeko's `mk_prepare_receptor.py` script.
    The cleaned receptor is saved in PDBQT format in the same directory as the input file.

    Parameters
    ----------
    receptor_path : str
        Path to the receptor file (PDB or CIF) that needs to be prepared.
    file_type : str or None, default = None
        File type of the receptor, either 'pdb' or 'cif'. If None, it will be inferred from the file extension.

    Returns
    -------
    tuple[str, dict]
        A tuple containing the path to the output receptor in PDBQT format and all template matching errors.
    """
    print("INFO: Preparing Receptor")
    errors: dict = {}
    # Determine file type if not specified
    if file_type is None:
        if receptor_path.lower().endswith(".pdb"):
            file_type = "pdb"
        elif receptor_path.lower().endswith(".cif"):
            file_type = "cif"
        else:
            raise ValueError(
                "File type not specified and cannot be inferred from file extension. "
                "Please specify 'pdb' or 'cif' as file_type."
            )

    # Assert that file_type is valid
    assert file_type in ["pdb", "cif"], "File type must be either 'pdb' or 'cif'"

    # Extract base path without extension
    pdbqt_path = os.path.splitext(output_path)[0]
    receptor_pdbqt_path = pdbqt_path + ".pdbqt"

    # Set up the command based on file type
    command = [mk_prepare_receptor]

    if file_type == "pdb":
        command.extend(["--read_pdb", receptor_path])
    else:  # cif
        command.extend(["-i", receptor_path])

    command.extend(["-o", pdbqt_path, "-p", "--allow_bad_res"])

    # Run the command
    result: subprocess.CompletedProcess = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    # Process any errors
    if result.stderr:
        lines: list = result.stderr.split("\n")
        for line in lines:
            if "Template matching failed for:" in line:
                pattern: str = r"\['(.*?)'\]"
                matches: list = re.findall(pattern, line)
                list_contents: list = matches[0].split("', '") if matches else []
                errors["meeko_bad_residues"] = list_contents

    return os.path.exists(receptor_pdbqt_path), errors


def prepare_ligand_meeko(
    ligand_sdf_path: str,
    ligand_pdbqt_path: str,
    skip_tautomer: bool = True,
    skip_acidbase: bool = False,
    pH: float = 7.4,
) -> bool:
    """
    Prepares a ligand structure using the Meeko pipeline.

    This function performs two steps to prepare a ligand for docking:
    1. It scrubs the ligand SDF file (using pH 7.4). Optionally, it generates tautomers and adjust acid-base protonation states.
    2. It converts the scrubbed ligand into the PDBQT format required for docking.

    The input SDF file is scrubbed and saved with "_scrubbed" appended to the filename.
    The output PDBQT file (output from `mk_prepare_ligand`) is saved in the specified path
    `ligand_pdbqt_path`.

    Parameters
    ----------
    ligand_sdf_path : str
        Path to the input ligand SDF file.
    ligand_pdbqt_path : str
        Path to save the output ligand PDBQT file.
    skip_tautomer : bool, default=True
        If True, skips tautomer generation during the scrubbing process.
    skip_acidbase : bool, default=False
        If True, skips acid-base protonation state adjustments during the scrubbing process.
    pH : float, default=7.4
        The pH value to use for the scrubbing process.

    Returns
    -------
    bool
        True if the ligand was successfully prepared, False otherwise.
    """
    ligand_name: str = os.path.basename(ligand_sdf_path)
    print(f"Preparing ligand <{ligand_name}> with Meeko")
    scrubbed_path: str = os.path.splitext(ligand_sdf_path)[0] + "_scrubbed.sdf"

    args: list[str] = []
    if skip_tautomer:
        args.append("--skip_tautomer")
    if skip_acidbase:
        args.append("--skip_acidbase")

    subprocess.run(
        [scrub, ligand_sdf_path, "-o", scrubbed_path, "--ph", str(pH), *args]
    )
    subprocess.run(
        preparation_command(
            mk_prepare_ligand, "-i", scrubbed_path, "-o", ligand_pdbqt_path
        )
    )
    return os.path.exists(ligand_pdbqt_path)


def calculate_symmetry_corrected_rmsd(
    ligand_sdf_path: str, mol: Chem.rdchem.Mol
) -> tuple[float | None, tuple[list[int], list[int]] | None]:
    """
    Calculate the symmetry-corrected RMSD between two molecules using the SpyRMSD library.

    Parameters
    ----------
    ligand_sdf_path : str
        Path to the reference ligand SDF file.

    mol : Chem.rdchem.Mol
        The molecule for which the RMSD is to be calculated.

    Returns
    -------
    tuple[float | None, tuple[list[int], list[int]] | None]
        A tuple containing:
        - The symmetry-corrected RMSD value between the two molecules (or None if the calculation fails).
        - The according isomorphism mapping as a tuple of two lists (or None if the calculation fails).
    """
    # Load the reference through the same RDKit path used for MBG labels so that
    # every explicit hydrogen is removed.  SpyRMSD's direct SDF loader retains
    # explicit hydrogens, while docked poses are hydrogen-free; that difference
    # makes otherwise identical heavy-atom graphs non-isomorphic.
    ref: SpyMolecule = SpyMolecule.from_rdkit(get_original_ligand(ligand_sdf_path))
    coords_ref: np.ndarray = ref.coordinates
    anum_ref: np.ndarray = ref.atomicnums
    adj_ref: np.ndarray = ref.adjacency_matrix

    mol: SpyMolecule = SpyMolecule.from_rdkit(mol)
    coords: list[np.ndarray] = mol.coordinates
    anum: np.ndarray = mol.atomicnums
    adj: np.ndarray = mol.adjacency_matrix

    try:
        rmsd: float
        best_isomorphism: tuple[list[int], list[int]]
        rmsd, best_isomorphism = symmrmsd(
            coordsref=coords_ref,
            coords=coords,
            apropsref=anum_ref,
            aprops=anum,
            amref=adj_ref,
            am=adj,
            return_best_isomorphism=True,
        )
        return rmsd, best_isomorphism
    except Exception as e:
        print("SymmRMSD Calculation failed with error:", e)
        return None, None


def get_biotite_structure_with_metals(
    pdb_id: str, output_pdb_path: str, ligand_ccd: str, output_ligand_sdf_path: str
) -> None:
    """
    Fetches a PDB file from the RCSB database and extracts the structure using Biotite.
    The function also downloads the ligand SDF file from the RCSB database.
    The output PDB structure contains all main chain atoms and metal atoms, while the ligand is saved as an SDF file.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the structure.
    output_pdb_path : str
        Path to save the output PDB file.
    ligand_ccd : str
        The CCD code of the ligand.
    output_ligand_sdf_path : str
        Path to save the ligand SDF file.

    Returns
    -------
    None
    """
    structure: biotite_array = get_structure(
        CIFFile.read(
            fetch(
                pdb_ids=pdb_id,
                format="cif",
                overwrite=False,
            )
        ),
        model=1,
        use_author_fields=False,
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
    )

    filtered_biotite_atoms_array: list = []
    ligand_chain: str = ""

    for atom in structure:
        if not atom.hetero or (atom.hetero and atom.element in METAL_ATOMS_SET):
            filtered_biotite_atoms_array.append(atom)
        elif atom.res_name == ligand_ccd:
            ligand_chain: str = atom.chain_id

    rcsb: str = f"https://models.rcsb.org/v1/{pdb_id}/ligand?label_asym_id={ligand_chain}&encoding=sdf&filename={ligand_ccd}.sdf"
    folder_path: str = os.path.dirname(output_ligand_sdf_path)

    # Download the ligand SDF file
    subprocess.run(["curl", "-o", f"{ligand_ccd}_temp.sdf", rcsb], cwd=folder_path)

    input_sdf_path: str = f"{folder_path}/{ligand_ccd}_temp.sdf"
    suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(input_sdf_path)
    writer: Chem.rdmolfiles.SDWriter = Chem.rdmolfiles.SDWriter(output_ligand_sdf_path)

    for mol in suppl:
        if mol is not None:  # Avoid issues with invalid molecules
            writer.write(mol)
        else:
            print(f"Warning: Encountered an invalid molecule in {input_sdf_path}")
    writer.close()

    # Write the structure to a PDB file
    filtered_biotite_atoms_array: biotite_array = biotite_array(
        filtered_biotite_atoms_array
    )
    pdb_file: PDBFile = PDBFile()
    pdb_file.set_structure(filtered_biotite_atoms_array)
    pdb_file.write(output_pdb_path)


def add_dummy_atoms(
    pdb_path: str, output_path: str, atom_coords: list[tuple[float, float, float]]
) -> bool:
    """
    Adds dummy atoms to a PDB file and writes the modified structure to a new file.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    output_path : str
        Path to the output PDB file.
    atom_coords : list of tuple[float, float, float]
        List of tuples containing coordinates (x, y, z) for dummy atoms.

    Returns
    -------
    bool
        True if the dummy atoms were successfully added and the file was saved, False otherwise.
    """
    try:
        # Read the content of the original PDB file
        with open(pdb_path) as pdb_file:
            pdb_lines: list[str] = pdb_file.readlines()

        # Prepare to append dummy atoms
        dummy_atom_lines: list[str] = []
        atom_serial: int = 1  # Starting serial number for dummy atoms

        for atom_coord in atom_coords:
            x, y, z = atom_coord
            element: str = "U"  # Dummy element
            pdb_line: str = (
                f"HETATM{atom_serial:>5} {element:<2}    {element:<3}   1    "  # Atom serial, name, residue
                f"{x:8.3f}{y:8.3f}{z:8.3f}"  # Coordinates
                f"{1.00:6.2f}{0.00:6.2f}          {element:>2}"  # Occupancy, B-factor, element
            )
            dummy_atom_lines.append(pdb_line + "\n")
            atom_serial += 1

        # Write to the output PDB file
        with open(output_path, "w") as output_file:
            # Write original PDB content
            output_file.writelines(pdb_lines)

            # Write dummy atoms at the end
            output_file.writelines(dummy_atom_lines)

        print(f"Dummy atoms successfully added and saved to {output_path}.")
        return True

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False


def get_atom_ligand_min_distance_and_LCN(
    main_ligand_rdkit: Chem.rdchem.Mol,
    atom_coords: tuple[float, float, float],
    elements: set | None = None,
) -> tuple[float, int, str]:
    """
    Calculate the minimum distance between any atom of the main ligand and a given atom,
    along with the ligand coordination number (LCN) and element of the closest atom.

    Parameters
    ----------
    main_ligand_rdkit : Chem.rdchem.Mol
        The main ligand molecule.
    atom_coords : tuple[float, float, float]
        Coordinates of the atom to calculate the distance to.
    elements : set | None, default=None
        A set of element symbols to filter the ligand atoms. If None, all atoms are considered.

    Returns
    -------
    tuple [float, int, str]
        A tuple containing:
        - min_distance (float): Minimum distance between the atom and any atom in the ligand.
        - min_distance_LCN (int): Ligand coordination number of the closest atom.
        - min_distance_elem (str): Element symbol of the closest atom.
    """
    min_distance: float = 1000.0
    min_distance_LCN: int = 0
    min_distance_elem: str = ""
    for atom in main_ligand_rdkit.GetAtoms():
        conformer: Chem.Conformer = main_ligand_rdkit.GetConformer()
        atom_idx: int = atom.GetIdx()
        atom_elem: str = atom.GetSymbol()
        if elements and atom_elem not in elements:
            continue

        pos: Chem.rdGeometry.Point3D = conformer.GetAtomPosition(atom_idx)
        x: float = pos.x
        y: float = pos.y
        z: float = pos.z

        distance: float = np.linalg.norm(np.array([x, y, z]) - np.array(atom_coords))
        LCN: int = atom.GetDegree()  # number of directly-bonded neighbors
        LCN += 1  # Assuming the metal is always bound to the closest atom
        elem = atom.GetSymbol()

        if distance < min_distance:
            min_distance_LCN = LCN
            min_distance = distance
            min_distance_elem = elem

    return min_distance, min_distance_LCN, min_distance_elem


def get_hb_atoms(mol: Chem.rdchem.Mol) -> tuple[list[int], list[int]]:
    """
    Identify hydrogen bond donors and acceptors in an RDKit molecule using SMARTS patterns.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit molecule object.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - List of H-bond acceptor atom indices.
        - List of H-bond donor atom indices.
    """
    # SMARTS patterns for H-bond acceptors and donors
    # https://github.com/rdkit/rdkit/blob/eebad845f551b6b15f0655589a5ab67e52f8f53f/Code/GraphMol/Descriptors/Lipinski.cpp#L197
    HBA: str = "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
    HBD: str = "[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"

    # Create SMARTS pattern matchers
    hba_pattern: Chem.rdchem.Mol = Chem.MolFromSmarts(HBA)
    hbd_pattern: Chem.rdchem.Mol = Chem.MolFromSmarts(HBD)

    # Find all matches
    hba_matches: tuple = mol.GetSubstructMatches(hba_pattern)
    hbd_matches: tuple = mol.GetSubstructMatches(hbd_pattern)

    # Extract atom indices
    hba_atoms: list[int] = [match[0] for match in hba_matches]  # Each match is a tuple
    hbd_atoms: list[int] = [match[0] for match in hbd_matches]

    return hba_atoms, hbd_atoms


def get_rmsd_within_metal_radius(
    original_mol: Chem.rdchem.Mol,
    docked_mol: Chem.rdchem.Mol,
    best_isomorphism: tuple[list[int], list[int]],
    metal_coords: list[tuple[float, float, float]],
    dist_threshold: float = 3.5,
) -> float | None:
    """
    Calculate metal-binding-group RMSD between original and docked molecules.

    The group contains ligand N, O and S atoms within the specified radius of any
    metal in the reference structure. Uses the symmetry-corrected atom mapping.

    Parameters
    ----------
    original_mol : rdkit.Chem.Mol
        The original molecule with reference coordinates.
    docked_mol : rdkit.Chem.Mol
        The docked molecule.
    best_isomorphism : tuple[list[int], list[int]]
        The best isomorphism mapping between the original and docked molecule.
    metal_coords : list of tuple[float, float, float]
        List of metal coordinates in the reference structure's coordinate frame.
    dist_threshold : float, default=3.5
        Distance threshold in Angstroms to define the metal coordination sphere.

    Returns
    -------
    float or None
        RMSD for the metal-binding-group atoms, or None if no matching atoms are found.

    Raises
    ------
    ValueError
        If mapped atom coordinates cannot be read or the distance threshold is not
        positive.
    """
    if dist_threshold <= 0:
        raise ValueError("Distance threshold must be positive")

    docked_conformer: Chem.Conformer = docked_mol.GetConformer()
    orig_conformer: Chem.Conformer = original_mol.GetConformer()

    close_sum: float = 0
    close_cnt: int = 0

    HETERO_SYMBOLS = {"N", "O", "S"}

    metal_idcs: set[int] = set()
    for i, atom in enumerate(original_mol.GetAtoms()):
        if atom.GetSymbol() not in HETERO_SYMBOLS:
            continue
        for m_coords in metal_coords:
            if (
                np.linalg.norm(np.array(orig_conformer.GetAtomPosition(i)) - m_coords)
                < dist_threshold
            ):
                metal_idcs.add(i)
                break

    for ref_idx, docked_idx in zip(*best_isomorphism, strict=False):
        if ref_idx not in metal_idcs:
            continue
        try:
            orig_coords: Chem.rdGeometry.Point3D = orig_conformer.GetAtomPosition(
                ref_idx
            )
            docked_coords: Chem.rdGeometry.Point3D = docked_conformer.GetAtomPosition(
                docked_idx
            )
            dist_sqr: float = (orig_coords - docked_coords).LengthSq()

            close_sum += dist_sqr
            close_cnt += 1

        except Exception as e:
            raise ValueError(
                f"Error processing atoms {ref_idx}, {docked_idx}: {e!s}"
            ) from e

    if close_cnt == 0:
        print("No atoms found within metal coordination sphere")
        return None

    return np.sqrt(close_sum / close_cnt)


def generate_autodock_grid_cached(
    receptor_pdbqt: str,
    ligand_sdf: str,
    output_dir: str,
    system_name: str,
    box_extension: float = 4.0,
    spacing: float = 0.375,
    max_points: int = 255,
    timeout: int = 3600,
) -> str | None:
    """
    Generate AutoDock grid files with caching. Reuses existing grids if complete.

    Parameters
    ----------
    receptor_pdbqt : str
        Path to receptor PDBQT file
    ligand_sdf : str
        Path to reference ligand SDF file (for box calculation)
    output_dir : str
        Directory for output grid files
    system_name : str
        Base name for grid files
    box_extension : float, default=4.0
        Box extension around ligand in Angstroms
    spacing : float, default=0.375
        Grid spacing in Angstroms
    max_points : int, default=255
        Maximum number of grid points per dimension (AutoDock-GPU constraint)
    timeout : int, default=3600
        Maximum time in seconds to allow for grid generation before aborting

    Returns
    -------
    str | None
        Path to .maps.fld file if successful, None otherwise
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if grid already exists and is complete
    glg_file = os.path.join(output_dir, f"{system_name}_grid.glg")
    if os.path.exists(glg_file):
        with open(glg_file) as f:
            if "Successful Completion" in f.read():
                fld_files = glob(f"{system_name}_grid*.maps.fld", root_dir=output_dir)
                if fld_files:
                    return os.path.join(output_dir, fld_files[0])

    # Calculate box from ligand
    try:
        center_dict: dict = {}
        size_dict: dict = {}
        center_dict, size_dict = get_box(
            ligand_sdf, extending=box_extension, software="vina"
        )
        grid_center = np.array(
            [center_dict["center_x"], center_dict["center_y"], center_dict["center_z"]]
        )
        grid_size = np.array(
            [size_dict["size_x"], size_dict["size_y"], size_dict["size_z"]]
        )
    except Exception as e:
        print(f"Error calculating box for {system_name}: {e}")
        return None

    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if max_points <= 0:
        raise ValueError(f"max_points must be a positive integer, got {max_points}")

    # AutoDock-GPU constraint: maximum points per dimension (default 255)
    required_spacing = np.maximum(grid_size / max_points, spacing)
    final_spacing = float(np.max(required_spacing))

    # Generate GPF file
    gpf_file = os.path.join(output_dir, f"{system_name}_grid.gpf")
    try:
        gpf_string: str
        gpf_string, _ = get_gpf_string(
            grid_center.tolist(),
            grid_size.tolist(),
            receptor_pdbqt,
            AUTODOCK_RECEPTOR_TYPES,
            AUTODOCK_LIGAND_TYPES,
            map_prefix=f"{system_name}_grid_agrid",
            spacing=final_spacing,
        )
        with open(gpf_file, "w") as f:
            f.write(gpf_string)
    except Exception as e:
        print(f"Error creating GPF for {system_name}: {e}")
        return None

    # Run autogrid4
    try:
        cmd = ["autogrid4", "-p", gpf_file, "-l", glg_file]
        result = subprocess.run(
            cmd, cwd=output_dir, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"autogrid4 failed for {system_name}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running autogrid4 for {system_name}: {e}")
        return None

    # Check for output
    fld_files = glob(f"{system_name}_grid*.maps.fld", root_dir=output_dir)
    if fld_files:
        return os.path.join(output_dir, fld_files[0])

    return None


def _preprocess_single(info: dict) -> tuple[str, str | None]:
    """
    Worker routine used by :func:`preprocess_autodock_systems_parallel`.
    """
    grid_fld = generate_autodock_grid_cached(
        receptor_pdbqt=info["receptor_pdbqt"],
        ligand_sdf=info["ligand_sdf"],
        output_dir=info["output_dir"],
        system_name=info["system_id"],
        box_extension=info.get("box_extension", 4.0),
        spacing=info.get("spacing", 0.375),
        max_points=info.get("max_points", 255),
        timeout=info.get("timeout", 3600),
    )
    return (info["system_id"], grid_fld)


def preprocess_autodock_systems_parallel(
    systems_info: list[dict],
    num_workers: int = 16,
    box_extension: float = 4.0,
    spacing: float = 0.375,
    max_points: int = 255,
    timeout: int = 3600,
) -> dict[str, str]:
    """
    Generate AutoDock grids for multiple systems in parallel.

    Parameters
    ----------
    systems_info : list[dict]
        List of dicts with keys: system_id, receptor_pdbqt, ligand_sdf, output_dir
    num_workers : int, default=16
        Number of parallel workers
    box_extension : float, default=4.0
        Box extension around ligand
    spacing : float, default=0.375
        Grid spacing in Angstroms
    max_points : int, default=255
        Maximum number of grid points per dimension
    timeout : int, default=3600
        Maximum time in seconds to allow for grid generation before aborting

    Returns
    -------
    dict[str, str]
        Mapping of system_id to grid .fld file path
    """
    # Add parameters to each system info so the worker can access it.
    for info in systems_info:
        info["box_extension"] = box_extension
        info["spacing"] = spacing
        info["max_points"] = max_points
        info["timeout"] = timeout

    # Process in parallel
    if num_workers == 1:
        results = [_preprocess_single(info) for info in systems_info]
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_preprocess_single, systems_info)

    # Build result dict
    grid_map = {sys_id: grid_path for sys_id, grid_path in results if grid_path}

    return grid_map


def execute_autodock_batch(
    batch_systems: list[dict], nrun: int = 50, timeout: int = 7200
) -> None:
    """
    Execute AutoDock-GPU docking on a batch of systems.

    Parameters
    ----------
    batch_systems : list[dict]
        List of dicts with keys: grid_fld, ligand_pdbqt, output_name, output_dir
    nrun : int, default=50
        Number of AutoDock LGA runs
    timeout : int, default=7200
        Maximum time in seconds to allow for docking before aborting

    Returns
    -------
    None
        Outputs are written to disk
    """
    if not batch_systems:
        return

    # Create batch file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as batch_file:
        for system in batch_systems:
            batch_file.write(f"{system['grid_fld']}\n")
            batch_file.write(f"{system['ligand_pdbqt']}\n")
            batch_file.write(f"{system['output_name']}\n")
        batch_file_path = batch_file.name

    # Use first system's output dir as working directory
    working_dir = batch_systems[0]["output_dir"]

    try:
        # Run AutoDock-GPU
        cmd = [
            "autodock_gpu_64wi",
            "-B",
            batch_file_path,
            "-nrun",
            str(nrun),
            "--dlgoutput",
            "1",
            "--xmloutput",
            "1",
        ]
        result = subprocess.run(
            cmd, cwd=working_dir, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            print(f"AutoDock-GPU batch failed: {result.stderr}")

        # Move output files to respective directories
        for system in batch_systems:
            output_name = system["output_name"]
            for ext in [".dlg", ".xml"]:
                src = os.path.join(working_dir, f"{output_name}{ext}")
                dst = os.path.join(system["output_dir"], f"{output_name}{ext}")
                if os.path.exists(src) and src != dst:
                    with contextlib.suppress(Exception):
                        os.rename(src, dst)

    except Exception as e:
        print(f"Error executing AutoDock batch: {e}")
    finally:
        # Cleanup batch file
        with contextlib.suppress(Exception):
            os.remove(batch_file_path)


def _gpdock_sdf_to_mol2(sdf_path: str, output_mol2_path: str) -> None:
    """
    Convert an SD file to MOL2 using OpenBabel/pybel.

    Parameters
    ----------
    sdf_path : str
        Path to the input SD file.
    output_mol2_path : str
        Path to save the output MOL2 file.

    Returns
    -------
    None
        Outputs are written to disk.

    """
    mol = next(pybel.readfile("sdf", sdf_path))
    mol.write("mol2", output_mol2_path, overwrite=True)


def _fix_gpdock_metal_atom_names(pdb_path: str) -> None:
    """
    Rename metal atom names in a PDB file as required by GPDOCK:
      - Calcium (residue CA): atom name CA/CA2+ -> C0
      - Cadmium (residue CD): atom name CD/CD2+ -> CAD

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file to modify.

    Returns
    -------
    None
        The input PDB file is modified in place.
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    modified_lines: list[str] = []
    for line in lines:
        record = line[:6].strip()
        if record in ("HETATM", "ATOM"):
            res_name = line[17:20].strip().upper()
            atom_name = line[12:16].strip().upper()
            if res_name == "CA" and atom_name in ("CA", "CA2+"):
                line = line[:12] + " C0 " + line[16:]
            elif res_name == "CD" and atom_name in ("CD", "CD2+"):
                line = line[:12] + " CAD" + line[16:]
        modified_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(modified_lines)


def run_gpdock_docking(
    receptor_pdbqt: str,
    ligand_sdf: str,
    ion_code: int,
    num_waters: int,
    output_dir: str,
    multi_metal: bool = False,
    timeout: int = 90 * 60,
) -> str:
    """
    Run GPDOCK docking for a single system.

    Converts the receptor PDBQT -> PDB and ligand SDF -> MOL2, renames
    metal atoms (CA->C0, CD->CAD), and calls the appropriate GPDOCK binary
    (single- or multi-metal).  If the primary run produces an empty out.pdb,
    hydrogens are stripped with gp_heavy and docking is retried.

    Parameters
    ----------
    receptor_pdbqt : str
        Path to the merged receptor PDBQT file.
    ligand_sdf : str
        Path to the ligand SDF file.
    ion_code : int
        GPDOCK ION code (1-10).
    num_waters : int
        Number of water molecules for docking (0-2). Ignored for multi-metal.
    output_dir : str
        Directory where intermediate and output files are written.
    multi_metal : bool, default=False
        If True, uses the gpdock_multi_metal binary and omits -HOH.
    timeout : int, default=90*60
        Maximum time in seconds to allow for docking before aborting.

    Returns
    -------
    str
        Absolute path to the produced out.pdb.

    Raises
    ------
    RuntimeError
        If any conversion step or the GPDOCK binary itself fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- File conversion ---
    receptor_pdb = os.path.join(output_dir, "gpdock_receptor.pdb")
    ligand_mol2 = os.path.join(output_dir, "gpdock_ligand.mol2")

    try:
        convert_pdbqt_to_pdb_pybel(receptor_pdbqt, receptor_pdb)
    except Exception as e:
        raise RuntimeError(f"PDBQT->PDB conversion failed: {e}") from e

    if not os.path.exists(receptor_pdb) or os.path.getsize(receptor_pdb) == 0:
        raise RuntimeError(f"PDBQT->PDB conversion produced empty file: {receptor_pdb}")

    _fix_gpdock_metal_atom_names(receptor_pdb)

    try:
        _gpdock_sdf_to_mol2(ligand_sdf, ligand_mol2)
    except Exception as e:
        raise RuntimeError(f"SDF->MOL2 conversion failed: {e}") from e

    if not os.path.exists(ligand_mol2) or os.path.getsize(ligand_mol2) == 0:
        raise RuntimeError(f"SDF->MOL2 conversion produced empty file: {ligand_mol2}")

    # --- Run GPDOCK ---
    executable = GPDOCK_MULTI_METAL_PATH if multi_metal else GPDOCK_PATH
    out_pdb = os.path.join(output_dir, "out.pdb")

    def _run(lig_mol2: str) -> None:
        cmd = [
            executable,
            "-PRO",
            receptor_pdb,
            "-LIG",
            lig_mol2,
            "-ION",
            str(ion_code),
        ]
        if not multi_metal:
            cmd += ["-HOH", str(num_waters)]
        subprocess.run(
            cmd, cwd=output_dir, capture_output=True, text=True, timeout=timeout
        )

    _run(ligand_mol2)

    # --- Retry with heavy atoms if out.pdb is empty ---
    if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
        gp_heavy = os.path.join(os.path.dirname(executable), "gp_heavy")
        heavy_mol2 = os.path.join(output_dir, "heavy_lig.mol2")
        subprocess.run(
            [gp_heavy, "-LIG", ligand_mol2],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not os.path.exists(heavy_mol2) or os.path.getsize(heavy_mol2) == 0:
            raise RuntimeError("gp_heavy failed to produce heavy_lig.mol2")
        _run(heavy_mol2)

    if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
        raise RuntimeError(
            "GPDOCK produced an empty out.pdb (even after gp_heavy retry)"
        )

    return out_pdb


def extract_metallodock_pocket(
    receptor_pdb: str, ligand_sdf: str, output_pdb: str, ball_radius: float = 4.0
) -> bool:
    """
    Extract a binding pocket around the given ligand by finding all receptor atoms
    within a radius that fully contains a Vina-style box around the ligand.
    Keeps the entire residues.

    Parameters
    ----------
    receptor_pdb : str
        Path to the receptor PDB file.
    ligand_sdf : str
        Path to the ligand SDF file.
    output_pdb : str
        Path to save the extracted pocket PDB file to.
    ball_radius : float, default=4.0
        Radius in Angstroms of the ball used to encapsulate the Vina-style box around the ligand.

    Returns
    -------
    bool
        True if the pocket was extracted successfully, False otherwise.
    """
    try:
        pdb_file: PDBFile = PDBFile.read(receptor_pdb)
        try:
            u_rec: biotite_array = pdb_file.get_structure(
                model=1, extra_fields=["b_factor", "occupancy", "charge"]
            )
        except KeyError:
            u_rec: biotite_array = pdb_file.get_structure(
                model=1, extra_fields=["b_factor", "occupancy"]
            )

        u_lig: Chem.Mol = Chem.MolFromMolFile(ligand_sdf, removeHs=False)
        if u_lig is None:
            return False

        lig_positions: np.ndarray = u_lig.GetConformer().GetPositions()
        lig_center: np.ndarray = np.mean(lig_positions, axis=0, keepdims=True)
        lig_max_extent: float = np.max(
            scipy.spatial.distance.cdist(lig_positions, lig_center)
        )

        # Ball that fully contains the Vina-style box around the ligand
        def radius_encapsulating_vina_style_box(
            max_extent: float, box_extent: float
        ) -> float:
            return np.sqrt(3) * box_extent + max_extent

        # Calculate distance matrix
        dist_matrix: np.ndarray = scipy.spatial.distance.cdist(u_rec.coord, lig_center)
        close_atoms_idx: np.ndarray = np.where(
            dist_matrix[:, 0]
            <= radius_encapsulating_vina_style_box(lig_max_extent, ball_radius)
        )[0]

        if len(close_atoms_idx) == 0:
            return False

        # Get unique residues that have atoms within range
        close_atoms: biotite_array = u_rec[close_atoms_idx]
        unique_res: set[tuple[str, int, str]] = set(
            zip(
                close_atoms.chain_id,
                close_atoms.res_id,
                close_atoms.ins_code,
                strict=False,
            )
        )

        # Mask and extract entire residues
        mask: np.ndarray = np.array(
            [
                (c, r, i) in unique_res
                for c, r, i in zip(
                    u_rec.chain_id, u_rec.res_id, u_rec.ins_code, strict=False
                )
            ]
        )
        pocket: biotite_array = u_rec[mask]

        out_file: PDBFile = PDBFile()
        out_file.set_structure(pocket)
        out_file.write(output_pdb)
        return True

    except Exception as e:
        logger.error(f"Error extracting pocket: {e}")
        return False


def generate_blind_ligand(input_sdf: str, output_sdf: str, seed: int) -> bool:
    """
    Generate a new ligand conformation from the input SDF and save it to the output SDF.

    Parameters
    ----------
    input_sdf : str
        Path to the input SDF file containing the original ligand structure.
    output_sdf : str
        Path to save the output SDF file containing the generated ligand structure.
    seed : int
        Random seed for reproducibility of the conformation generation.

    Returns
    -------
    bool
        True if the generated ligand was saved successfully, False otherwise.
    """
    try:
        mol: Chem.Mol = Chem.MolFromMolFile(input_sdf, removeHs=False)
        if mol is None:
            return False

        mol = Chem.RemoveAllHs(mol)

        res: int = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
            if res != 0:
                logger.error("Warning: RDKit could not embed blind ligand")
                return False

        Chem.MolToMolFile(mol, output_sdf)
        return True
    except Exception as e:
        logger.error(f"Error generating blind ligand: {e}")
        return False
