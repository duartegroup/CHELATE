"""
Description
-----------
This module provides utility functions for data curation.

Functions
---------
- `get_PLINDEX_row(system_id: str) -> pd.DataFrame`:
    Retrieves a row from the PLINDEX dataset based on the given system ID.

- `get_metals_from_ligand_dir(ligands_path: str) -> list[tuple[str, tuple[float, float, float]]]`:
    Extracts metal atoms and their coordinates from ligand files in a specified directory.

- `get_ligand_metal_min_distance(ligands_path: str) -> list`:
    Calculates the minimum distance between ligands and metal ions in a complex.

- `get_atom_ligand_min_distance(ligand: Chem.rdchem.Mol, atom_coords: list) -> float`:
    Calculates the minimum distance between any atom of a ligand and a given atom's coordinates.
"""

import logging
import os

import numpy as np
import pandas as pd
from params import COLS_OF_INTEREST, HETERO_ATOMS_NUM_SET, METAL_ATOMS_SET
from plinder.core.scores import query_index
from rdkit import Chem

# By default a metal is a positively charged single-atom ligand whose element is a
# metal. Ammonium (NH4+) is stored in PLINDER's SDFs as one N atom with charge +1, so
# without the element check it would count as a "metal". Set this to False
# (``generate_metal_ligand_data.py --paper_metal_rule``) to reproduce the curation used
# for the published dataset, which also accepted positively charged single-atom ions such
# as NH4+ (27 training systems, none of which reach the feature CSVs).
REQUIRE_METAL_ELEMENT: bool = True


def get_PLINDEX_row(system_id: str) -> pd.DataFrame:
    """
    Retrieves a row from the PLINDEX dataset based on the given system ID.

    Parameters
    ----------
    system_id : str
        The unique identifier for the system.

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the row from the PLINDEX dataset that matches
        the specified system ID and filters.

    Notes
    -----
    The function filters the dataset to include only rows where:
        - `system_id` matches the provided `system_id`.
        - `system_type` is "holo".
        - `ligand_is_proper` is `True`.
        - `ligand_is_cofactor` is `False`.
    """

    rows: pd.DataFrame = query_index(
        columns=COLS_OF_INTEREST,
        splits=["*"],
        filters=[
            ("system_id", "==", system_id),
            ("system_type", "==", "holo"),
            ("ligand_is_proper", "==", "True"),
            ("ligand_is_cofactor", "==", "False"),
        ],
    )
    return rows


def get_metals_from_ligand_dir(
    ligands_path: str,
) -> list[tuple[str, tuple[float, float, float]]]:
    """
    Extract metal atoms and their coordinates from ligand files in a specified directory.

    This function processes all SD files in the given directory, identifying metal atoms
    (positively charged single-atom molecules whose element is in ``METAL_ATOMS_SET``)
    and recording their element symbols and 3D coordinates. The element check can be
    switched off; see ``REQUIRE_METAL_ELEMENT``.

    Parameters
    ----------
    ligands_path : str
       Path to the directory containing ligand SD files.

    Returns
    -------
    list of tuple:
        A list of tuples containing two elements, the element symbol of the metal atom
        and its (x, y, z) coordinates.

    Example
    -------
    >>> metals = get_metals_from_ligand_dir("/path/to/ligand/files")
    >>> print(metals[0])
    ('Fe', (1.234, 5.678, 9.012))
    """

    # get all paths in folder ligands_path
    ligand_files: list[str] = [
        os.path.join(ligands_path, f)
        for f in os.listdir(ligands_path)
        if f.endswith(".sdf")
    ]

    metals: list[tuple[str, tuple[float, float, float]]] = []

    for ligand_file in ligand_files:
        suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(ligand_file)
        # Iterate over the molecules in the file
        for mol in suppl:
            if mol is None:
                logging.warning(f"Failed to read molecule from {ligand_file}")
                continue

            # Get the number of atoms
            num_atoms: int = mol.GetNumAtoms()
            conformer: Chem.Conformer = mol.GetConformer()

            if num_atoms != 1:
                continue

            for atom in mol.GetAtoms():
                atom_idx: int = atom.GetIdx()
                atom_symbol: str = atom.GetSymbol()

                # skip non-metal cations such as ammonium (NH4+)
                if atom_symbol.upper() not in METAL_ATOMS_SET:
                    continue

                # check for positive charge
                formal_charge: int = atom.GetFormalCharge()
                if formal_charge <= 0:
                    continue  # Skip atoms that are not positively charged

                # Skip non-metal cations such as NH4+ (see REQUIRE_METAL_ELEMENT at
                # the top of this module).
                if REQUIRE_METAL_ELEMENT and atom_symbol.upper() not in METAL_ATOMS_SET:
                    continue

                # Get the coordinates of the atom
                pos: Chem.rdGeometry.Point3D = conformer.GetAtomPosition(atom_idx)
                x: float = pos.x
                y: float = pos.y
                z: float = pos.z

                metals.append((atom_symbol, (x, y, z)))
    return metals


def get_ligand_metal_min_distance(
    ligands_path: str, metals: list[tuple[str, tuple[float, float, float]]]
) -> list[tuple[str, float]]:
    """
    Calculate the minimum distance between ligands and metal ions in a complex.

    This function processes ligand files from a specified directory, identifies metal ions
    using the get_metals_from_ligand_dir function, and then calculates the minimum distance
    between each multi-atom ligand and any metal ion in the structure.

    Parameters
    ----------
    ligands_path : str
        Path to the directory containing ligand SDF files. These should be from a single complex.
    metals : list[tuple[str, tuple[float, float, float]]]
        A list of tuples containing the element symbol of the metal atom and its (x, y, z) coordinates, e.g., as obtained from get_metals_from_ligand_dir.

    Returns
    -------
    list of tuple:
        A list where each tuple contains two elements, the filename of the ligand SD file,
        and the minimum distance between the ligand and any metal ion.
    """
    # get all paths in folder ligands_path
    ligand_files: list[str] = [
        os.path.join(ligands_path, f) for f in os.listdir(ligands_path)
    ]

    res: list[tuple[str, float]] = []
    for ligand_file in ligand_files:
        suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(ligand_file)
        # assuming the sdf only holds one molecule
        rdkit_mol: Chem.rdchem.Mol = next(iter(suppl))

        if rdkit_mol is None:
            # remove all systems based on this complex by setting the distance to 1000.0, causing it to be removed by the minimum distance threshold
            logging.warning(
                f"Failed to read molecule from {ligand_file}. This system will be removed from the dataset by setting the minimum distance to 1000."
            )
            return [(f, 1000.0) for f in ligand_files]

        min_distance: float = 1000.0
        num_atoms: int = rdkit_mol.GetNumAtoms()

        # Identify multi-atom ligands
        if num_atoms > 1:
            for metal in metals:
                min_distance = min(
                    min_distance, get_atom_ligand_min_distance(rdkit_mol, metal[1])
                )

        if min_distance < 1000.0:
            res.append((ligand_file, min_distance))

    for ligand_path, min_dist in res:
        logging.info(
            f"Ligand found ({os.path.basename(ligand_path)}). Min_distance: {min_dist}."
        )

    return res


def get_atom_ligand_min_distance(
    ligand: Chem.rdchem.Mol, atom_coords: tuple[float, float, float]
) -> float:
    """
    This function calculates the minimum distance between any atom of a ligand and another atom.

    Parameters
    ----------
    ligand : Chem.rdchem.Mol
        Ligand molecule.
    atom_coords : tuple[float, float, float]
        xyz coords of the atom.

    Returns
    -------
    float
        Minimum distance of any atom of the ligand and the atom, or 1000.0.
    """
    min_distance: float = 1000.0
    conformer: Chem.Conformer = ligand.GetConformer()
    for atom in ligand.GetAtoms():
        if atom.GetAtomicNum() not in HETERO_ATOMS_NUM_SET:
            continue  # Skip non-hetero atoms
        atom_idx: int = atom.GetIdx()

        pos: Chem.rdGeometry.Point3D = conformer.GetAtomPosition(atom_idx)
        x: float = pos.x
        y: float = pos.y
        z: float = pos.z

        distance: float = np.linalg.norm(np.array([x, y, z]) - np.array(atom_coords))
        min_distance: float = min(min_distance, distance)

    return min_distance


def get_hetero_atoms_within_cutoff_single_ligand(
    ligand: Chem.rdchem.Mol,
    metal: tuple[str, tuple[float, float, float]],
    cutoff: float = 5.0,
) -> list[tuple[int, tuple[float, float, float]]]:
    """
    Find hetero atoms in a single ligand molecule within a specified distance cutoff from a given metal.

    Parameters
    ----------
    ligand : Chem.rdchem.Mol
        RDKit molecule object representing the ligand.
    metal : tuple[str, tuple[float, float, float]]
        Tuple containing metal name and 3D coordinates (name, (x, y, z)).
    cutoff : float, optional
        Maximum distance threshold in Angstroms (default is 5.0).

    Returns
    -------
    list[tuple[int, tuple[float, float, float]]]
        List of tuples containing atom indices and coordinates for hetero atoms
        within the cutoff distance: [(atom_idx, (x, y, z)), ...].
    """
    conformer: Chem.Conformer = ligand.GetConformer()

    res: list[tuple[int, tuple[float, float, float]]] = []
    metal_coords = metal[1]
    for atom in ligand.GetAtoms():
        # check if it is a hetero atom
        if atom.GetAtomicNum() in HETERO_ATOMS_NUM_SET:
            atom_idx: int = atom.GetIdx()

            pos: Chem.rdGeometry.Point3D = conformer.GetAtomPosition(atom_idx)
            x, y, z = pos.x, pos.y, pos.z
            distance: float = np.linalg.norm(
                np.array([x, y, z]) - np.array(metal_coords)
            )

            # check if the distance is within cutoff
            if distance < cutoff:
                res += [(atom.GetIdx(), (x, y, z))]

    return res


def get_hetero_atoms_within_cutoff(
    ligands_path: str,
    metals: list[tuple[str, tuple[float, float, float]]],
    cutoff: float = 5.0,
) -> list[
    tuple[
        tuple[str, tuple[float, float, float]],  # metal tuple
        str,  # ligand filename
        list[tuple[int, tuple[float, float, float]]],  # hetero atoms within cutoff
    ]
]:
    """
    Process multiple ligand files to find hetero atoms within cutoff distance from a set of metals.

    Parameters
    ----------
    ligands_path : str
        Directory path containing ligand SDFs.
    metals : list[tuple[str, tuple[float, float, float]]]
        List of metal tuples with name and coordinates [(name, (x, y, z)), ...].
    cutoff : float, optional
        Maximum distance threshold in Angstroms (default is 5.0).

    Returns
    -------
    tuple[tuple[str, tuple[float, float, float]], str, list[tuple[int, tuple[float, float, float]]]]
        List of tuples containing metal info, ligand filename, and nearby hetero atoms:
        [(metal_tuple, filename, hetero_atoms_list), ...].
    """
    ligand_files: list[str] = [
        os.path.join(ligands_path, f) for f in os.listdir(ligands_path)
    ]

    res: list[
        tuple[
            tuple[str, tuple[float, float, float]],  # metal tuple
            str,  # ligand filename
            list[tuple[int, tuple[float, float, float]]],  # hetero atoms within cutoff
        ]
    ] = []
    for ligand_file in ligand_files:
        suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(ligand_file)
        rdkit_mol: Chem.rdchem.Mol = next(
            iter(suppl)
        )  # assuming the sdf only holds one molecule

        if rdkit_mol is None:
            # Here, we simply catch the error and go on. Because we set the distance to 1000 before, the following parts will be irrelevant for the dataset regardless
            continue

        num_atoms: int = rdkit_mol.GetNumAtoms()

        if num_atoms > 1:
            for metal in metals:
                hetero_atoms_within_cutoff = (
                    get_hetero_atoms_within_cutoff_single_ligand(
                        rdkit_mol, metal, cutoff
                    )
                )

                res += [
                    (metal, os.path.basename(ligand_file), hetero_atoms_within_cutoff)
                ]

    return res
