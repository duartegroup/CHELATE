"""
Description
-----------
This module contains functions for finding and aligning an alternative structure to an original one.

Functions
---------
- `get_biotite_structure_with_metals(pdb_id: str, chain_id: str) -> biotite.structure.array`:
    Retrieves and filters a biotite structure for a specific chain and metals.

- `update_metal_identities(ligand_folder_path: str, atom_array: biotite.structure.AtomArray, dist_threshold: float = 2.0) -> list[biotite.structure.Atom]`:
    Updates the identities of metal atoms in the provided structure based on their proximity to metals in a reference structure.

- `get_metals_within_rm_metals_above_thresh(atoms_array: list[biotite.structure.Atom], main_ligand: Chem.rdchem.Mol, ligand_folder_path: str, thresholds: list[float] = [4.0, 10.0]) -> tuple[list[biotite.structure.Atom], list[int]]`:
    Count metal occurrences within given thresholds to a main ligand, and filter out metals beyond the largest threshold. Only the N closest metals to the main ligand are retained, where N is the number of metals in a reference structure.

- `get_anchor_rmsd(input_struct: biotite.structure.AtomArray, ref_struct: biotite.structure.AtomArray, input_anchor_indices: list[int], ref_anchor_indices: list[int]) -> float`:
    Calculates the RMSD between an input protein structure and a reference structure, only considering specified "anchor" atoms.

- `get_minimum_protein_ligand_distance(ligand: Chem.rdchem.Mol, protein: biotite.structure.AtomArray) -> float`:
    Calculate the minimum distance between a ligand and any protein atom.

- `process_row(row: pd.Series, alt_structures_df: pd.DataFrame, save_superimposed_structures: bool = True, cif_output_dir: str | None = None) -> dict[str, str | float | int] | None`:
    Processes a single row of a DataFrame to identify and align the best alternative structure for a given system.

- `merge_and_add_good_apo(plinder_df: pd.DataFrame, query_df: pd.DataFrame, good_apo_threshold: float = 70.0) -> pd.DataFrame`:
    Merges the DataFrame containing the results of the query for alternative structures with the original PLINDER DataFrame.

- `run_query(plinder_df: pd.DataFrame, has_good_apo_threshold: float = 70.0, save_superimposed_structures: bool = True) -> pd.DataFrame`:
    Iterates through the rows of the provided DataFrame, and gets the best alternative structures for each system.

- `run_single(system_id: str) -> dict[str, str | float | int] | None`:
    Get the best alternative structure for a single system.

Notes
-----
Unfortunately, biotite lacks type hints, so IDEs might not be able to recognize corresponding type annotations.
"""

import logging
import os

import biotite
import dataset_utils
import get_linked_structure
import numpy as np
import pandas as pd
from biotite.database.rcsb import fetch
from biotite.structure import array as biotite_array
from biotite.structure import superimpose_homologs
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure
from params import METAL_ATOMS_SET
from plinder.core import PlinderSystem
from rdkit import Chem


def get_biotite_structure_with_metals(
    pdb_id: str, chain_id: str
) -> biotite.structure.array:
    """
    Retrieves and filters a biotite structure for a specific chain and metals.

    This function fetches the structure of a given PDB ID from the Protein Data Bank (PDB)
    in CIF format, and filters it to only include atoms that are part of the specified chain
    or that are metals (determined by `METAL_ATOMS_SET`, which is defined in `params.py`).

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the protein structure to be fetched.
    chain_id : str
        The chain identifier (e.g., 'A', 'B') for which atoms should be filtered.

    Returns
    -------
    biotite.structure.array
        An `AtomArray` containing the filtered structure, with atoms from the specified chain
        and any metal atoms.
    """
    structure: biotite.structure.AtomArray = get_structure(
        CIFFile.read(
            fetch(
                pdb_ids=pdb_id,
                format="cif",
                overwrite=False,
            )
        ),
        model=1,
        use_author_fields=False,
    )

    filtered_biotite_atoms_array: list[biotite.structure.Atom] = []

    for atom in structure:
        if atom.chain_id == chain_id or (
            atom.hetero and atom.element in METAL_ATOMS_SET
        ):
            filtered_biotite_atoms_array.append(atom)

    return biotite_array(filtered_biotite_atoms_array)


def update_metal_identities(
    ligand_folder_path: str,
    atom_array: biotite.structure.AtomArray,
    dist_threshold: float = 2.0,
) -> list[biotite.structure.Atom]:
    """
    Updates the identities of metal atoms in the provided structure based on their proximity
    to metals in a reference structure. The reference metals are extracted from ligand files
    in the specified directory.

    Parameters
    ----------
    ligand_folder_path : str
        Path to the directory containing the ligand files, from which metal atoms are extracted.
    atom_array : biotite.structure.AtomArray
        A biotite `AtomArray` containing protein chain atoms, and metals to be updated.
    dist_threshold : float, default: 2.0
        The distance threshold (in Angstroms) to determine whether a metal in the provided
        structure corresponds to a metal ion in the reference structure.

    Returns
    -------
    list of biotite.structure.Atom
        A list of updated biotite `Atom` objects, where the identities of metal atoms have been
        updated to match those from the reference structure, if they fall within the distance
        threshold.

    Notes
    -----
    - For meaningful results, the provided structure should be aligned with the reference
      structure first.
    - Only heteroatoms classified as metals (`atom.hetero` and `atom.element in METAL_ATOMS_SET`,
      `METAL_ATOMS_SET` defined in `params.py`) are considered for updates.
    """
    holo_metals: list[tuple[str, tuple[float, float, float]]] = (
        dataset_utils.get_metals_from_ligand_dir(ligand_folder_path)
    )
    updated_structure: list[biotite.structure.Atom] = []

    for atom in atom_array:
        if atom.hetero and atom.element in METAL_ATOMS_SET:
            apo_metal_coords: np.ndarray = atom.coord
            for holo_metal_element, holo_metal_coords in holo_metals:
                apo_holo_distance: float = np.linalg.norm(
                    np.array(holo_metal_coords) - np.array(apo_metal_coords)
                )

                if (
                    apo_holo_distance <= dist_threshold
                    and atom.element != holo_metal_element
                ):
                    atom.res_name = holo_metal_element.upper()
                    atom.element = holo_metal_element.upper()
                    atom.atom_name = holo_metal_element.upper()

        updated_structure.append(atom)

    return updated_structure


def get_metals_within_rm_metals_above_thresh(
    atoms_array: list[biotite.structure.Atom],
    main_ligand: Chem.rdchem.Mol,
    ligand_folder_path: str,
    thresholds: list[float] | None = None,
) -> tuple[list[biotite.structure.Atom], list[int]]:
    """
    Count metal occurrences within given thresholds to a main ligand, and filter out metals
    beyond the largest threshold. Only the N closest metals to the main ligand are retained,
    where N is the number of metals in a reference structure.

    This function processes a list of atoms, filtering out metal atoms that are beyond a
    specified maximum distance from the `main_ligand`, counting the number of metal atoms
    that fall below each of the provided distance thresholds, and retaining only as many
    metals as are present in the reference structure.

    Parameters
    ----------
    atoms_array : list of biotite.structure.Atom
        A list of biotite `Atom` objects to be processed, including metal atoms.
    main_ligand : Chem.rdchem.Mol
        The RDKit `Mol` object representing the main ligand.
    ligand_folder_path : str
        Path to the directory containing the ligand files of a reference structure, from
        which metal atoms are extracted.
    thresholds : list of float, default: [4.0, 10.0]
        A list of distance thresholds (in Angstroms). Metals within each threshold are counted.
        The last value in the list is treated as the upper bound for filtering metals, i.e.,
        metals beyond this threshold are removed.

    Returns
    -------
    list of biotite.structure.Atom
        A list of biotite `Atom` objects, with metals filtered based on the distance thresholds
        and the number of metals in the reference structure.
    list of int
        A list of counts indicating how many metal atoms fall below each threshold in `thresholds`.

    Notes
    -----
    Uses the `METAL_ATOMS_SET` defined in `params.py` file to identify metal atoms.
    """
    if thresholds is None:
        thresholds = [4.0, 10.0]

    metals_below_thresholds: list[int] = [0] * len(thresholds)
    filtered_biotite_atoms_array: list[biotite.structure.Atom] = []
    ub: float = thresholds[-1]  # largest threshold as upper bound for removing metals

    holo_metals: list[tuple[str, tuple[float, float, float]]] = (
        dataset_utils.get_metals_from_ligand_dir(ligand_folder_path)
    )
    N_holo_metals: int = len(holo_metals)
    metals: list[tuple[float, biotite.structure.Atom]] = []

    for atom in atoms_array:
        if atom.hetero and atom.element in METAL_ATOMS_SET:
            metal_ligand_min_distance: float = (
                dataset_utils.get_atom_ligand_min_distance(main_ligand, atom.coord)
            )
            metals.append((metal_ligand_min_distance, atom))
        # if atom not metal, add to structure
        else:
            filtered_biotite_atoms_array.append(atom)

    metals.sort(key=lambda x: x[0])  # sort by distance

    for i, (d, metal) in enumerate(metals):
        # count number of metals below given threshold
        for j, threshold in enumerate(thresholds):
            if d <= threshold:
                metals_below_thresholds[j] += 1

        # only add first N_holo_metals metals and if distance below upper bound
        if i < N_holo_metals and d <= ub:
            filtered_biotite_atoms_array.append(metal)

    return filtered_biotite_atoms_array, metals_below_thresholds


def get_anchor_rmsd(
    input_struct: biotite.structure.AtomArray,
    ref_struct: biotite.structure.AtomArray,
    input_anchor_indices: list[int],
    ref_anchor_indices: list[int],
) -> float:
    """
    Calculates the RMSD between an input protein structure and a reference structure,
    only considering specified "anchor" atoms.

    Parameters
    ----------
    input_struct : biotite.structure.AtomArray
        Input protein structure.
    ref_struct : biotite.structure.AtomArray
        Reference protein structure.
    input_anchor_indices : list[int]
        A list of indices for the anchor atoms in the input structure.
    ref_anchor_indices : list[int]
        A list of indices for the anchor atoms in the reference structure.

    Returns
    -------
    float
        The RMSD between the anchor points in the input and reference structures.

    Notes
    -----
    The anchor atoms in the input structure should correspond to the anchor atoms in the
    reference structure. The anchor points in the reference structure are authorative, i.e.,
    the RMSD is calculated based on the anchor points in the reference structure.
    """
    total_squared_diffs: float = 0.0
    total_num_diffs: int = 0

    for i, ref_idx in enumerate(ref_anchor_indices):
        ref_atom: np.ndarray = ref_struct[ref_idx]
        inp_idx: int = input_anchor_indices[i]
        inp_atom: np.ndarray = input_struct[inp_idx]

        inp_xyz: np.ndarray = inp_atom.coord
        ref_xyz: np.ndarray = ref_atom.coord

        squared_diffs: np.ndarray = np.square(inp_xyz - ref_xyz)
        total_squared_diffs += np.sum(squared_diffs)
        total_num_diffs += len(inp_xyz)

    global_rmsd: float = np.sqrt(total_squared_diffs / total_num_diffs)
    return global_rmsd


def get_minimum_protein_ligand_distance(
    ligand: Chem.rdchem.Mol, protein: biotite.structure.AtomArray
) -> float:
    """
    Calculate the minimum distance between a ligand and any protein atom.

    Parameters
    ----------
    ligand : Chem.rdchem.Mol
        The RDKit `Mol` object representing the ligand. Assumes the molecule has a
        valid 3D conformer.
    protein : biotite.structure.AtomArray
        The biotite `AtomArray` representing the protein structure.

    Returns
    -------
    float
        The minimum Euclidean distance (in Angstroms) between any atom in `ligand`
        and any atom in `protein`.
    """
    min_distance: float = np.inf
    conformer: Chem.Conformer = ligand.GetConformer()

    for atom in ligand.GetAtoms():
        atom_idx: int = atom.GetIdx()
        pos: Chem.rdGeometry.Point3D = conformer.GetAtomPosition(atom_idx)
        x: float = pos.x
        y: float = pos.y
        z: float = pos.z

        for prot_atom in protein:
            distance: float = np.linalg.norm(np.array([x, y, z]) - prot_atom.coord)
            min_distance = min(min_distance, distance)

    return min_distance


def process_row(
    row: pd.Series,
    alt_structures_df: pd.DataFrame,
    save_superimposed_structures: bool = True,
    cif_output_dir: str | None = None,
    plinder_systems_path: str | None = None,
) -> dict[str, str | float | int] | None:
    """
    Processes a single row of a DataFrame to identify and align the best alternative
    structure for a given system.

    This function performs the following steps:
    1. Retrieves the receptor and ligand information for the query system.
    2. Finds the best alternative structure based on the highest `pocket_lddt` or `pocket_fident` value.
    3. Loads the alternative structure and aligns it to the original structure.
    4. Updates mismatched metals, and removes superfluous metals.
    5. Calculates structural metrics such as anchor RMSD and minimum protein-ligand distance.
    6. Optionally saves the aligned structure as a PDB file.

    Parameters
    ----------
    row : pd.Series
        A single row of a DataFrame for the system to be processed.
    alt_structures_df : pd.DataFrame
        DataFrame containing alternative structures for all systems.
    save_superimposed_structures : bool, default: True
        Whether to save the aligned structure as a PDB file.
    cif_output_dir : str, optional
        If given, also write the aligned structure as a Biotite mmCIF file to
        `{cif_output_dir}/{system_id}/{system_id}__{target_system}.cif`. This is the
        cross-docking input layout expected by `docking/run_docking.py` when
        `cif_output_dir` is a `{split}_systems` directory. Chain IDs are kept unchanged,
        since mmCIF allows multi-character chain IDs.
    plinder_systems_path : str, optional
        Local PLINDER systems directory. When omitted, resolve system files through
        ``PlinderSystem``.

    Returns
    -------
    dict or None
        A dictionary containing the processed results for the given system if successful,
        or None if an error occured. The output dictionary contains the following keys and values:
        - `system_id`: str, system ID of the processed system.
        - `target_system`: str, PDB ID and chain ID of alternative system as `<PDB ID>_<chain ID>`.
        - `pocket_fident`: float, sequence identity of the corresponding ligand-binding region of the alternative system to the query system.
        - `pocket_lddt`: float, structural similarity between corresponding ligand-binding region of the alternative system to the query system.
        - `protein_fident_qcov_weighted_sum`: float, weighted sum of protein fidelity and query coverage.
        - `protein_fident_weighted_sum`: float, weighted sum of protein fidelity.
        - `minimum_prot_lig_dist`: float, minimum distance between the main ligand and any protein atom in the alternative structure.
        - `anchor_rmsd`: float, anchor RMSD between alternative and query structures.
        - `num_metals_below_10A_main_ligand`: int, count of metals within 10A of the ligand in the alternative structure.
        - `num_metals_below_4A_main_ligand`: int, count of metals within 4A of the ligand in the alternative structure.

    Notes
    -----
    Skips systems with multiple chains, as PLINDER does not currently provide alternative structures for such systems.
    """
    system_id: str = row["system_id"]
    logging.info(f"Processing {system_id}.")

    n_chains: int = len(system_id.split("__")[2].split("_"))
    if n_chains > 1:
        logging.info(f"{system_id} has multiple chains. Skipping.")
        return None

    ligand_asym_id: str = row["ligand_asym_id"]
    if plinder_systems_path is None:
        plinder_system: PlinderSystem = PlinderSystem(system_id=system_id)
        receptor_pdb_path: str = plinder_system.receptor_pdb
        ligands_path: str = (
            receptor_pdb_path.split("receptor.pdb")[0] + "/ligand_files/"
        )
        ligand_paths = plinder_system.ligand_sdfs
        ligand_path = next(
            path
            for key, path in ligand_paths.items()
            if ligand_asym_id == key.split(".")[-1]
        )
    else:
        system_path = os.path.join(plinder_systems_path, system_id)
        receptor_pdb_path = os.path.join(system_path, "receptor.pdb")
        ligands_path = os.path.join(system_path, "ligand_files") + "/"
        ligand_path = os.path.join(
            ligands_path, f"{row['ligand_instance']}.{ligand_asym_id}.sdf"
        )
    suppl: Chem.SDMolSupplier = Chem.SDMolSupplier(ligand_path)
    main_ligand_rdkit: Chem.rdchem.Mol = next(iter(suppl))

    # Find best apo system
    best_plddt_row: pd.Series = pd.Series()
    best_plddt: float = 0
    best_pocket_fident_row: pd.Series = pd.Series()
    best_pocket_fident: float = 0
    cur_alt_systems: pd.DataFrame = alt_structures_df[
        alt_structures_df["query_system"] == system_id
    ]
    for _, alt_row in cur_alt_systems.iterrows():
        if alt_row["pocket_lddt"] > best_plddt:
            best_plddt_row: pd.Series = alt_row
            best_plddt: float = alt_row["pocket_lddt"]
        if alt_row["pocket_fident"] > best_pocket_fident:
            best_pocket_fident_row: pd.Series = alt_row
            best_pocket_fident = alt_row["pocket_fident"]

    # prefer pocket_lddt over pocket_fident
    if best_plddt == 0 or len(best_plddt_row) == 0:
        if best_pocket_fident != 0 and len(best_pocket_fident_row) > 0:
            best_row: pd.Series = best_pocket_fident_row
        else:
            logging.info("No alternative structure found.")
            return None
    else:
        best_row = best_plddt_row

    # Processing starts here
    target_system: str = best_row["target_system"]
    target_pdb_id: str = target_system.split("_")[0]
    target_chain_id: str = target_system.split("_")[1]

    try:
        file: PDBFile = PDBFile.read(receptor_pdb_path)
    except Exception as e:
        logging.warning(
            f"Could not read receptor PDB file for {system_id} at {receptor_pdb_path}: {e}"
        )
        return None
    try:
        original_pdb_structure: biotite.structure.AtomArray = file.get_structure()
    except Exception as e:
        logging.warning(
            f"Could not get structure from PDB file for {system_id} at {receptor_pdb_path}: {e}"
        )
        return None

    try:
        # Generate biotite atoms array of target chain of target system and metals
        target_biotite_atom_array: biotite.structure.array = (
            get_biotite_structure_with_metals(target_pdb_id, target_chain_id)
        )
    except Exception as e:
        logging.warning(
            f"Could not get biotite structure for {target_pdb_id} chain {target_chain_id}: {e}"
        )
        return None

    # Align structures
    aligned_target_structure: biotite.structure.AtomArray
    original_anchor_indices: list[int]
    target_anchor_indices: list[int]
    try:
        (
            aligned_target_structure,
            _,
            original_anchor_indices,
            target_anchor_indices,
        ) = superimpose_homologs(original_pdb_structure, target_biotite_atom_array)
    except ValueError:
        logging.warning(
            f"Could not superimpose structures for {system_id} with target {target_system} --> Skipping."
        )
        return None

    # Update mismatching metals
    target_atoms_updated_metals: list[biotite.structure.Atom] = update_metal_identities(
        ligands_path, aligned_target_structure
    )

    # Count and remove superfluous metals
    final_target_atoms: list[biotite.structure.Atom] = []
    metals_below_thresholds: list[int]
    final_target_atoms, metals_below_thresholds = (
        get_metals_within_rm_metals_above_thresh(
            target_atoms_updated_metals, main_ligand_rdkit, ligands_path
        )
    )
    num_metals_below_4A_main_ligand: int
    num_metals_below_10A_main_ligand: int
    num_metals_below_4A_main_ligand, num_metals_below_10A_main_ligand = (
        metals_below_thresholds
    )

    # Calculate anchor rmsd and minimum protein ligand distance
    anchor_rmsd: float = get_anchor_rmsd(
        aligned_target_structure,
        original_pdb_structure[0],
        target_anchor_indices,
        original_anchor_indices,
    )
    minimum_prot_lig_dist: float = get_minimum_protein_ligand_distance(
        main_ligand_rdkit, aligned_target_structure
    )

    # Collect results in dictionary format for later aggregation
    result: dict[str, str | float | int] = {
        "system_id": system_id,
        "target_system": target_system,
        "pocket_fident": best_row["pocket_fident"],
        "pocket_lddt": best_row["pocket_lddt"],
        "protein_fident_qcov_weighted_sum": best_row[
            "protein_fident_qcov_weighted_sum"
        ],
        "protein_fident_weighted_sum": best_row["protein_fident_weighted_sum"],
        "minimum_prot_lig_dist": minimum_prot_lig_dist,
        "anchor_rmsd": anchor_rmsd,
        "num_metals_below_10A_main_ligand": num_metals_below_10A_main_ligand,
        "num_metals_below_4A_main_ligand": num_metals_below_4A_main_ligand,
    }

    if cif_output_dir is not None:
        cif_atoms_array: biotite.structure.array = biotite_array(final_target_atoms)
        cif_dir = os.path.join(cif_output_dir, system_id)
        os.makedirs(cif_dir, exist_ok=True)
        cif_file: CIFFile = CIFFile()
        set_structure(cif_file, cif_atoms_array)
        cif_file.write(os.path.join(cif_dir, f"{system_id}__{target_system}.cif"))

    if save_superimposed_structures:
        final_target_atoms_array: biotite.structure.array = biotite_array(
            final_target_atoms
        )

        # Check if any chain ID has more than 1 character (which is not allowed by PDB format but can happen with mmCIF)
        ids = {atom.chain_id for atom in final_target_atoms_array}
        has_multi_char_chains = any(len(chain_id) > 1 for chain_id in ids)

        if has_multi_char_chains:
            # Create mapping from potentially multi-character chain IDs to single characters
            chain_mapping = {}
            for chain_counter, chain_id in enumerate(
                sorted(ids)
            ):  # Sort for consistent mapping
                chain_mapping[chain_id] = chr(ord("A") + chain_counter)

            # Apply chain ID mapping
            for i, atom in enumerate(final_target_atoms_array):
                atom.chain_id = chain_mapping[atom.chain_id]
                final_target_atoms_array[i] = atom

        save_path = "superimposed_structures"
        pdb_file: PDBFile = PDBFile()
        pdb_file.set_structure(final_target_atoms_array)
        os.makedirs(save_path, exist_ok=True)
        pdb_file.write(f"{save_path}/{system_id}__{target_system}.pdb")
    return result


def merge_and_add_good_apo(
    plinder_df: pd.DataFrame,
    query_df: pd.DataFrame,
    good_apo_threshold: float = 70.0,
) -> pd.DataFrame:
    """
    Merges the DataFrame containing the results of the query for alternative structures with
    the original PLINDER DataFrame. Adds a 'has_good_apo' column based on the `pocket_lddt`
    or `pocket_fident` scores and if the alternative structure contains metals within 4A of
    the main ligand.

    Parameters
    ----------
    plinder_df : pd.DataFrame
        The original PLINDER DataFrame.
    query_df : pd.DataFrame
        The DataFrame containing the results of the query for alternative structures.
    good_apo_threshold: float, default: 70.0
        Threshold for the `pocket_lddt` or `pocket_fident` scores, above which an alternative
        structure is considered "good".

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    # Add columns to plinder_df
    merged_df: pd.DataFrame = pd.merge(plinder_df, query_df, on="system_id", how="left")

    # Add has_good_apo column with good_apo_threshold
    has_good_apo_dict: dict[str, bool] = {}
    for _, row in merged_df.iterrows():
        if (
            (row["pocket_lddt"] >= good_apo_threshold)
            or (row["pocket_fident"] >= good_apo_threshold)
        ) and row["num_metals_below_4A_main_ligand"] > 0:
            has_good_apo_dict[row["system_id"]] = True

    merged_df["has_good_apo"] = False
    merged_df.loc[
        merged_df["system_id"].isin(has_good_apo_dict.keys()), "has_good_apo"
    ] = True
    return merged_df


def run_query(
    plinder_df: pd.DataFrame,
    good_apo_threshold: float = 70.0,
    save_superimposed_structures: bool = True,
) -> pd.DataFrame:
    """
    Iterates through the rows of the provided DataFrame, and gets the best alternative
    structures for each system. Will load cached results if available.

    This function performs the following steps:
    1. Gets the DataFrame linking system IDs to alternative structures.
    2. Processes each row of the input DataFrame to identify and align the best alternative
    structure, optionally saving the aligned structures.
    3. Updates the input DataFrame with the processed results.

    Parameters
    ----------
    plinder_df : pd.DataFrame
        The input DataFrame to be processed.
    good_apo_threshold: float, default: 70.0
        Threshold for the `pocket_lddt` or `pocket_fident` scores, above which an alternative
        structure is considered "good".
    save_superimposed_structures: bool, default: True
        Should the superimposed structures be saved.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with the processed results for each system.

    Notes
    -----
    Because predicted structures won't have metals, we only consider experimental structures.
    """
    if os.path.exists("cached_data/matched_final.csv"):
        logging.info("Superimposed data already exists. Loading...")
        matched_df: pd.DataFrame = pd.read_csv(
            "cached_data/matched_final.csv", keep_default_na=False, na_values=[""]
        )
        return merge_and_add_good_apo(plinder_df, matched_df, good_apo_threshold)

    logging.info("No cached results found. Running query now. This may take a while.")

    if not os.path.exists("cached_data/alt_structures_final.csv"):
        logging.info(
            "System IDs have not been linked to alternative structures yet. Querying now. This may take a while."
        )
        system_ids: list[str] = plinder_df["system_id"].unique().tolist()
        get_linked_structure.run_query(
            system_ids
        )  # saves the results under cached_data/alt_structures_final.csv

    alt_structures_df: pd.DataFrame = pd.read_csv(
        "cached_data/alt_structures_final.csv", keep_default_na=False, na_values=[""]
    )
    # Predicted structures won't have metals, so we only consider experimental structures
    alt_structures_df = alt_structures_df[alt_structures_df["type"] == "apo"]
    logging.info(f"Found {len(alt_structures_df)} alternative systems.")

    # Main loop
    results: list[dict] = []
    for _, row in plinder_df.iterrows():
        res: dict[str, str | float | int] | None = process_row(
            row, alt_structures_df, save_superimposed_structures
        )
        if res:
            results.append(res)

    # save results
    os.makedirs("cached_data", exist_ok=True)
    matched_df: pd.DataFrame = pd.DataFrame(results)
    matched_df.to_csv("cached_data/matched_final.csv", index=False)
    plinder_df = merge_and_add_good_apo(plinder_df, matched_df, good_apo_threshold)

    return plinder_df


def run_single(system_id: str) -> dict[str, str | float | int] | None:
    """
    Get the best alternative structure for a single system.

    Parameters
    ----------
    system_id : str
        The system ID to be processed.

    Returns
    -------
    dict or None
        A dictionary containing the processed results for the given system if successful,
        or None if an error occured.
    """
    print(f"Getting the best alternative structure for {system_id}.")

    rows: pd.DataFrame = dataset_utils.get_PLINDEX_row(system_id)
    if len(rows) == 0:
        print("ERROR: No rows with proper ligands found.")
        return

    # We only need to query once, even if multiple rows are present, since it's the same system
    row: pd.Series = rows.iloc[0]

    # Get all alternative structures from PLINDER
    linked_structures_df: pd.DataFrame | None = get_linked_structure.process_system(
        system_id
    )
    if linked_structures_df is None:
        print("ERROR: No alternative structures found.")
        return

    # Predicted structures won't have metals, so we only consider experimental structures
    linked_structures_df = linked_structures_df[linked_structures_df["type"] == "apo"]
    print(f"Found {len(linked_structures_df)} apo structures.")

    result: dict[str, str | float | int] | None = process_row(row, linked_structures_df)
    if result is None:
        print("ERROR: Could not process alternative structure(s).")
        return

    print("Aligned structure saved.")
    return result
