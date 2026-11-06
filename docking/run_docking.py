"""
run_docking.py
Performs molecular docking for protein-ligand systems using VINA, SMINA, or GNINA.
Processes datasets, prepares files, and executes docking simulations with support
for both re-docking and cross-docking workflows. Includes options for hydrogen
placement optimization and customizable docking parameters.

Usage: Run with arguments:
    --split: Split that should be docked [train, val, test].
    --method: Docking method [VINA, SMINA, GNINA]
    --id: Job identifier
    --redocking / --no-redocking: Re-docking or cross-docking (default: cross-docking)
    --add_hydrogen / --no-add_hydrogen: Optimize hydrogens with Reduce2 (default: on)
    --exhaustiveness: Search exhaustiveness
    --num_modes: Number of poses to generate
    --energy_range: Max energy window for Vina/Smina (kcal/mol; GNINA does not support it)
    --box_extension: Search space extension (Å)
    --dataset_path: Path to CSV file containing the dataset
    --representative_train_systems_path: Path to representative systems CSV file for the training set, if applicable.
    --systems_root: Directory holding the {split}_systems folders (default: $SYSTEMS_ROOT or repository/docking)

Example:
    python docking/run_docking.py --split test --method GNINA --id 42 --redocking \
        --dataset_path data_curation/cached_data/final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.csv
"""

import argparse
import concurrent.futures
import json
import os
import pickle
import shlex
import subprocess
import threading

import docking_utils
import pandas as pd
from cif_utils import prepare_ligand_receptor_source
from docking_params import (
    CACHED_DATA_PATH,
    CLUSTER_REPRESENTATIVES_CSV,
    DATASET_CSV,
    DEFAULT_SYSTEMS_ROOT,
    GNINA_PATH,
    GPDOCK_METAL_MAPPING,
    METAL_LIGAND_DATA_JSON,
    METALLODOCK_MODEL,
    METALLODOCK_VOC,
    PLINDER_SYSTEMS_PATH,
    SMINA_PATH,
    VINA_PATH,
)
from tqdm import tqdm

from docking.box_options import fixed_box_options
from docking.receptor_gate import require_complete_receptor


def run_docking(
    split: str = "test",
    method: str = "VINA",
    id: str = "1",
    redocking: bool = True,
    add_hydrogen: bool = True,
    exhaustiveness: int = 32,
    num_modes: int = 100,
    energy_range: float = 20.0,
    box_extension: float = 4.0,
    dataset_path: str = DATASET_CSV,
    representative_train_systems_path: str = CLUSTER_REPRESENTATIVES_CSV,
    systems_root: str = DEFAULT_SYSTEMS_ROOT,
    metal_ligand_data_path: str | None = None,
    autodock_nrun: int = 100,
    autodock_batch_size: int = 8,
    autodock_spacing: float = 0.375,
    autodock_max_points: int = 255,
    autodock_prep_timeout: int = 3600,
    autodock_docking_timeout: int = 7200,
    num_workers: int = 1,
    output_suffix: str = "",
    gpdock_num_waters: int = 0,
    gpdock_fe_oxidation_state: int = 6,
    gpdock_docking_timeout: int = 90 * 60,
    metallodock_min_method: str = "uff",
    metallodock_batch_size: int = 64,
    metallodock_pocket_extraction_radius: float = 4.0,
    unseen_cluster_csv_path: str = f"{CACHED_DATA_PATH}/split_clusters.csv",
    unseen_map_path: str | None = None,
):
    """
    Run molecular docking for a set of protein-ligand systems.

    This function processes each system in the dataset, prepares receptor and ligand files,
    and performs docking using the specified method (VINA, SMINA, GNINA, AUTODOCK,
    METALLODOCK, or GPDOCK). Results are saved to CSV files. To run this function, ensure
    the presence of a `metal_ligand_data` file in `data_curation/cached_data`.

    Parameters
    ----------
    split : str, default="test"
        Split that should be docked (train, val, or test). All matching systems from the dataset
        will be docked.
    method : str, default="VINA"
        Docking method to use (VINA, SMINA, GNINA, AUTODOCK, METALLODOCK, or GPDOCK)
    id : str, default="1"
        Identifier for this docking run
    redocking : bool, default=True
        If True, performs re-docking (same protein structure), else cross-docking
    add_hydrogen : bool, default=True
        If True, adds hydrogen atoms to the receptor using Reduce
    exhaustiveness : int, default=32
        Search exhaustiveness parameter for docking
    num_modes : int, default=100
        Number of binding modes to generate
    energy_range : float, default=20.0
        Maximum energy difference between best and worst binding mode
    box_extension : float, default=4.0
        Amount by which to extend the docking box beyond the ligand dimensions (Angstroms)
    dataset_path : str, default="data_curation/cached_data/final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.csv"
        Path to CSV file containing the dataset
    representative_train_systems_path : str, default="data_curation/cached_data/cluster_representatives_5.csv"
        Path to CSV file containing representative systems for the training set. Only
        needed if the split is 'train'.
    systems_root : str, default=repository/docking
        Directory containing the train_systems, val_systems, and test_systems folders.
    metal_ligand_data_path : str, optional
        Explicit path to the cached metal-coordinate JSON. When omitted, the path is
        derived from the dataset filename and the current working directory.
    autodock_nrun : int, default=100
        Number of AutoDock LGA runs (only used when method="AUTODOCK")
    autodock_batch_size : int, default=8
        AutoDock batch size for GPU processing (only used when method="AUTODOCK")
    autodock_spacing : float, default=0.375
        Grid spacing in Angstroms for AutoDock (only used when method="AUTODOCK")
    autodock_max_points : int, default=255
        Max points per grid dimension for AutoDock (only used when method="AUTODOCK")
    autodock_prep_timeout : int, default=3600
        Maximum time in seconds to allow for AutoDock grid generation and preparation before aborting (only used when method="AUTODOCK")
    autodock_docking_timeout : int, default=7200
        Maximum time in seconds to allow for AutoDock (batch) docking before aborting (only used when method="AUTODOCK")
    num_workers : int, default=1
        Number of parallel workers for docking (used for AutoDock grid generation and docking parallelization)
    output_suffix : str, default=""
        Optional suffix for aggregate result and error CSVs. This lets independent
        dataset shards run concurrently without overwriting each other's summaries.
    gpdock_num_waters : int, default=0
        Number of water molecules for GPDOCK docking (0-2, only used when method="GPDOCK").
        Ignored for multi-metal systems.
    gpdock_fe_oxidation_state : int, default=6
        Fe oxidation state for GPDOCK: 5=Fe2+, 6=Fe3+ (only used when method="GPDOCK").
    gpdock_docking_timeout : int, default=90*60
        Maximum time in seconds to allow for GPDOCK docking before aborting (only used when method="GPDOCK").
    metallodock_min_method : str, default="uff"
        Force-field minimisation method for MetalloDock: 'uff' (fast) or 'openff2' (more accurate,
        only used when method="METALLODOCK").
    metallodock_batch_size : int, default=64
        Batch size for MetalloDock model inference (only used when method="METALLODOCK").
    metallodock_pocket_extraction_radius : float, default=4.0
        Radius in Angstroms of the ball used to encapsulate the Vina-style box around the ligand when extracting the binding pocket (only used when method="METALLODOCK").
    unseen_cluster_csv_path : str, default="data_curation/cached_data/split_clusters.csv"
        Path to the clustering CSV file, assigning systems to clusters. Only needed for unseen split.
    unseen_map_path : str, default=None
        Path to a map pickle file that maps systems to be either in the training or unseen set. Only needed for unseen split.

    Returns
    -------
    None
        Results are saved to CSV files in the specified `<split>_systems/"`
    """
    dataset_identifier = (
        os.path.basename(dataset_path).replace("final_dataset_", "").replace(".csv", "")
    )  # assumes format from data_curation/generate_dataset.py
    if metal_ligand_data_path is None:
        default_dataset_identifier = (
            os.path.basename(DATASET_CSV)
            .replace("final_dataset_", "")
            .replace(".csv", "")
        )
        if dataset_identifier == default_dataset_identifier:
            metal_ligand_data_path = METAL_LIGAND_DATA_JSON
        else:
            metal_ligand_data_path = os.path.join(
                os.path.dirname(os.path.abspath(dataset_path)),
                f"metal_ligand_data_{dataset_identifier}.json",
            )
    with open(metal_ligand_data_path) as f:
        metal_ligand_data: dict = json.load(f)

    data_path: str = os.path.join(systems_root, f"{split}_systems")

    docking_type: str = "redocked" if redocking else "apo_docked"
    if add_hydrogen:
        docking_type += "_FH"

    # "NA" is a valid ligand chain ID and must not be parsed as a missing value.
    plinder_df: pd.DataFrame = pd.read_csv(dataset_path, keep_default_na=False)
    plinder_df = plinder_df[plinder_df["split"] == split]

    # For the train set, we only dock the representative systems
    if split == "train":
        repr: pd.DataFrame = pd.read_csv(
            representative_train_systems_path, keep_default_na=False
        )
        num_systems_before: int = len(plinder_df)
        plinder_df = plinder_df[
            plinder_df["system_id"].isin(repr["representative_system_id"])
        ]
        num_systems_after: int = len(plinder_df)
        if num_systems_before != num_systems_after:
            print(
                f"Filtered training set to {num_systems_after} representative systems (from {num_systems_before} total systems in the training split)."
            )

    elif split == "unseen":
        plinder_df: pd.DataFrame = pd.read_csv(dataset_path, keep_default_na=False)

        # delete removed systems
        num_systems_before: int = len(plinder_df)
        plinder_df = plinder_df[plinder_df["split"] != "removed"]
        num_systems_after: int = len(plinder_df)
        if num_systems_before != num_systems_after:
            print(
                f"Removed {num_systems_before - num_systems_after} systems marked as 'removed' in the dataset CSV."
            )

        cluster_df = pd.read_csv(unseen_cluster_csv_path, keep_default_na=False)[
            ["system_id", "cluster"]
        ]
        plinder_df = plinder_df.merge(cluster_df, on="system_id", how="left")
        assert unseen_map_path, "unseen_map_path must be provided for unseen split"
        with open(unseen_map_path, "rb") as map_f:
            temporal_map: dict = pickle.load(map_f)
        plinder_df["is_cluster_unseen"] = plinder_df["system_id"].map(
            lambda x: temporal_map.get(x, {}).get("is_cluster_unseen", False)
        )
        plinder_df = plinder_df[plinder_df["is_cluster_unseen"].fillna(False)]
        cluster_sizes = (
            plinder_df.groupby("cluster").size().reset_index(name="cluster_size")
        )
        plinder_df = plinder_df.merge(cluster_sizes, on="cluster", how="left")
        plinder_df = plinder_df.groupby("cluster").sample(n=1, random_state=int(id))

    global_df: pd.DataFrame = pd.DataFrame()
    errors_df: pd.DataFrame = pd.DataFrame(columns=["system_id", "error_message"])

    # For AutoDock: collect systems during loop for batch processing
    if method == "AUTODOCK":
        autodock_systems: list[dict] = []
        autodock_systems_lock: threading.Lock = threading.Lock()

    def _dock_row(row):
        """
        Helper function to process a single row.

        Returns a tuple of (results_df, system_id, alt_system_id, ligand_id, extra_errors),
        or (None, system_id, None, None, extra_errors) if an error occurs.
        """
        extra_errors: list[dict] = []
        system_id: str = row["system_id"]
        try:
            target_system: str = row["target_system"]
            alt_system_id: str = f"{system_id}__{target_system}"
            ligand_instance: str = row["ligand_instance"]
            ligand_asym_id: str = row["ligand_asym_id"]
            ligand_id: str = f"{ligand_instance}.{ligand_asym_id}"
            metal_coords: list[tuple[float, float, float]] = [
                m[1] for m in metal_ligand_data[system_id]["metals"]
            ]

            print("-" * 50)
            print(f"Docking {system_id}")
            print("-" * 50)

            base_path: str = f"{data_path}/{system_id}"
            os.makedirs(base_path, exist_ok=True)

            ### Copy files if not present
            ligand_chains: list[str] = system_id.split("__")[-1].split("_")
            for asym_id in ligand_chains:
                if not os.path.exists(f"{base_path}/{asym_id}.sdf"):
                    subprocess.run(
                        [
                            "cp",
                            f"{PLINDER_SYSTEMS_PATH}/{system_id}/ligand_files/{asym_id}.sdf",
                            f"{base_path}/",
                        ]
                    )
                    if not os.path.exists(f"{base_path}/{asym_id}.sdf"):
                        raise FileNotFoundError(
                            f"Missing ligand SDF: {PLINDER_SYSTEMS_PATH}/{system_id}/ligand_files/{asym_id}.sdf"
                        )

            if not os.path.exists(f"{base_path}/system.cif"):
                subprocess.run(
                    [
                        "cp",
                        f"{PLINDER_SYSTEMS_PATH}/{system_id}/system.cif",
                        f"{base_path}/",
                    ]
                )
                if not os.path.exists(f"{base_path}/system.cif"):
                    raise FileNotFoundError(
                        f"Missing system CIF: {PLINDER_SYSTEMS_PATH}/{system_id}/system.cif"
                    )

            ### ** FILE PATHS **
            # Re-docking
            if redocking:
                pair_input_path = f"{base_path}/receptor_inputs/{ligand_id}"
                system_cif: str = f"{base_path}/system.cif"
                system_cif_wo_solvent: str = f"{pair_input_path}/system_wo_solvent.cif"
                system_cif_sanitized: str = f"{pair_input_path}/system_sanitized.cif"
                system_cif_wo_unsupported: str = (
                    f"{pair_input_path}/system_wo_unsupported.cif"
                )
                system_cif_reduce: str = (
                    f"{pair_input_path}/system_wo_unsupported_FH.cif"
                )
                system_cif_unsupported: str = (
                    f"{pair_input_path}/system_unsupported.cif"
                )
                system_pdbqt_unsupported: str = (
                    f"{pair_input_path}/system_unsupported.pdbqt"
                )
                receptor_pdbqt_path_merged: str = (
                    f"{base_path}/receptor_merged_{ligand_id}.pdbqt"
                )

            # Cross-docking
            else:
                system_cif = f"{base_path}/{alt_system_id}.cif"
                system_cif_wo_solvent = f"{base_path}/{alt_system_id}_wo_solvent.cif"
                system_cif_sanitized = f"{base_path}/{alt_system_id}_sanitized.cif"
                system_cif_wo_unsupported = (
                    f"{base_path}/{alt_system_id}_wo_unsupported.cif"
                )
                system_cif_reduce = f"{base_path}/{alt_system_id}_wo_unsupported_FH.cif"
                system_cif_unsupported = f"{base_path}/{alt_system_id}_unsupported.cif"
                system_pdbqt_unsupported = (
                    f"{base_path}/{alt_system_id}_unsupported.pdbqt"
                )
                receptor_pdbqt_path_merged = (
                    f"{base_path}/{alt_system_id}_merged_{ligand_id}.pdbqt"
                )

            ligand_sdf_path: str = f"{base_path}/{ligand_id}.sdf"
            ligand_pdbqt_path: str = f"{base_path}/{ligand_id}.pdbqt"

            ### ** RECEPTOR PIPELINE **
            if not os.path.exists(receptor_pdbqt_path_merged):
                if redocking:
                    prepare_ligand_receptor_source(
                        system_cif, ligand_id, pair_input_path
                    )
                else:
                    if not docking_utils.remove_solvent_cif(
                        system_cif, system_cif_wo_solvent
                    ):
                        raise RuntimeError(
                            f"Error removing solvent from system CIF: {system_id}"
                        )
                    system_cif_sanitized = system_cif_wo_solvent

                # Remove unsupported atoms
                if not docking_utils.remove_unsupported_atoms_cif(
                    system_cif_sanitized,
                    system_cif_wo_unsupported,
                    system_cif_unsupported,
                ):
                    raise RuntimeError(
                        f"Error removing unsupported atoms from system CIF: {system_id}"
                    )

                # Convert unsupported CIF to PDBQT
                if redocking:
                    if not docking_utils.convert_unsupported_cif_to_pdbqt(
                        system_cif_unsupported,
                        system_pdbqt_unsupported,
                        format_type="PLINDER",
                    ):
                        raise RuntimeError(
                            f"Error converting CIF file containing unsupported atoms to PDBQT: {system_id}"
                        )
                else:
                    if not docking_utils.convert_unsupported_cif_to_pdbqt(
                        system_cif_unsupported,
                        system_pdbqt_unsupported,
                        format_type="BIOTITE",
                    ):
                        raise RuntimeError(
                            f"Error converting CIF file containing unsupported atoms to PDBQT: {system_id}"
                        )

                # Optimize hydrogen placement using Reduce
                if add_hydrogen:
                    try:
                        add_hydrogen_reduce_success, _ = (
                            docking_utils.add_hydrogen_reduce(
                                system_cif_wo_unsupported, system_cif_reduce
                            )
                        )
                        if not add_hydrogen_reduce_success:
                            raise RuntimeError("Hydrogen addition using Reduce failed")

                        # Cleans a CIF file by removing empty lines and unnecessary indentations, overwriting the original file.
                        if not docking_utils.trim_cif_file(system_cif_reduce):
                            raise RuntimeError("Error correcting Reduce output")

                        # If Reduce succeeds, use its output
                        system_cif_wo_unsupported = system_cif_reduce

                    except Exception as err:
                        print(
                            f"WARNING: Hydrogen optimization failed for {system_id}. Proceeding with original system CIF."
                        )
                        extra_errors.append(
                            {
                                "system_id": system_id,
                                "error_message": f"Reduce failed: {err}",
                            }
                        )

                receptor_pdbqt_path: str = system_cif_wo_unsupported.replace(
                    ".cif", ".pdbqt"
                )
                # Prepare receptor with Meeko
                receptor_prep_success, _ = docking_utils.prepare_receptor_meeko(
                    system_cif_wo_unsupported, receptor_pdbqt_path, file_type="cif"
                )
                if not receptor_prep_success:
                    raise RuntimeError(
                        f"Error preparing receptor with Meeko: {system_id}"
                    )

                # Merge receptor with unsupported atoms
                if not docking_utils.merge_pdbqt_files_with_fixed_indices(
                    receptor_pdbqt_path,
                    system_pdbqt_unsupported,
                    receptor_pdbqt_path_merged,
                ):
                    raise RuntimeError(
                        f"Error merging receptor with unsupported atoms: {system_id}"
                    )

            receptor_pdbqt: str = receptor_pdbqt_path_merged
            if method in {"VINA", "SMINA", "GNINA"}:
                center, size = docking_utils.get_box(
                    ligand_sdf_path, extending=box_extension, software="vina"
                )
                shared_box = {**center, **size}
                require_complete_receptor(
                    system_cif,
                    receptor_pdbqt,
                    ligand_id if redocking else None,
                    docking_box=shared_box,
                )

            ### ** LIGAND PIPELINE **
            # Not collapsed: merging the guard with the preparation call would hide a
            # side-effecting step inside a boolean expression.
            # GNINA consumes the pH-prepared SDF directly. Regenerating a PDBQT
            # here is unnecessary and can overwrite a validated scrubbed SDF.
            needs_ligand_pdbqt = method not in [
                "METALLODOCK",
                "GNINA",
            ] and not os.path.exists(ligand_pdbqt_path)
            if needs_ligand_pdbqt:  # noqa: SIM102
                if not docking_utils.prepare_ligand_meeko(
                    ligand_sdf_path, ligand_pdbqt_path
                ):
                    raise RuntimeError(f"Error preparing ligand: {system_id}")

            ### ** OUTPUT **
            if method in ["VINA", "SMINA"]:
                docked_path: str = f"{base_path}/{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_eRange={energy_range}_lig={ligand_id}.pdbqt"
            else:  # GNINA
                docked_path = f"{base_path}/{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_prepPH=7.4_lig={ligand_id}.sdf.gz"

            ### ** DOCKING **
            if method == "VINA":
                if not os.path.exists(docked_path):
                    # Create docking config file
                    config_path: str = f"{base_path}/cfg.txt"
                    config_lines: list[str] = [
                        f"center_x = {center['center_x']}",
                        f"center_y = {center['center_y']}",
                        f"center_z = {center['center_z']}",
                        f"size_x = {size['size_x']}",
                        f"size_y = {size['size_y']}",
                        f"size_z = {size['size_z']}",
                    ]
                    with open(config_path, "w") as file:
                        file.write("\n".join(config_lines) + "\n")

                    # Run AutoDock Vina
                    vina_command: list[str] = [
                        VINA_PATH,
                        "--receptor",
                        receptor_pdbqt,
                        "--ligand",
                        ligand_pdbqt_path,
                        "--config",
                        config_path,
                        "--exhaustiveness",
                        str(exhaustiveness),
                        "--num_modes",
                        str(num_modes),
                        "--energy_range",
                        str(energy_range),
                        "--out",
                        docked_path,
                    ]
                    subprocess.run(vina_command)
                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    docked_path,
                    ligand_sdf_path,
                    metal_coords,
                    docking_type="vina_pdbqt",
                )

            if method == "SMINA":
                if not os.path.exists(docked_path):
                    smina_cmd: list[str] = [
                        SMINA_PATH,
                        "--receptor",
                        receptor_pdbqt,
                        "--ligand",
                        ligand_pdbqt_path,
                        *fixed_box_options(method, shared_box),
                        f"--exhaustiveness={exhaustiveness}",
                        "--num_modes",
                        str(num_modes),
                        "--energy_range",
                        str(energy_range),
                        "--out",
                        docked_path,
                    ]
                    # Run SMINA
                    subprocess.run(smina_cmd)
                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    docked_path, ligand_sdf_path, metal_coords, docking_type="smina"
                )
            if method == "GNINA":
                if not os.path.exists(docked_path):
                    scrubbed_ligand_path = (
                        os.path.splitext(ligand_sdf_path)[0] + "_scrubbed.sdf"
                    )
                    if not os.path.exists(scrubbed_ligand_path):
                        raise FileNotFoundError(
                            f"Scrubber-prepared ligand not found: {scrubbed_ligand_path}"
                        )
                    gnina_command: list[str] = [
                        *shlex.split(GNINA_PATH),
                        "--receptor",
                        receptor_pdbqt,
                        "--ligand",
                        scrubbed_ligand_path,
                        f"--exhaustiveness={exhaustiveness}",
                        "--num_modes",
                        str(num_modes),
                        "--seed",
                        str(id),
                        *fixed_box_options(method, shared_box),
                        "--out",
                        docked_path,
                    ]
                    # Run GNINA
                    subprocess.run(gnina_command, check=True)

                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    docked_path, ligand_sdf_path, metal_coords, docking_type="gnina"
                )

            elif method == "GPDOCK":
                docked_path = f"{base_path}/GPDOCK_{docking_type}_id={id}_HOH={gpdock_num_waters}_lig={ligand_id}_out.pdb"

                if not os.path.exists(docked_path):
                    metals: list = metal_ligand_data[system_id]["metals"]
                    multi_metal: bool = len(metals) > 1
                    metal_symbol: str = metals[0][0].upper()

                    if metal_symbol == "FE":
                        ion_code: int = gpdock_fe_oxidation_state  # 5=Fe2+, 6=Fe3+
                    elif metal_symbol in GPDOCK_METAL_MAPPING:
                        ion_code = GPDOCK_METAL_MAPPING[metal_symbol]
                    else:
                        raise RuntimeError(
                            f"Metal '{metal_symbol}' not supported by GPDOCK (system: {system_id})"
                        )

                    raw_out = docking_utils.run_gpdock_docking(
                        receptor_pdbqt,
                        ligand_sdf_path,
                        ion_code,
                        gpdock_num_waters,
                        base_path + f"/GPDOCK_raw_{docking_type}_id={id}/",
                        multi_metal=multi_metal,
                        timeout=gpdock_docking_timeout,
                    )
                    os.rename(raw_out, docked_path)

                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    docked_path, ligand_sdf_path, metal_coords, docking_type="gpdock"
                )

            elif method == "AUTODOCK":
                docked_dlg: str = (
                    f"{base_path}/AUTODOCK_{docking_type}_id={id}_nrun={autodock_nrun}_"
                    f"boxExt={box_extension}_{system_id}_lig={ligand_id}.dlg"
                )

                # Check if already docked (use cached result)
                if os.path.exists(docked_dlg):
                    results_df: pd.DataFrame = docking_utils.get_docking_data(
                        docked_dlg,
                        ligand_sdf_path,
                        metal_coords,
                        docking_type="autodock",
                    )

                    if redocking:
                        results_df["system_id"] = system_id
                    else:
                        results_df["alt_system_id"] = alt_system_id
                    results_df["ligand_id"] = ligand_id
                else:
                    # Collect for batch processing
                    with autodock_systems_lock:
                        autodock_systems.append(
                            {
                                "system_id": system_id,
                                "ligand_id": ligand_id,
                                "receptor_pdbqt": receptor_pdbqt,
                                "ligand_pdbqt": ligand_pdbqt_path,
                                "ligand_sdf": ligand_sdf_path,
                                "metal_coords": metal_coords,
                                "output_dir": base_path,
                                "docked_dlg": docked_dlg,
                                "output_name": os.path.basename(docked_dlg).replace(
                                    ".dlg", ""
                                ),
                                "redocking": redocking,
                                "alt_system_id": (
                                    alt_system_id if not redocking else None
                                ),
                            }
                        )

                # Skip immediate execution, will be processed in batch after the loop
                return (None, system_id, alt_system_id, ligand_id, extra_errors)

            elif method == "METALLODOCK":
                docked_path = f"{base_path}/METALLODOCK_none_leak_{docking_type}_id={id}_lig={ligand_id}_min_method={metallodock_min_method}.sdf"

                md_pdbid: str = (
                    (system_id if redocking else alt_system_id) + f"__{ligand_id}"
                ).replace(".", "_")
                md_work_dir: str = f"{base_path}/METALLODOCK_{docking_type}_id={id}"
                md_pocket_pdb: str = f"{md_work_dir}/receptor_pocket.pdb"
                md_ligand_dir: str = f"{md_work_dir}/ligands"
                md_ligand_sdf: str = f"{md_ligand_dir}/{md_pdbid}.sdf"
                md_out_dir: str = f"{md_work_dir}/result"

                os.makedirs(md_ligand_dir, exist_ok=True)
                os.makedirs(md_out_dir, exist_ok=True)

                if not os.path.exists(docked_path):
                    # 1. Convert receptor to PDB
                    md_receptor_pdb = f"{md_work_dir}/full_receptor.pdb"
                    if not os.path.exists(md_receptor_pdb):  # noqa: SIM102
                        if not docking_utils.convert_pdbqt_to_pdb_pybel(
                            receptor_pdbqt, md_receptor_pdb
                        ):
                            raise RuntimeError(
                                f"Error converting unified PDBQT to PDB for MetalloDock: {system_id}"
                            )

                    # 2. Extract pocket based on the box coordinates
                    if not os.path.exists(md_pocket_pdb):
                        if not docking_utils.extract_metallodock_pocket(
                            md_receptor_pdb,
                            ligand_sdf_path,
                            md_pocket_pdb,
                            ball_radius=metallodock_pocket_extraction_radius,
                        ):
                            raise RuntimeError(
                                f"Error extracting bounding pocket for MetalloDock: {system_id}"
                            )

                        # Add metal coordination bonds for proper graph generation
                        docking_utils.add_conect_to_pdb(md_pocket_pdb)

                    # 3. Generate ligand conformer
                    if not os.path.exists(md_ligand_sdf):  # noqa: SIM102
                        if not docking_utils.generate_blind_ligand(
                            ligand_sdf_path, md_ligand_sdf, seed=int(id)
                        ):
                            raise RuntimeError(
                                f"Error generating input ligand conformer for MetalloDock: {system_id}"
                            )

                    # 4. Docking
                    if not docking_utils.run_metallodock_subprocess(
                        "virtual_screening.py",
                        {
                            "--pocket_file": md_pocket_pdb,
                            "--ligand_file": md_ligand_dir,
                            "--out_dir": md_out_dir,
                            "--model_file": METALLODOCK_MODEL,
                            "--voc_file": METALLODOCK_VOC,
                            "--min_method": metallodock_min_method,
                            "--batch_size": str(metallodock_batch_size),
                            "--random_seed": str(id),
                            "--num_poses": str(num_modes),
                        },
                    ):
                        raise RuntimeError(
                            f"MetalloDock virtual_screening.py failed: {system_id}"
                        )

                    # The output SDF will be the minimized poses if minimization is activated
                    output_sdf: str = f"{md_out_dir}/{md_pdbid}_poses.sdf"
                    if not os.path.exists(output_sdf):
                        raise RuntimeError(
                            f"MetalloDock did not produce any output SDF for: {md_pdbid}"
                        )
                    os.rename(output_sdf, docked_path)

                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    docked_path,
                    ligand_sdf_path,
                    metal_coords,
                    docking_type="metallodock",
                )

            return (results_df, system_id, alt_system_id, ligand_id, extra_errors)

        except Exception as err:
            print(f"Error processing {system_id}: {err}")
            extra_errors.append({"system_id": system_id, "error_message": str(err)})
            return (None, system_id, None, None, extra_errors)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for (
            results_df,
            system_id,
            alt_system_id,
            ligand_id,
            extra_errors,
        ) in tqdm(
            executor.map(_dock_row, [row for _, row in plinder_df.iterrows()]),
            total=len(plinder_df),
            desc="Docking",
        ):
            for e in extra_errors:
                errors_df = pd.concat([errors_df, pd.DataFrame([e])], ignore_index=True)
            if results_df is not None:
                if redocking:
                    results_df["system_id"] = system_id
                else:
                    results_df["alt_system_id"] = alt_system_id
                results_df["ligand_id"] = ligand_id
                global_df = pd.concat([global_df, results_df])

    # AutoDock batch processing
    if method == "AUTODOCK" and autodock_systems:
        print("=" * 60)
        print(f"AUTODOCK BATCH PROCESSING: {len(autodock_systems)} systems")
        print("=" * 60)

        # Step 1: Generate all grids in parallel
        print(f"\n[1/3] Generating {len(autodock_systems)} grids in parallel...")

        systems_for_grid: list[dict] = [
            {
                "system_id": s["system_id"],
                "receptor_pdbqt": s["receptor_pdbqt"],
                "ligand_sdf": s["ligand_sdf"],
                "output_dir": s["output_dir"] + f"/grids_{docking_type}",
            }
            for s in autodock_systems
        ]

        grid_map: dict[str, str] = docking_utils.preprocess_autodock_systems_parallel(
            systems_for_grid,
            num_workers=num_workers,
            box_extension=box_extension,
            spacing=autodock_spacing,
            max_points=autodock_max_points,
            timeout=autodock_prep_timeout,
        )

        print(
            f"  → Generated {len(grid_map)}/{len(autodock_systems)} grids successfully"
        )

        # Update systems with grid paths, filter failures
        valid_systems: list[dict] = []
        for system in autodock_systems:
            grid_fld: str | None = grid_map.get(system["system_id"])
            if grid_fld:
                system["grid_fld"] = grid_fld
                valid_systems.append(system)
            else:
                errors_df = pd.concat(
                    [
                        errors_df,
                        pd.DataFrame(
                            [
                                {
                                    "system_id": system["system_id"],
                                    "error_message": "Grid generation failed",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        # Step 2: Execute docking in batches
        print(f"\n[2/3] Executing docking in batches of {autodock_batch_size}...")

        num_batches: int = (
            len(valid_systems) + autodock_batch_size - 1
        ) // autodock_batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * autodock_batch_size
            batch_end = min(batch_start + autodock_batch_size, len(valid_systems))
            batch: list[dict] = valid_systems[batch_start:batch_end]

            print(
                f"  → Batch {batch_idx + 1}/{num_batches}: docking systems {batch_start + 1}-{batch_end}"
            )

            docking_utils.execute_autodock_batch(
                batch, nrun=autodock_nrun, timeout=autodock_docking_timeout
            )

        # Step 3: Parse all results
        print(f"\n[3/3] Parsing {len(valid_systems)} result files...")

        parsed_count = 0
        for system in valid_systems:
            try:
                results_df: pd.DataFrame = docking_utils.get_docking_data(
                    system["docked_dlg"],
                    system["ligand_sdf"],
                    system["metal_coords"],
                    docking_type="autodock",
                )

                if system["redocking"]:
                    results_df["system_id"] = system["system_id"]
                else:
                    results_df["alt_system_id"] = system["alt_system_id"]
                results_df["ligand_id"] = system["ligand_id"]

                global_df = pd.concat([global_df, results_df])
                parsed_count += 1

            except Exception as err:
                print(f"  ✗ Error parsing {system['system_id']}: {err}")
                errors_df = pd.concat(
                    [
                        errors_df,
                        pd.DataFrame(
                            [
                                {
                                    "system_id": system["system_id"],
                                    "error_message": f"Result parsing failed: {err}",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        print(f"\n  → Successfully parsed {parsed_count}/{len(valid_systems)} systems")
        print("=" * 60)

    # Save results & errors
    if method == "AUTODOCK":
        output_csv = f"{data_path}/AUTODOCK_{docking_type}_id={id}_nrun={autodock_nrun}_boxExt={box_extension}.csv"
        errors_csv = f"{data_path}/errors_AUTODOCK_{docking_type}_id={id}_nrun={autodock_nrun}_boxExt={box_extension}.csv"
    elif method == "GPDOCK":
        output_csv = (
            f"{data_path}/GPDOCK_{docking_type}_id={id}_HOH={gpdock_num_waters}.csv"
        )
        errors_csv = f"{data_path}/errors_GPDOCK_{docking_type}_id={id}_HOH={gpdock_num_waters}.csv"
    elif method == "METALLODOCK":
        output_csv = f"{data_path}/METALLODOCK_{docking_type}_id={id}_min_method={metallodock_min_method}.csv"
        errors_csv = f"{data_path}/errors_METALLODOCK_{docking_type}_id={id}_min_method={metallodock_min_method}.csv"
    elif method == "GNINA":
        output_csv = f"{data_path}/{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_prepPH=7.4.csv"
        errors_csv = f"{data_path}/errors_{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_prepPH=7.4.csv"
    else:
        output_csv = f"{data_path}/{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_eRange={energy_range}.csv"
        errors_csv = f"{data_path}/errors_{method}_{docking_type}_id={id}_exhaust={exhaustiveness}_numModes={num_modes}_eRange={energy_range}.csv"

    if output_suffix:
        output_stem, output_ext = os.path.splitext(output_csv)
        errors_stem, errors_ext = os.path.splitext(errors_csv)
        output_csv = f"{output_stem}_{output_suffix}{output_ext}"
        errors_csv = f"{errors_stem}_{output_suffix}{errors_ext}"

    global_df.to_csv(output_csv, index=False)
    errors_df.to_csv(errors_csv, index=False)
    print("INFO: Docking process completed. Results & errors saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Docking.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=(
            "Split that should be docked: train, val, test, or unseen (the "
            "leakage-corrected subsets used for GPDOCK/MetalloDock; needs "
            "--unseen_map_path)."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["VINA", "SMINA", "GNINA", "AUTODOCK", "METALLODOCK", "GPDOCK"],
        help="The docking method to be used.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="1",
        help="Job ID used to identify results with otherwise identical settings.",
    )
    parser.add_argument(
        "--redocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="True = Re-Docking, False = Cross-Docking.",
    )
    parser.add_argument(
        "--add_hydrogen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize hydrogen placement with Reduce2, if possible.",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=32,
        help="Exhaustiveness for docking.",
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        default=100,
        help="Maximum number of poses to generate.",
    )
    parser.add_argument(
        "--energy_range",
        type=float,
        default=20.0,
        help="Max energy window in kcal/mol for Vina and Smina (unused by GNINA).",
    )
    parser.add_argument(
        "--box_extension",
        type=float,
        default=4.0,
        help="Extension of search space around provided ligand in Å.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_CSV,
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--representative_train_systems_path",
        type=str,
        default=CLUSTER_REPRESENTATIVES_CSV,
        help="Path to the representative systems CSV file for the training set, if applicable.",
    )
    parser.add_argument(
        "--systems_root",
        default=os.environ.get("SYSTEMS_ROOT", DEFAULT_SYSTEMS_ROOT),
        help=(
            "Directory containing the train_systems, val_systems, and test_systems "
            "folders (default: $SYSTEMS_ROOT, or repository/docking if unset)."
        ),
    )
    parser.add_argument(
        "--metal_ligand_data_path",
        default=None,
        help="Explicit path to the cached metal-coordinate JSON.",
    )
    parser.add_argument(
        "--autodock_nrun",
        type=int,
        default=100,
        help="Number of AutoDock LGA runs (default: 100).",
    )
    parser.add_argument(
        "--autodock_batch_size",
        type=int,
        default=8,
        help="AutoDock batch size for GPU processing (default: 8).",
    )
    parser.add_argument(
        "--autodock_spacing",
        type=float,
        default=0.375,
        help="Grid spacing for AutoDock in Angstroms (default: 0.375).",
    )
    parser.add_argument(
        "--autodock_max_points",
        type=int,
        default=255,
        help="Max points per dimension for AutoDock grid (default: 255).",
    )
    parser.add_argument(
        "--autodock_prep_timeout",
        type=int,
        default=3600,
        help="Maximum time in seconds to allow for AutoDock grid generation before aborting.",
    )
    parser.add_argument(
        "--autodock_docking_timeout",
        type=int,
        default=7200,
        help="Maximum time in seconds to allow for AutoDock (batch) docking before aborting.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for docking (default: 1).",
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Optional suffix for aggregate result and error CSV filenames.",
    )
    parser.add_argument(
        "--gpdock_num_waters",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Number of water molecules for GPDOCK docking (0-2, default: 0).",
    )
    parser.add_argument(
        "--gpdock_fe_oxidation_state",
        type=int,
        default=6,
        choices=[5, 6],
        help="Fe oxidation state for GPDOCK: 5=Fe2+, 6=Fe3+ (default: 6).",
    )
    parser.add_argument(
        "--gpdock_docking_timeout",
        type=int,
        default=90 * 60,
        help="Maximum time in seconds to allow for GPDOCK docking before aborting.",
    )
    parser.add_argument(
        "--metallodock_min_method",
        type=str,
        default="uff",
        choices=["uff", "openff2"],
        help="Force-field minimisation for MetalloDock: 'uff' (fast) or 'openff2' (more accurate). Default: uff.",
    )
    parser.add_argument(
        "--metallodock_batch_size",
        type=int,
        default=64,
        help="Batch size for MetalloDock model inference (default: 64).",
    )
    parser.add_argument(
        "--metallodock_pocket_extraction_radius",
        type=float,
        default=4.0,
        help="Radius in Angstroms of the ball used to encapsulate the Vina-style box around the ligand when extracting the binding pocket.",
    )
    parser.add_argument(
        "--unseen_cluster_csv_path",
        type=str,
        default=f"{CACHED_DATA_PATH}/split_clusters.csv",
        help="Path to the clustering CSV file (map systems to clusters).",
    )
    parser.add_argument(
        "--unseen_map_path",
        type=str,
        default=None,
        help="Path to a map pickle file that maps systems to be either in the training or unseen set.",
    )
    args = parser.parse_args()

    run_docking(
        split=args.split,
        method=args.method,
        id=args.id,
        redocking=args.redocking,
        add_hydrogen=args.add_hydrogen,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        box_extension=args.box_extension,
        dataset_path=args.dataset_path,
        representative_train_systems_path=args.representative_train_systems_path,
        systems_root=args.systems_root,
        metal_ligand_data_path=args.metal_ligand_data_path,
        autodock_nrun=args.autodock_nrun,
        autodock_batch_size=args.autodock_batch_size,
        autodock_spacing=args.autodock_spacing,
        autodock_max_points=args.autodock_max_points,
        autodock_prep_timeout=args.autodock_prep_timeout,
        autodock_docking_timeout=args.autodock_docking_timeout,
        num_workers=args.num_workers,
        output_suffix=args.output_suffix,
        gpdock_num_waters=args.gpdock_num_waters,
        gpdock_fe_oxidation_state=args.gpdock_fe_oxidation_state,
        gpdock_docking_timeout=args.gpdock_docking_timeout,
        metallodock_min_method=args.metallodock_min_method,
        metallodock_batch_size=args.metallodock_batch_size,
        metallodock_pocket_extraction_radius=args.metallodock_pocket_extraction_radius,
        unseen_cluster_csv_path=args.unseen_cluster_csv_path,
        unseen_map_path=args.unseen_map_path,
    )
