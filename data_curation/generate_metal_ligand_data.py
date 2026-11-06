"""
Description
-----------
This script processes a list of system IDs to compute:
1. The minimum distances between ligands and metals in each system
2. The metal atoms and their coordinates in each system

The results are merged into a single dictionary with the format:
system_id: {
    'min_distances': list of tuples (ligand_file, distance),
    'metals': list of tuples (metal_type, coordinates),
    'hetero_atoms': list of (metal, ligand_file, hetero atoms within 5 A)
}

The results are saved periodically and at the end of the processing.

Usage
-----
Run this script with a path to a CSV file containing system IDs in a 'system_id' column.
The results will be saved in a JSON file.

Example:
    python generate_metal_ligand_data.py path/to/systems.csv my_identifier

Functions
---------
- process_system(system_id: str) -> tuple: Processes a single system and returns its results.
- save_progress(data_safe: ThreadSafeDict) -> None: Saves the current progress to a file.
- run_query(system_ids: list, identifier: str) -> None: Processes a list of system IDs using a
  thread pool and saves the results to cached_data/metal_ligand_data_{identifier}.json.
"""

import argparse
import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import dataset_utils
import pandas as pd
from dataset_utils import (
    get_hetero_atoms_within_cutoff,
    get_ligand_metal_min_distance,
    get_metals_from_ligand_dir,
)
from plinder.core import PlinderSystem
from tqdm import tqdm


class ThreadSafeDict:
    """
    A thread-safe dictionary that can be accessed by multiple threads.

    Attributes
    ----------
    dict : dict
        The dictionary to store data.

    lock : threading.Lock
        A lock to ensure thread safety.

    Methods
    -------
    __setitem__(key: Any, value: Any) -> None
        Set an item in the dictionary.

    __getitem__(key: Any) -> Any
        Get an item from the dictionary.

    get_dict() -> dict
        Get the dictionary.
    """

    def __init__(self, initial_dict=None):
        self.dict: dict = initial_dict or {}
        self.lock: threading.Lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value

    def __getitem__(self, key):
        with self.lock:
            return self.dict[key]

    def get_dict(self):
        with self.lock:
            return dict(self.dict)


def process_system(system_id: str) -> tuple[str, dict | None, str | None]:
    """
    Process a single system and return the system ID, minimum distances, and metal information.

    Parameters
    ----------
    system_id : str
        The system ID to process.

    Returns
    -------
    tuple[str, dict | None, str | None]:
        A tuple containing:
        - The system ID (str)
        - A dictionary mapping the system ID to:
            - 'min_distance': A list of tuples (ligand_id, distance) representing the minimum distances
                between ligands and metals.
            - 'metals': A list of metals, where each metal is represented as a tuple containing:
                - 'type': The metal type (str)
                - 'coordinates': The coordinates of the metal as a tuple of floats [x, y, z].
        - An error message (str) if an error occurred, or None if no error occurred.
    """
    try:
        plinder_system: PlinderSystem = PlinderSystem(system_id=system_id)
        receptor_pdb_path: str = plinder_system.receptor_pdb
        ligands_path: str = (
            receptor_pdb_path.split(".pdb")[0].split("receptor")[0] + "ligand_files/"
        )

        # Get metal atoms and their coordinates
        metals: list[tuple[str, tuple[float, float, float]]] = (
            get_metals_from_ligand_dir(ligands_path)
        )
        has_metals: bool = len(metals) > 0

        if has_metals:
            # Get minimum distances between ligands and metals
            min_distances: list[tuple[str, float]] = get_ligand_metal_min_distance(
                ligands_path, metals
            )

            # Get hetero atoms within cutoff
            hetero_atoms: list[
                tuple[
                    tuple[str, tuple[float, float, float]],
                    str,
                    list[tuple[int, tuple[float, float, float]]],
                ]
            ] = get_hetero_atoms_within_cutoff(ligands_path=ligands_path, metals=metals)
        else:
            min_distances: list = []
            hetero_atoms: list = []

        # Format result in the requested structure
        result = {
            "min_distances": min_distances,
            "metals": metals,
            "hetero_atoms": hetero_atoms,
        }

        return system_id, result, None
    except Exception as e:
        return system_id, None, str(e)


def save_progress(data_safe: ThreadSafeDict) -> None:
    """
    Save the current progress to a file.

    Parameters
    ----------
    data_safe : ThreadSafeDict
        The thread-safe dictionary containing the current progress.

    Returns
    -------
    None
    """
    current_dict: dict = data_safe.get_dict()
    with open("metal_ligand_data_progress.json", "w") as f:
        json.dump(current_dict, f, indent=2)


def run_query(system_ids: list[str], identifier: str) -> None:
    """
    Process a list of system IDs using a thread pool and save the results periodically.

    Parameters
    ----------
    system_ids : list[str]
        A list of system IDs to process.
    identifier : str
        An identifier which will be used to name the output JSON file.

    Returns
    -------
    None
    """
    data_safe: ThreadSafeDict = ThreadSafeDict()
    processed_count: int = 0
    save_lock: threading.Lock = threading.Lock()
    num_threads = multiprocessing.cpu_count() + 4
    print(f"Number of threads {num_threads}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_system: dict = {
            executor.submit(process_system, system_id): system_id
            for system_id in system_ids
        }

        # Track progress with tqdm
        with tqdm(
            total=len(system_ids), desc="Processing systems", unit="system"
        ) as pbar:
            for future in as_completed(future_to_system):
                system_id: str
                result: dict | None
                error: str | None
                system_id, result, error = future.result()

                if error is None:
                    data_safe[system_id] = result
                    processed_count += 1

                    # Save progress periodically
                    if processed_count % 100 == 0:
                        with save_lock:
                            save_progress(data_safe)
                else:
                    print(f"Error processing system {system_id}: {error}")

                pbar.update(1)

    # Save final results
    final_dict: dict[str, dict] = data_safe.get_dict()
    os.makedirs("cached_data", exist_ok=True)
    with open(f"cached_data/metal_ligand_data_{identifier}.json", "w") as f:
        json.dump(final_dict, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process system IDs to compute metal-ligand distances, metal types and their coordinates."
    )
    parser.add_argument(
        "datapath",
        type=str,
        help="Path to CSV file containing system IDs in a 'system_id' column",
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Output name: cached_data/metal_ligand_data_{identifier}.json",
    )
    parser.add_argument(
        "--paper_metal_rule",
        action="store_true",
        help=(
            "Reproduce the curation used for the published dataset: accept every "
            "positively charged single-atom ligand as a metal, including NH4+. By "
            "default, a metal ligand must be a metal element."
        ),
    )

    args = parser.parse_args()
    if args.paper_metal_rule:
        dataset_utils.REQUIRE_METAL_ELEMENT = False

    # Read system IDs from CSV
    df = pd.read_csv(args.datapath)
    system_ids = df["system_id"].tolist()
    run_query(system_ids, args.identifier)
