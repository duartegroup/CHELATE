"""
Description
-----------
This module contains functions for finding alternative protein structures using PLINDER's `multi_query_protein_similarity` function.

Functions
---------
- `process_system(sys_id: str) -> pd.DataFrame | None`:
    Uses PLINDER's `multi_query_protein_similarity` to search for alternative structures for the given system.

- `save_results(results: list[pd.DataFrame], batch_number: int) -> None`:
    Saves a batch of results to a CSV file.

- `run_query(system_ids: list[str]) -> None`:
    Uses PLINDER's `multi_query_protein_similarity` to search for alternative structures for the given system IDs.
"""

import logging
import os
from glob import glob

import pandas as pd
from plinder.core.scores.protein import multi_query_protein_similarity


def process_system(sys_id: str) -> pd.DataFrame | None:
    """
    Uses PLINDER's `multi_query_protein_similarity` to search for alternative structures
    for the given system.

    Parameters
    ----------
    sys_id : str
        System ID to be processed.

    Returns
    -------
    pd.DataFrame | None
        DataFrame containing alternative structures for the given system ID
        or None if the query results in an error.
    """
    try:
        filter_criteria: dict[str, int] = {
            "pocket_fident": 0,
            "protein_fident_weighted_sum": 0,
            "protein_fident_qcov_weighted_sum": 0,
            "pocket_lddt": 0,
        }

        # Get predicted structures
        res_pred = multi_query_protein_similarity(
            system_id=sys_id, search_db="pred", filter_criteria=filter_criteria
        )
        res_pred["type"] = "pred"

        # Get apo structures
        res_apo = multi_query_protein_similarity(
            system_id=sys_id, search_db="apo", filter_criteria=filter_criteria
        )
        res_apo["type"] = "apo"

        # Combine results
        return pd.concat([res_pred, res_apo])

    except Exception as e:
        logging.error(f"Error processing system {sys_id}: {e!s}")
        return None


def save_results(results: list[pd.DataFrame], batch_number: int) -> None:
    """
    Saves a batch of results to a CSV file as `alt_structures_batch_<batch_number>.csv`.

    Parameters
    ----------
    results : list[pd.DataFrame]
        List of DataFrames to be saved.

    batch_number : int
        Batch number to be used in the file name.

    Returns
    -------
    None
    """
    combined_results: pd.DataFrame = pd.concat(results)
    combined_results.to_csv(f"./alt_structures_batch_{batch_number}.csv")


def get_cached_systems() -> tuple[set[str], int]:
    """
    Retrieves the set of system IDs that have already been processed and cached. Also
    returns the number of cached batch files found.

    Returns
    -------
    set[str]
        Set of system IDs that have been processed.
    int
        Number of cached batch files found. This is the batch number.
    """
    cached_batch_files: list[str] = glob("./alt_structures_batch_*.csv")
    if cached_batch_files:
        cached_systems: set[str] = set()
        for file in cached_batch_files:
            df: pd.DataFrame = pd.read_csv(file)
            cached_systems.update(df["query_system"].unique())

        logging.info(
            f"Found {len(cached_systems)} cached systems that have already been linked to alternative structures."
        )
        return cached_systems, len(cached_batch_files)
    else:
        return set(), 0


def run_query(system_ids: list[str]) -> None:
    """
    Uses PLINDER's `multi_query_protein_similarity` to search for alternative structures for
    the given system IDs.

    Since pd.DataFrame sizes can get large quite quickly, system IDs are processed in
    batches, and merged into a final CSV file at the end. This also avoids the need to
    re-run everything in case something goes wrong. For that reason, caching is also
    implemented to skip already processed systems.

    Saves the final results in `cached_data/alt_structures_final.csv` and systems without
    alternative structures or that threw an error when queried in
    `cached_data/systems_without_alt_structures_final.csv`.

    Parameters
    ----------
    system_ids : list[str]
        List of system ID to be processed.

    Returns
    -------
    None
    """

    logging.info("Starting query for alternative structures.")
    logging.info(f"Systems to process: {len(system_ids)}.")
    batch_size: int = 1000
    current_batch: list[pd.DataFrame] = []
    systems_without_alt: list[str] = []
    cached_systems: set[str]
    batch_number: int
    cached_systems, batch_number = get_cached_systems()

    for i, sys_id in enumerate(system_ids, 1):
        if sys_id in cached_systems:
            continue

        result: pd.DataFrame | None = process_system(sys_id)

        if result is None or len(result) == 0:
            systems_without_alt.append(sys_id)
        else:
            current_batch.append(result)

        if len(current_batch) >= batch_size:
            save_results(current_batch, batch_number)
            current_batch: list[pd.DataFrame] = []
            batch_number += 1

        if i % 100 == 0:
            logging.info(f"Processed {i}/{len(system_ids)} systems.")

    if current_batch:
        save_results(current_batch, batch_number)
        batch_number += 1

    # Combine all batches into one final CSV and remove batch files
    batch_files: list[str] = [
        f"./alt_structures_batch_{i}.csv" for i in range(batch_number)
    ]
    combined_all_results: pd.DataFrame = pd.concat(
        [pd.read_csv(file) for file in batch_files]
    )

    # Save the combined results into one final CSV file
    os.makedirs("cached_data", exist_ok=True)
    combined_all_results.to_csv("cached_data/alt_structures_final.csv", index=False)

    # Remove the batch files after combining
    for file in batch_files:
        os.remove(file)

    # Save systems without alternative structures or that threw an error when queried
    pd.DataFrame(systems_without_alt, columns=["system_id"]).to_csv(
        "cached_data/systems_without_alt_structures_final.csv"
    )
