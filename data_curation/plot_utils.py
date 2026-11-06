"""
Description
-----------
This module contains utility functions for plotting dataset property distributions and metal occurrences.

Functions
---------
- `plot_histogram_prop(plinder_df: pd.DataFrame, column: str, name: str, save_path: str, density_threshold: float = 0.001) -> None`:
    Plot a property distribution for train, and validation sets. Property must be numerical. The plot is saved in the specified directory.
    Bins with density below threshold are excluded.

- `plot_plinder_split(split_name: str) -> None`:
    Wrapper function for SplitPropertiesPlotter.from_files() from PLINDER. Plots are saved in the 'split_plots' directory.

- `get_metal_counts(ligand_metals_dict: dict[str, list[tuple[str, tuple[float, float, float]]]], system_ids: list[str] = None) -> Counter`:
    Counts the number of occurrences of each metal in the provided set of systems. Multiple occurrences of the same metal in a single system are counted only once.

- `plot_metal_distribution(ligand_metals_dict_json_path: str, plinder_df: pd.DataFrame, save_path: str, min_metal_cnt: int = 30) -> None`:
    Plot the distribution of metals for train and test sets, and a comparison between train and test sets.
"""

import contextlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from params import PLINDER_LOCAL_STORAGE, SPLIT_COLORS
from plinder.core.split.plot import SplitPropertiesPlotter


def plot_histogram_prop(
    plinder_df: pd.DataFrame,
    column: str,
    name: str,
    save_path: str,
    density_threshold: float = 0.001,
) -> None:
    """
    Plot a property distribution for train, validation, and test sets. Property must be numerical.
    The plot is saved in the specified directory. Bins with density below threshold are excluded.

    Parameters
    ----------
    plinder_df : pd.DataFrame
        DataFrame containing the property (column), system IDs, and split information.
    column : str
        Column name to plot
    name : str
        Display name for the property
    save_path : str
        Path where the plots will be saved
    density_threshold : float, optional
        Minimum density threshold to include in plot (default: 0.01)

    Returns
    -------
    None
    """
    os.makedirs(save_path, exist_ok=True)

    if column not in plinder_df.columns:
        print(f"WARNING: {column} column not found, aborting plotting")
        return

    # Separate data by split
    split_dict: dict[str, str] = dict(
        zip(plinder_df["system_id"], plinder_df["split"], strict=False)
    )
    train_df: pd.Series = plinder_df[
        plinder_df["system_id"].isin(
            {system_id for system_id, split in split_dict.items() if split == "train"}
        )
    ][column]
    validation_df: pd.Series = plinder_df[
        plinder_df["system_id"].isin(
            {
                system_id
                for system_id, split in split_dict.items()
                if split == "validation"
            }
        )
    ][column]
    test_df: pd.Series = plinder_df[
        plinder_df["system_id"].isin(
            {system_id for system_id, split in split_dict.items() if split == "test"}
        )
    ][column]

    # Create histogram bins
    all_df: pd.Series = pd.concat([train_df, validation_df, test_df])

    # First calculate histogram to find density threshold
    counts, bin_edges = np.histogram(all_df, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the range where density is above threshold
    mask = counts >= density_threshold
    if mask.any():
        min_x = bin_centers[mask][0]
        max_x = bin_centers[mask][-1]
        # Add a small buffer
        range_buffer = (max_x - min_x) * 0.1
        plot_range = (min_x - range_buffer, max_x + range_buffer)
    else:
        plot_range = (all_df.min(), all_df.max())

    # Create new bins within the filtered range
    bins: np.ndarray = np.linspace(plot_range[0], plot_range[1], 30)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(
        train_df,
        bins=bins,
        alpha=0.5,
        label=f"Train (N={len(train_df)})",
        color=SPLIT_COLORS["train"],
        density=True,
    )
    plt.hist(
        validation_df,
        bins=bins,
        alpha=0.5,
        label=f"Validation (N={len(validation_df)})",
        color=SPLIT_COLORS["val"],
        density=True,
    )
    plt.hist(
        test_df,
        bins=bins,
        alpha=0.5,
        label=f"Test (N={len(test_df)})",
        color=SPLIT_COLORS["test"],
        density=True,
    )

    # Customize plot
    plt.xlabel(name, fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"{save_path}/{name}_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_plinder_split(split_name: str, save_path: str = ".") -> None:
    """
    Wrapper function for SplitPropertiesPlotter.from_files() from PLINDER. Plots are
    saved in a 'split_plots' directory inside ``save_path``.

    Parameters
    ----------
    split_name : str
        The name of the split, used to locate the corresponding split file.
    save_path : str, optional
        Directory in which PLINDER creates its 'split_plots' folder. Defaults to the
        current working directory.

    Returns
    -------
    None
    """
    storage = Path(PLINDER_LOCAL_STORAGE).resolve()
    os.makedirs(os.path.join(save_path, "split_plots"), exist_ok=True)

    # SplitPropertiesPlotter always writes to "split_plots" in the working directory,
    # so run it from inside save_path.
    with contextlib.chdir(save_path):
        SplitPropertiesPlotter.from_files(
            data_dir=storage,
            split_file=storage / "splits" / f"split_{split_name}.parquet",
        )


def get_metal_counts(
    ligand_metals_dict: dict[str, list[tuple[str, tuple[float, float, float]]]],
    system_ids: list[str] | None = None,
) -> Counter:
    """
    Counts the number of occurrences of each metal in the provided set of systems.
    Multiple occurrences of the same metal in a single system are counted only once.

    Parameters
    ----------
    ligand_metals_dict : dict[str, list[tuple[str, tuple[float, float, float]]]]
        Dictionary with system IDs as keys and list of contained metal and their coordinates as values.
    system_ids : list[str] | None, default: None
        Optional list of system_ids to filter by.

    Returns
    -------
    Counter
        Counter object with metal counts
    """
    metal_counts: Counter = Counter()

    for system_id, metals in ligand_metals_dict.items():
        if system_ids is None or system_id in system_ids:
            # Get unique metals in this system
            unique_metals: set = {metal[0] for metal in metals}
            for metal in unique_metals:
                metal_counts[metal] += 1

    return metal_counts


def plot_metal_distribution(
    metal_ligand_data_json_path: str,
    plinder_df: pd.DataFrame,
    save_path: str,
    min_metal_cnt: int = 30,
) -> None:
    """
    Plot the distribution of metals for train and test sets, and a comparison between train
    and test sets. Plots are saved in the specified directory.

    Parameters
    ----------
    metal_ligand_data_json_path : str
        Path to a JSON file containing a dictionary with system IDs as keys and dictionaries with
        'min_distance' and 'metals' as values.
    plinder_df : pd.DataFrame
        DataFrame containing information on which systems are in the train and test sets.
    save_path : str
        Path where the plots will be saved.
    min_metal_cnt : int, default: 30
        Minimum count of a metal to be included in the plot of overall metal distribution.

    Returns
    -------
    None
    """
    os.makedirs(save_path, exist_ok=True)

    try:
        with open(metal_ligand_data_json_path) as f:
            metal_ligand_data: dict[
                str, dict[str, float | list[list[str, tuple[float, float, float]]]]
            ] = json.load(f)
    except FileNotFoundError:
        logging.error(
            f"Can't plot metal distribution: {metal_ligand_data_json_path} not found."
        )
        return

    # Extract just the metals data from the merged format
    ligand_metals_dict: dict[str, list[list[str, tuple[float, float, float]]]] = {
        system_id: data["metals"] for system_id, data in metal_ligand_data.items()
    }

    split_dict: dict[str, str] = dict(
        zip(plinder_df["system_id"], plinder_df["split"], strict=False)
    )

    # 1. Overall metal distribution (sorted)
    plt.figure(figsize=(12, 6))
    metal_counts: Counter = get_metal_counts(ligand_metals_dict)
    # Sort by count
    metals: tuple[str, ...]
    counts: tuple[int, ...]
    metals, counts = zip(
        *sorted(metal_counts.items(), key=lambda x: x[1], reverse=True), strict=False
    )

    # filter metals with counts < min_metal_cnt
    metal_list: list[str] = [
        metal
        for metal, count in zip(metals, counts, strict=False)
        if count > min_metal_cnt
    ]
    count_list: list[int] = [count for count in counts if count > min_metal_cnt]

    plt.bar(metal_list, count_list)
    plt.title(
        f"Distribution of Metals Across All Systems (filtered for counts > {min_metal_cnt})"
    )
    plt.xlabel("Metal")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_path}/metal_distribution_all.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Test set metal distribution (sorted)
    test_systems: list[str] = list(
        {system_id for system_id, split in split_dict.items() if split == "test"}
    )
    test_metal_counts: Counter = get_metal_counts(ligand_metals_dict, test_systems)
    # Sort by count
    test_metals: tuple[str, ...]
    test_counts: tuple[int, ...]
    test_metals, test_counts = zip(
        *sorted(test_metal_counts.items(), key=lambda x: x[1], reverse=True),
        strict=False,
    )

    plt.figure(figsize=(12, 6))
    plt.bar(test_metals, test_counts)
    # plt.title("Distribution of Metals in Test Set")
    plt.xlabel("Metal")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"{save_path}/metal_distribution_test.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()
    # 3. Train vs Test comparison (relative counts, sorted by total counts)
    train_systems: list[str] = list(
        {system_id for system_id, split in split_dict.items() if split == "train"}
    )
    train_metal_counts: Counter = get_metal_counts(ligand_metals_dict, train_systems)

    # Calculate total counts
    total_train: int = sum(train_metal_counts.values())
    total_test: int = sum(test_metal_counts.values())

    # Get combined counts for sorting
    combined_counts: dict[str, int] = {
        metal: (train_metal_counts.get(metal, 0) + test_metal_counts.get(metal, 0))
        for metal in set(train_metal_counts.keys()) | set(test_metal_counts.keys())
    }

    # Sort metals by combined counts
    all_metals: list[str] = sorted(
        combined_counts.keys(), key=lambda x: combined_counts[x], reverse=True
    )

    # Select top N metals (default N=10)
    N: int = 10
    top_metals: list[str] = all_metals[:N]

    # Calculate relative frequencies in percentage for top metals
    train_rel_freqs: list[float] = [
        train_metal_counts.get(metal, 0) / total_train * 100 for metal in top_metals
    ]
    test_rel_freqs: list[float] = [
        test_metal_counts.get(metal, 0) / total_test * 100 for metal in top_metals
    ]

    # Create grouped bar chart
    plt.figure(figsize=(12, 6))
    x: range = range(len(top_metals))
    width: float = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        train_rel_freqs,
        width,
        label="Train",
        color="#FF999A",
    )
    plt.bar(
        [i + width / 2 for i in x],
        test_rel_freqs,
        width,
        label="Test",
        color="#74BAFF",
    )

    plt.xlabel("Metal", fontsize=16)
    plt.ylabel("Relative Frequency (%)", fontsize=16)
    plt.xticks(x, top_metals, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    # Add hlines but disable the full grid
    ax = plt.gca()  # Get the current axes
    plt.grid(False)
    ax.yaxis.grid(
        True, linestyle="--", linewidth=0.5, alpha=0.7
    )  # Enable horizontal grid lines
    ax.xaxis.grid(False)  # Disable vertical grid lines

    plt.tight_layout()

    plt.savefig(
        f"{save_path}/metal_distribution_train_test_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
