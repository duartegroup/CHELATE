"""
Description
-----------
This script generates a comprehensive metalloprotein dataset by querying and processing
data from the PLINDER database. The dataset includes information about protein-ligand
interactions in metalloprotein systems and applies several filtering and processing
steps to ensure data quality and relevance.

Usage
-----
To generate the dataset, run the script as follows:
```bash
python generate_dataset.py --run_resplit --ligand_metal_min_threshold 4.0 \
                           --good_apo_threshold 70.0 \
                           --save_superimposed_structures \
                           --max_ligand_mw 750.0 \
                           --plot_split
```

To explicitly disable a boolean option, use the `--no-<option>` flag, e.g.:
```bash
python generate_dataset.py --no-plot_split --no-save_superimposed_structures
```
Warning
-------
Resplitting the dataset can take a long time.

Functions
---------
- `get_dataset(ligand_metal_min_threshold: float = 4.0, has_good_apo_threshold: float = 70.0, resplit: bool = True, split_name: str = "v2", save_superimposed_structures: bool = True, plot_split: bool = True,) -> pd.DataFrame`:
    Generates the full metalloprotein dataset.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import generate_metal_ligand_data
import pandas as pd
import plot_utils
import resplit_plinder
import superimpose_structures
from params import COLS_OF_INTEREST
from plinder.core.scores import query_index


def get_dataset(
    ligand_metal_min_threshold: float = 4.0,
    max_ligand_mw: float = 750.0,
    has_good_apo_threshold: float = 70.0,
    resplit: bool = True,
    split_name: str = "v2",
    save_superimposed_structures: bool = True,
    plot_split: bool = True,
) -> pd.DataFrame:
    """
    Generate the full metalloprotein dataset.

    The function prints summary statistics at each stage of processing, including:
    - The number of entries in each data split (train, validation, test, and removed).
    - The total number of entries.
    - The number of unique PDB IDs in the dataset.

    Parameters
    ----------
    ligand_metal_min_threshold : float, default: 4.0
        Threshold in Angstrom for filtering based on ligand metal distance. Must be <= 4.0,
        because PLINDER only includes metals within 4.0 A of the ligand.
    has_good_apo_threshold : float, default: 70.0
        Minimum `pocket_lddt`/`pocket_fident` value for good apo structures.
    resplit : bool, default: True
        Should data be resplit with PLINDER.
    split_name : str, default: "v2"
        Identifier for the resplit dataset. Used for saving the resplit dataset.
    save_superimposed_structures : bool, default: True
        Should superimposed structures be saved.
    plot_split : bool, default: True
        Should final dataset be plotted. Plots are saved in the `split_plots` directory.
    max_ligand_mw : float, default: 750.0
        Maximum ligand molecular weight in Dalton (g/mol). Ligands with a higher
        molecular weight are filtered out.

    Returns
    -------
    pandas.DataFrame
        A processed dataframe containing information about protein-ligand
        interactions in metalloprotein systems.
    """
    resplit_tag: str = "_resplit" if resplit else ""
    ligand_metal_min_threshold_tag: str = str(ligand_metal_min_threshold)
    has_good_apo_threshold_tag: str = str(has_good_apo_threshold)
    max_ligand_mw_tag: str = str(max_ligand_mw)
    identifier: str = f"distTresh={ligand_metal_min_threshold_tag}_mwThresh={max_ligand_mw_tag}_goodApoThresh={has_good_apo_threshold_tag}{resplit_tag}"
    dataset_name: str = f"final_dataset_{identifier}"

    plinder_df: pd.DataFrame = query_index(
        columns=COLS_OF_INTEREST,
        splits=[
            "train",
            "val",
            "test",
            "removed",
        ],
    )

    logging.info("Raw PLINDER data:")
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    def check_overlap(system_id: str) -> bool:
        """
        Helper function to check if protein chains and ligand chains of a PLINDER system intersect.
        Some PLINDER systems do not separate the ligand and protein chains properly.

        Parameters
        ----------
        system_id : str
            PLINDER system id to be checked

        Returns
        ----------
        bool
            Boolean if ligand and protein chains overlap
        """
        _, _, protein_chains, ligand_chains = system_id.split("__")
        protein_chains: list[str] = protein_chains.split("_")
        ligand_chains: list[str] = ligand_chains.split("_")
        return len(set(protein_chains) & set(ligand_chains)) > 0

    # Dropping systems with overlapping ligand and receptor
    plinder_df = plinder_df[~plinder_df["system_id"].map(check_overlap)]
    logging.info("After systems with overlapping ligand and receptor:")
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    # Filtering systems with ligand_weight > max_ligand_mw
    plinder_df = plinder_df[plinder_df["ligand_molecular_weight"] <= max_ligand_mw]
    logging.info(f"After removing systems with ligand > {max_ligand_mw} Dalton:")
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    # Dropping rows with improper ligands (ions or artifacts)
    plinder_df = plinder_df[plinder_df["ligand_is_proper"]]

    logging.info("After filtering out improper ligands:")
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    # metal_ligand_data.json also contains the minimum distances between ligands and each metal ion
    # in the PLINDER structure. Used for filtering out non-metalloproteins and systems with
    # no metal within a given threshold to the ligand.
    if not os.path.exists(f"cached_data/metal_ligand_data_{identifier}.json"):
        logging.info(
            f"cached_data/metal_ligand_data_{identifier}.json not found, running query now."
        )
        system_ids = set(plinder_df["system_id"])
        generate_metal_ligand_data.run_query(
            system_ids=system_ids, identifier=identifier
        )

    with open(f"cached_data/metal_ligand_data_{identifier}.json") as f:
        metal_ligand_data: dict[
            str,
            dict[
                str,
                float
                | list[list[str, tuple[float, float, float]]]
                | list[
                    tuple[
                        tuple[str, tuple[float, float, float]],
                        str,
                        list[tuple[int, tuple[float, float, float]]],
                    ]
                ],
            ],
        ] = json.load(f)

    plinder_df["has_metal"] = False
    plinder_df["min_distance"] = None
    for i, row in plinder_df.iterrows():
        system_id: str = row["system_id"]
        if system_id in metal_ligand_data:
            # Check if the system has metals
            if (
                metal_ligand_data[system_id]["metals"]
                and len(metal_ligand_data[system_id]["metals"]) > 0
            ):
                plinder_df.at[i, "has_metal"] = True
                # Set the minimum distance if available. Match the ligand file name
                # exactly: a substring test on the asym id also matched e.g. "A" in
                # "1.BA.sdf" and ignored the ligand instance.
                ligand_file = f"{row['ligand_instance']}.{row['ligand_asym_id']}.sdf"
                for lig_path, dist in metal_ligand_data[system_id]["min_distances"]:
                    if Path(lig_path).name == ligand_file:
                        plinder_df.at[i, "min_distance"] = dist
                        break
            else:
                plinder_df.at[i, "has_metal"] = False
        else:
            logging.warning(f"Metal search incomplete for {system_id}.")

    # PLINDER's system_has_binding_affinity is True if ANY ligand in that system has a binding affinity,
    # so we create a new column for filtering if this particular ligand has a binding affinity.
    plinder_df["has_binding_affinity"] = plinder_df["ligand_binding_affinity"] >= 1e-7

    # Filtering non-metalloproteins
    plinder_df = plinder_df[plinder_df["has_metal"]]
    logging.info("After removing non-metalloprotein systems:")
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    # Filtering metalloproteins with no metal within given threshold to ligand
    plinder_df = plinder_df[plinder_df["min_distance"] <= ligand_metal_min_threshold]
    logging.info(
        f"After removing all systems with no ligand within {ligand_metal_min_threshold} A to a metal:"
    )
    print(plinder_df["split"].value_counts())
    print(
        f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
    )
    print("-" * 50)

    # Link best alternative systems
    logging.info("Getting best alternative structures.")
    plinder_df = superimpose_structures.run_query(
        plinder_df, has_good_apo_threshold, save_superimposed_structures
    )

    if resplit:
        if os.path.exists(f"cached_data/{dataset_name}.csv"):
            # keep_default_na=False: a ligand chain "NA" is an identifier, not missing.
            plinder_df: pd.DataFrame = pd.read_csv(
                f"cached_data/{dataset_name}.csv", keep_default_na=False, na_values=[""]
            )
            logging.info("Loaded resplit dataset.")
        else:
            logging.info("Dataset not re-split yet. Running re-split now.")
            logging.info("This might require a lot of memory (100GB+).")
            plinder_df: pd.DataFrame = resplit_plinder.resplit(plinder_df, split_name)

        logging.info("After re-splitting:")
        print(plinder_df["split"].value_counts())
        print(
            f"Unique PDBs: {len(set(plinder_df['entry_pdb_id']))}, Total rows {len(plinder_df)}"
        )
        print("-" * 50)

    if plot_split:
        logging.info("Creating figures about final dataset.")
        plots_save_path = "./split_plots/"
        plot_utils.plot_plinder_split(split_name)
        plot_utils.plot_histogram_prop(
            plinder_df,
            "ligand_molecular_weight",
            "Ligand Molecular Weight (Da)",
            plots_save_path,
        )
        plot_utils.plot_histogram_prop(
            plinder_df,
            "ligand_num_heavy_atoms",
            "Ligand Number of Heavy Atoms (-)",
            plots_save_path,
        )
        plot_utils.plot_histogram_prop(
            plinder_df,
            "ligand_num_rot_bonds",
            "Ligand Number Rotatable Bonds (-)",
            plots_save_path,
        )
        plot_utils.plot_metal_distribution(
            f"cached_data/metal_ligand_data_{identifier}.json",
            plinder_df,
            plots_save_path,
        )

    # save final dataset
    logging.info("Saving resplit dataset.")
    os.makedirs("cached_data", exist_ok=True)
    plinder_df.to_csv(f"cached_data/{dataset_name}.csv", index=False)

    return plinder_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive metalloprotein dataset by querying and processing data from the PLINDER database."
    )
    parser.add_argument(
        "--run_resplit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Should data be resplit with PLINDER",
    )
    parser.add_argument(
        "--ligand_metal_min_threshold",
        type=float,
        default=4.0,
        help="Threshold in Angstrom for filtering based on ligand metal distance",
    )
    parser.add_argument(
        "--max_ligand_mw",
        type=float,
        default=750.0,
        help="Maximum ligand molecular weight in Dalton (g/mol)",
    )
    parser.add_argument(
        "--good_apo_threshold",
        type=float,
        default=70.0,
        help="Minimum pocket-lddt/pocket_fident value for good apo structures, [0, 100]",
    )
    parser.add_argument(
        "--save_superimposed_structures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Should superimposed structures be saved",
    )
    parser.add_argument(
        "--plot_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Should final dataset be plotted",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    plinder_df = get_dataset(
        ligand_metal_min_threshold=args.ligand_metal_min_threshold,
        max_ligand_mw=args.max_ligand_mw,
        has_good_apo_threshold=args.good_apo_threshold,
        resplit=args.run_resplit,
        save_superimposed_structures=args.save_superimposed_structures,
        plot_split=args.plot_split,
    )
