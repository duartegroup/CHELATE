"""
Description
-----------
This module contains functionality to perform a custom resplit of the PLINDER dataset.

Files Saved
-----------
- `PLINDER_LOCAL_STORAGE/index/annotation_table.parquet`: Updated annotation table with
  'has_good_apo' column. The untouched table is kept as `annotation_table_original.parquet`.
- `split_clusters.csv`: CSV file containing the system ids with the new split and cluster column.
- `train_cluster_representatives.csv`: CSV file containing one representative of each train cluster.

Functions
---------
- `resplit(plinder_df: pd.DataFrame, split_name: str) -> pd.DataFrame`:
    Perform custom resplit of the PLINDER dataset.
"""

import logging
import os
import shutil

import pandas as pd
from omegaconf import DictConfig
from params import PLINDER_LOCAL_STORAGE
from plinder.data.splits import get_default_config, split

# Seed for choosing train-cluster representatives, so the docked training subset is
# reproducible.
RANDOM_SEED: int = 42


def resplit(plinder_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Perform custom resplit of the PLINDER dataset. Changes to default configuration:
    - `num_test_representatives`: 5 (PLINDER_DEFAULT=2)
    - `min_test_cluster_size`: 3 (PLINDER_DEFAULT=5)
    - `min_test_leakage_count`: 20 (PLINDER_DEFAULT=30)
    - `min_val_cluster_size`: 5 (PLINDER_DEFAULT=30)
    - `num_val_representatives`: 5 (PLINDER_DEFAULT=3)
    - Add `has_good_apo==True` and `system_ligand_has_cofactor==False` as additional criteria for the test set.

    Parameters
    ----------
    plinder_df : pd.DataFrame
        DataFrame to be resplit, annotated with the presence of good apo structures.
    split_name : str
        Identifier for the resplit dataset. Used for saving the resplit dataset.

    Returns
    -------
    pd.DataFrame
        The resplit dataset with the new split column.

    Notes
    -----
    This will require a lot of memory (100GB+).
    """
    # Read the original annotation table. On a re-run, the live table already has the
    # extra column, so always start from the untouched backup made on the first run.
    index_dir = f"{PLINDER_LOCAL_STORAGE}/index"
    table_path = f"{index_dir}/annotation_table.parquet"
    original_path = f"{index_dir}/annotation_table_original.parquet"
    if not os.path.exists(original_path):
        shutil.copy2(table_path, original_path)
    parquet_annotation_table: pd.DataFrame = pd.read_parquet(original_path)

    # Add a new column 'has_good_apo' to the annotation table
    has_good_apo_set: set = set(plinder_df.loc[plinder_df["has_good_apo"], "system_id"])
    parquet_annotation_table["has_good_apo"] = parquet_annotation_table[
        "system_id"
    ].isin(has_good_apo_set)

    # Save the annotation table with the added column; the original stays backed up
    parquet_annotation_table.to_parquet(table_path)

    v2_config: DictConfig = get_default_config()
    v2_config.split.num_test_representatives = 5  # PLINDER_DEFAULT=2
    v2_config.split.min_test_cluster_size = 3  # PLINDER_DEFAULT=5
    v2_config.split.min_test_leakage_count = 20  # PLINDER_DEFAULT=30
    v2_config.split.min_val_cluster_size = 5  # PLINDER_DEFAULT=30
    v2_config.split.num_val_representatives = 5  # PLINDER_DEFAULT=3

    # Add has_good_apo and system_ligand_has_cofactor as additional criteria
    v2_config.split.test_additional_criteria.append(["has_good_apo", "==", "True"])
    v2_config.split.test_additional_criteria.append(
        ["system_ligand_has_cofactor", "==", "False"]
    )

    my_systems: set[str] = set(plinder_df["system_id"].unique())

    # resplit the dataset
    resplit_df = split(
        data_dir=PLINDER_LOCAL_STORAGE,
        cfg=v2_config,
        relpath=split_name,
        selected_systems=my_systems,
    )
    resplit_df.to_csv("cached_data/split_clusters.csv", index=False)

    # Save one representative of each train cluster
    representatives: dict = {}
    train_df = resplit_df[resplit_df["split"] == "train"]
    for cluster, group in train_df.groupby("cluster"):
        selected: pd.Series = group.sample(1, random_state=RANDOM_SEED).iloc[0]
        representatives[cluster] = selected["system_id"]

    representative_df: pd.DataFrame = pd.DataFrame(
        list(representatives.items()), columns=["cluster", "representative_system_id"]
    )
    representative_df.to_csv(
        "cached_data/train_cluster_representatives.csv", index=False
    )

    # Overwriting split column
    split_dict: dict[str, str] = dict(
        zip(resplit_df["system_id"], resplit_df["split"], strict=False)
    )
    plinder_df = plinder_df[plinder_df["system_id"].isin(split_dict.keys())]
    plinder_df["split"] = plinder_df["system_id"].map(split_dict)

    plinder_df["combined_id"] = (
        plinder_df["system_id"] + "__" + plinder_df["ligand_asym_id"]
    )
    plinder_df = plinder_df.drop_duplicates(subset="combined_id")
    plinder_df = plinder_df.drop(columns=["combined_id"])

    # print some stats about the resplit dataset
    logging.info("Finished resplitting.")
    logging.info("Dataset statistics after resplitting:")
    print(
        plinder_df.groupby(["split", "has_binding_affinity"])
        .size()
        .unstack(fill_value=0)
    )
    print("-" * 50)
    print(
        plinder_df.groupby(["split", "ligand_is_cofactor"]).size().unstack(fill_value=0)
    )
    print("-" * 50)
    print(plinder_df.groupby(["split", "has_good_apo"]).size().unstack(fill_value=0))
    print("-" * 50)

    return plinder_df
