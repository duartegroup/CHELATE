"""
Description
-----------
This script selects up to N randomly chosen systems from every train cluster. The
selected systems form the training subset that is docked and featurised; the paper
docks "a maximum of 5 randomly selected complexes per training cluster".

The clusters come from `split_clusters.csv`, which `resplit_plinder.py` writes. Only
systems with `split == "train"` are considered. If a curated dataset CSV is given, only
train systems that are also present in that dataset are sampled. Clusters with N or fewer
systems are taken in full; larger clusters are sampled without replacement using a fixed
seed, so the output is reproducible for a given input.

Reproducibility of the shipped file
-----------------------------------
`cached_data/cluster_representatives_5.csv` (3,755 systems from 1,507 clusters, 4,083
system-ligand pairs in the final dataset) is the file that was used for docking,
feature generation, and training. Its cluster assignment matches the `split_clusters.csv`
of the final dataset exactly, and every cluster contributes min(5, cluster size)
systems, which is what this script produces. The random choices within clusters larger
than five are not recorded, however, so this script cannot regenerate the shipped file
bit-for-bit with any seed. Use the shipped CSV to
reproduce the published results, and this script to regenerate an equivalent subset for
a new dataset. The script never overwrites an existing output file unless
`--overwrite` is given.

Files Saved
-----------
- `cached_data/cluster_representatives_5.csv` (default): CSV file with the columns
  `cluster`, `representative_system_id`, and `representative_rank` (1-based order of
  the sampled systems within the cluster).

Usage
-----
Run from the `data_curation` directory:
```bash
python sample_train_representatives.py --output cached_data/cluster_representatives_5_regenerated.csv \
    --reference cached_data/cluster_representatives_5.csv
```
"""

import argparse
import os

import pandas as pd

# Seed for choosing train-cluster representatives, matching resplit_plinder.py.
RANDOM_SEED: int = 42
MAX_PER_CLUSTER: int = 5


def sample_representatives(
    split_clusters_df: pd.DataFrame,
    max_per_cluster: int = MAX_PER_CLUSTER,
    seed: int = RANDOM_SEED,
    allowed_system_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Select up to `max_per_cluster` random train systems from every train cluster.

    Parameters
    ----------
    split_clusters_df : pd.DataFrame
        Output of `resplit_plinder.py` with at least the columns `system_id`, `split`,
        and `cluster`.
    max_per_cluster : int, default: 5
        Maximum number of systems taken from each cluster.
    seed : int, default: 42
        Random seed passed to `DataFrame.sample` for every cluster.
    allowed_system_ids : set of str, optional
        If given, only these systems are eligible (e.g. the train systems of the
        curated dataset).

    Returns
    -------
    pd.DataFrame
        DataFrame with the columns `cluster`, `representative_system_id`, and
        `representative_rank`.
    """
    train_df = split_clusters_df[split_clusters_df["split"] == "train"]
    train_df = train_df.drop_duplicates(subset="system_id")
    if allowed_system_ids is not None:
        train_df = train_df[train_df["system_id"].isin(allowed_system_ids)]

    rows: list[tuple[str, str, int]] = []
    for cluster, group in train_df.groupby("cluster", sort=True):
        selected = group.sample(min(max_per_cluster, len(group)), random_state=seed)
        for rank, system_id in enumerate(selected["system_id"], start=1):
            rows.append((cluster, system_id, rank))

    return pd.DataFrame(
        rows,
        columns=["cluster", "representative_system_id", "representative_rank"],
    )


def compare_with_reference(new_df: pd.DataFrame, reference_df: pd.DataFrame) -> None:
    """
    Print how closely a newly sampled subset matches a reference subset.

    Parameters
    ----------
    new_df : pd.DataFrame
        Newly sampled representatives.
    reference_df : pd.DataFrame
        Reference representatives, e.g. the shipped `cluster_representatives_5.csv`.
    """
    new_sets = new_df.groupby("cluster")["representative_system_id"].apply(set)
    ref_sets = reference_df.groupby("cluster")["representative_system_id"].apply(set)
    shared = new_sets.index.intersection(ref_sets.index)
    same_size = sum(len(new_sets[c]) == len(ref_sets[c]) for c in shared)
    same_members = sum(new_sets[c] == ref_sets[c] for c in shared)
    overlap = len(
        set(new_df["representative_system_id"])
        & set(reference_df["representative_system_id"])
    )
    print(
        f"Clusters: {len(new_sets)} new, {len(ref_sets)} reference, {len(shared)} shared"
    )
    print(f"Shared clusters with the same number of systems: {same_size}")
    print(f"Shared clusters with identical systems: {same_members}")
    print(
        f"Systems in both subsets: {overlap} of {len(new_df)} new and "
        f"{len(reference_df)} reference systems"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select up to N random systems from every train cluster."
    )
    parser.add_argument(
        "--split-clusters",
        default="cached_data/split_clusters.csv",
        help="Cluster assignment written by resplit_plinder.py",
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional curated dataset CSV; only its train systems are sampled",
    )
    parser.add_argument(
        "--output",
        default="cached_data/cluster_representatives_5.csv",
        help="Output CSV path",
    )
    parser.add_argument("--max-per-cluster", type=int, default=MAX_PER_CLUSTER)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional existing representative CSV to compare the new subset against",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise SystemExit(
            f"{args.output} already exists. The shipped file is the one used for the "
            "published results; choose another --output or pass --overwrite."
        )

    split_clusters_df = pd.read_csv(args.split_clusters, keep_default_na=False)
    allowed_system_ids = None
    if args.dataset_csv is not None:
        # "NA" is a valid ligand chain ID and must not be parsed as a missing value.
        dataset_df = pd.read_csv(args.dataset_csv, keep_default_na=False)
        allowed_system_ids = set(
            dataset_df.loc[dataset_df["split"] == "train", "system_id"]
        )

    representatives_df = sample_representatives(
        split_clusters_df,
        max_per_cluster=args.max_per_cluster,
        seed=args.seed,
        allowed_system_ids=allowed_system_ids,
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    representatives_df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(representatives_df)} systems from "
        f"{representatives_df['cluster'].nunique()} clusters to {args.output}"
    )

    if args.reference is not None:
        reference_df = pd.read_csv(args.reference, keep_default_na=False)
        compare_with_reference(representatives_df, reference_df)


if __name__ == "__main__":
    main()
