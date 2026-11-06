"""
Description
-----------
This script writes the superimposed apo structures used as cross-docking receptors in
the layout that `docking/run_docking.py` expects:

    {systems_root}/{split}_systems/{system_id}/{system_id}__{target_system}.cif

`superimpose_structures.run_query` (called by `generate_dataset.py`) selects and aligns
the apo structures, but it saves them as PDB files under `superimposed_structures/`,
and it returns cached results without re-aligning once `cached_data/matched_final.csv`
exists. This script therefore re-runs `superimpose_structures.process_row` for every
system of one split and writes each aligned structure directly as a Biotite mmCIF file.

Relation to the published inputs
--------------------------------
The script processes every test-split row of the final dataset CSV in order, so for
systems with several ligand rows the last row's alignment is written, as in the
published cross-docking CIFs. On eight test systems (1J3R, 1Z4K, 2CJ9, 2GGX, 2Z4V,
4YWY, 6IL2, 8J3C) the atoms, coordinates (RMSD and maximum difference 0 Å), label and
author chain IDs, metal identities and metal coordinates are identical to the published
CIFs. The published inputs differ from this script's output only in CIF metadata: their
metal residues are numbered -1 rather than 1, and their atom-site loop has 21 rather
than 18 columns, with the extra fields `atom_id`, `b_factor`, `occupancy`, and `charge`.
The docking pipeline does not use these fields.

Usage
-----
Run from any directory in `plinder_env`:
```bash
python data_curation/export_crossdocking_inputs.py --split test
```
Existing CIF files are skipped unless `--overwrite` is given.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import superimpose_structures

REPO = Path(__file__).resolve().parents[1]
CACHED_DATA = REPO / "data_curation" / "cached_data"
DEFAULT_SYSTEMS_ROOT = REPO / "docking"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write superimposed apo structures as cross-docking input CIFs."
    )
    parser.add_argument("--split", default="test", help="Dataset split to export")
    parser.add_argument(
        "--dataset-csv",
        default=CACHED_DATA
        / "final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.csv",
        help="Curated dataset CSV produced by generate_dataset.py",
    )
    parser.add_argument(
        "--alt-structures-csv",
        default=CACHED_DATA / "alt_structures_final.csv",
        help="Alternative-structure table produced by get_linked_structure.py",
    )
    parser.add_argument(
        "--systems-root",
        default=os.environ.get("SYSTEMS_ROOT", DEFAULT_SYSTEMS_ROOT),
        help=(
            "Directory holding the {split}_systems folders "
            "(default: $SYSTEMS_ROOT or repository/docking)"
        ),
    )
    parser.add_argument(
        "--plinder-systems",
        default=os.environ.get("PLINDER_SYSTEMS_PATH"),
        help=(
            "Optional local PLINDER systems directory. When omitted, the PLINDER "
            "client resolves the system files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite CIF files that already exist",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # "NA" is a valid ligand chain ID and must not be parsed as a missing value.
    dataset_df = pd.read_csv(args.dataset_csv, keep_default_na=False)
    dataset_df = dataset_df[dataset_df["split"] == args.split]
    # Only systems with a good apo structure are cross-docked.
    dataset_df = dataset_df[dataset_df["target_system"] != ""]
    # Rows are processed in order, so the last row of a system is the one written, as in
    # the published inputs.
    dataset_df = dataset_df.drop_duplicates(subset="system_id", keep="last")

    alt_structures_df = pd.read_csv(
        args.alt_structures_csv, keep_default_na=False, na_values=[""]
    )
    # Predicted structures have no metals, so only experimental structures are used.
    alt_structures_df = alt_structures_df[alt_structures_df["type"] == "apo"]

    cif_output_dir = os.path.join(args.systems_root, f"{args.split}_systems")
    n_written, n_skipped, n_failed = 0, 0, 0
    for _, row in dataset_df.iterrows():
        system_id = row["system_id"]
        expected_cif = os.path.join(
            cif_output_dir, system_id, f"{system_id}__{row['target_system']}.cif"
        )
        if os.path.exists(expected_cif) and not args.overwrite:
            n_skipped += 1
            continue

        try:
            result = superimpose_structures.process_row(
                row,
                alt_structures_df,
                save_superimposed_structures=False,
                cif_output_dir=cif_output_dir,
                plinder_systems_path=args.plinder_systems,
            )
        except Exception as err:
            logging.warning(f"{system_id} failed: {err}")
            result = None

        if result is None:
            n_failed += 1
            continue
        if result["target_system"] != row["target_system"]:
            logging.warning(
                f"{system_id}: aligned {result['target_system']}, but the dataset "
                f"lists {row['target_system']}; run_docking.py will not find this file."
            )
        n_written += 1

    print(
        f"Wrote {n_written} CIFs to {cif_output_dir}; skipped {n_skipped} existing; "
        f"{n_failed} failed."
    )


if __name__ == "__main__":
    main()
