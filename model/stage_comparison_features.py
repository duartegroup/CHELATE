"""Stage the frozen comparison inputs without changing numeric CSV text.

Use the committed availability-only cohort for model evaluation, the distance
baseline, the training-fraction sweep and Spearman plots. Train/validation files
are copied unchanged; only test poses are filtered by existing input eligibility.
No model, prediction cutoff or success threshold is read.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from model.cohorts import (
    METHODS,
    REPEATS,
    SCENARIOS,
    complete_native_cohort,
    read_cohort,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_pose_rows(source, destination, selected):
    """Preserve every original byte of each selected, uniquely identified pose."""
    written = set()
    with Path(source).open("rb") as source_file, Path(destination).open("xb") as out:
        header = source_file.readline()
        names = next(csv.reader([header.decode()]))
        indices = [names.index(name) for name in ["system_id", "ligand_id", "pose"]]
        out.write(header)
        for line in source_file:
            values = next(csv.reader([line.decode()]))
            key = (values[indices[0]], values[indices[1]], int(values[indices[2]]))
            if key not in selected:
                continue
            if key in written:
                raise ValueError("Duplicate selected pose identifier")
            written.add(key)
            out.write(line)
    if written != selected:
        raise ValueError("Selected pose identifiers are missing from the input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = read_cohort(args.cohort)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for method in METHODS:
        for split in ["train", "val"]:
            source = (
                args.features_dir / f"features_{split}_{method}_redocked_FH_id=1.csv"
            )
            before = digest(source)
            destination = args.output_dir / source.name
            shutil.copy2(source, destination)
            if digest(destination) != before or digest(source) != before:
                raise ValueError("Train/validation input changed during staging")
            records.append(
                {"file": source.name, "source_sha256": before, "sha256": before}
            )
        for scenario in SCENARIOS:
            for repeat in REPEATS:
                source = (
                    args.features_dir
                    / f"features_test_{method}_{scenario}_id={repeat}.csv"
                )
                before = digest(source)
                columns = [
                    "system_id",
                    "ligand_id",
                    "pose",
                    "close_rmsd_3.5_hetero",
                    "symm_rmsd",
                    "CNNscore" if method == "GNINA" else "docking_score",
                ]
                frame = pd.read_csv(
                    source, usecols=columns, keep_default_na=False, na_values=[""]
                )
                selected = complete_native_cohort(frame, pairs, method)
                keys = set(
                    selected[["system_id", "ligand_id", "pose"]].itertuples(
                        index=False, name=None
                    )
                )
                if len(keys) != len(selected):
                    raise ValueError("Duplicate selected pose identifiers")
                destination = args.output_dir / source.name
                copy_pose_rows(source, destination, keys)
                if digest(source) != before:
                    raise ValueError("Test input changed during staging")
                records.append(
                    {
                        "file": source.name,
                        "source_sha256": before,
                        "sha256": digest(destination),
                        "rows": len(keys),
                        "pairs": len(pairs),
                    }
                )
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "cohort_sha256": digest(args.cohort),
                "pairs": len(pairs),
                "records": records,
                "numeric_text_preserved": True,
                "models_or_cutoffs_read": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
