"""Build the chelation and metal-ligand distance caches for Figs. S4 and S5.

The PLINDER structures are external inputs.  By default they are expected in the
standard project layout next to this repository's dataset root; every input and output
path can also be supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATASET_ROOT = REPO_ROOT.parents[1]

DEFAULT_DATASET_CSV = (
    REPO_ROOT
    / "figures"
    / "inputs"
    / "final_dataset_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_"
    "resplit_NEW.csv"
)
DEFAULT_METAL_JSON = (
    REPO_ROOT
    / "data_curation"
    / "cached_data"
    / "metal_ligand_data_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_"
    "resplit.json"
)
DEFAULT_PLINDER_SYSTEMS = (
    DATASET_ROOT.parent / "plinder" / "plinder" / "2024-06" / "v2" / "systems"
)

CHELATION_THRESHOLD_A = 3.5
DONOR_ELEMENTS = frozenset({"N", "O", "S"})
OTHER_HETERO_ELEMENTS = frozenset(
    {"P", "F", "Cl", "Br", "I", "Se", "As", "B", "Si", "Te"}
)
ALL_HETERO_ELEMENTS = DONOR_ELEMENTS | OTHER_HETERO_ELEMENTS
AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "MSE",
        "SEC",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--metal-json", type=Path, default=DEFAULT_METAL_JSON)
    parser.add_argument("--plinder-systems", type=Path, default=DEFAULT_PLINDER_SYSTEMS)
    parser.add_argument(
        "--chelation-output", type=Path, default=HERE / "chelation_cache.pkl"
    )
    parser.add_argument(
        "--distances-output",
        type=Path,
        default=HERE / "metal_ligand_distances_cache.pkl",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def extract_chelation(task: tuple) -> tuple[str, list[tuple[str, str, float]]]:
    """Return the protein residues chelating each metal in one PLINDER system."""
    system_id, entry, systems_root = task
    receptor = systems_root / system_id / "receptor.pdb"
    metals = entry.get("metals", [])
    if not receptor.is_file() or not metals:
        return system_id, []

    try:
        structure = gemmi.read_structure(str(receptor))
        donors = []
        for chain in structure[0]:
            for residue in chain:
                if residue.name not in AMINO_ACIDS:
                    continue
                for atom in residue:
                    if (
                        atom.element.name in DONOR_ELEMENTS
                        and not atom.element.is_hydrogen
                    ):
                        donors.append(
                            (
                                (chain.name, str(residue.seqid), residue.name),
                                atom.pos,
                            )
                        )
        interactions = []
        for metal_name, metal_coords in metals:
            metal_position = gemmi.Position(*metal_coords)
            closest_by_residue: dict[tuple[str, str, str], float] = {}
            # PLINDER receptor PDBs use a dummy 1 Å unit cell. The receptor is an
            # assembled molecular structure in the same Cartesian frame as the
            # metal coordinates, so periodic boundary conditions do not apply.
            for residue, donor_position in donors:
                distance = donor_position.dist(metal_position)
                if distance > CHELATION_THRESHOLD_A:
                    continue
                closest_by_residue[residue] = min(
                    distance, closest_by_residue.get(residue, float("inf"))
                )
            interactions.extend(
                (metal_name, residue_name, distance)
                for (_, _, residue_name), distance in sorted(closest_by_residue.items())
            )
        return system_id, interactions
    except (RuntimeError, ValueError):
        return system_id, []


def build_chelation_cache(
    system_ids: list[str], metal_data: dict, systems_root: Path, workers: int
) -> dict[str, list[tuple[str, str, float]]]:
    tasks = [
        (system_id, metal_data.get(system_id, {}), systems_root)
        for system_id in system_ids
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_chelation, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Chelating residues"
        ):
            system_id, interactions = future.result()
            results[system_id] = interactions
    return results


def ligand_id(row: pd.Series) -> str:
    return f"{row['ligand_instance']}.{row['ligand_asym_id']}"


def metal_ligand_distance_rows(
    dataset: pd.DataFrame, metal_data: dict, systems_root: Path
) -> list[dict]:
    """Calculate all crystallographic ligand-heteroatom-to-metal distances."""
    rows = []
    for count, (_, record) in enumerate(dataset.iterrows(), start=1):
        system_id = record["system_id"]
        current_ligand = ligand_id(record)
        metals = metal_data.get(system_id, {}).get("metals", [])
        sdf = systems_root / system_id / "ligand_files" / f"{current_ligand}.sdf"
        if not metals or not sdf.is_file():
            continue
        try:
            molecule = next(
                (mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=True) if mol),
                None,
            )
        except OSError:
            molecule = None
        if molecule is None:
            continue

        conformer = molecule.GetConformer()
        for atom in molecule.GetAtoms():
            element = atom.GetSymbol()
            if element not in ALL_HETERO_ELEMENTS:
                continue
            position = conformer.GetAtomPosition(atom.GetIdx())
            atom_coords = np.array((position.x, position.y, position.z))
            for metal_symbol, metal_coords in metals:
                rows.append(
                    {
                        "system_id": system_id,
                        "ligand_id": current_ligand,
                        "split": record["split"],
                        "atom_idx": atom.GetIdx(),
                        "element": element.upper(),
                        "metal_symbol": metal_symbol,
                        "distance": float(
                            np.linalg.norm(atom_coords - np.asarray(metal_coords))
                        ),
                    }
                )
        if count % 500 == 0:
            print(f"Processed {count:,}/{len(dataset):,} dataset rows", flush=True)
    return rows


def write_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")


def main() -> None:
    args = parse_args()
    for path in (args.dataset_csv, args.metal_json, args.plinder_systems):
        if not path.exists():
            raise FileNotFoundError(path)

    dataset = pd.read_csv(args.dataset_csv, keep_default_na=False)
    dataset = dataset[dataset["split"].isin(("train", "val", "test"))].copy()
    with args.metal_json.open() as handle:
        metal_data = json.load(handle)

    chelation = build_chelation_cache(
        dataset["system_id"].drop_duplicates().tolist(),
        metal_data,
        args.plinder_systems,
        args.workers,
    )
    write_pickle(chelation, args.chelation_output)

    distance_rows = metal_ligand_distance_rows(
        dataset, metal_data, args.plinder_systems
    )
    write_pickle(pd.DataFrame(distance_rows), args.distances_output)


if __name__ == "__main__":
    main()
