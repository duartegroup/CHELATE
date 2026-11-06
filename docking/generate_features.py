import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from docking_params import (
    CLUSTER_REPRESENTATIVES_CSV,
    DATASET_CSV,
    DEFAULT_SYSTEMS_ROOT,
    METAL_LIGAND_DATA_JSON,
)
from rdkit import Chem

from chelate.features import compute_contact_features, compute_gasteiger_charges
from chelate.poses import read_poses
from chelate.receptor import extract_metals, load_receptor_pdb
from chelate.schema import get_score_col_for_method
from docking.receptor_pdb_cache import ensure_receptor_pdb


def _get_docked_path(
    data_path: str,
    system_id: str,
    ligand_id: str,
    method: str,
    run_type: str,
    run_id: str,
    exhaustiveness: int = 32,
    num_modes: int = 100,
    energy_range: float = 20.0,
) -> str:
    """
    Resolve the path to the docked-pose file for one system/ligand/method combination.

    The filename follows ``run_docking.py``, which writes e.g. ``eRange=20.0``. If only
    a file with the integer spelling ``eRange=20`` exists, that file is used.

    Parameters
    ----------
    data_path : str
        Root directory containing per-system docking results.
    system_id : str
        PLINDER system identifier.
    ligand_id : str
        Ligand identifier (``instance.asym_id`` format).
    method : str
        Docking program: ``"SMINA"``, ``"VINA"``, or ``"GNINA"``.
    run_type : str
        Docking run type (e.g. ``"apo_docked_FH"`` or ``"redocked_FH"``).
    run_id : str
        Docking job identifier used in the filename.
    exhaustiveness, num_modes, energy_range : int, int, float
        Docking settings, as passed to ``run_docking.py``.

    Returns
    -------
    str
        Path to the docked-pose file.
    """
    prefix = (
        f"{data_path}/{system_id}/{method}_{run_type}_id={run_id}_"
        f"exhaust={exhaustiveness}_numModes={num_modes}"
    )
    if method == "GNINA":
        return f"{prefix}_prepPH=7.4_lig={ligand_id}.sdf.gz"
    if method not in {"SMINA", "VINA"}:
        raise ValueError(f"Unknown docking method: {method!r}")

    # run_docking.py writes Vina poses as PDBQT and exports them to SDF next to it.
    ext = "pdbqt" if method == "SMINA" else "sdf"
    candidates = [
        f"{prefix}_eRange={energy_range}_lig={ligand_id}.{ext}",
        f"{prefix}_eRange={energy_range:g}_lig={ligand_id}.{ext}",
    ]
    return next((c for c in candidates if os.path.exists(c)), candidates[0])


def _process_one_system(
    system_id: str,
    target_system: str,
    ligand_id: str,
    split_name: str,
    method: str,
    run_type: str,
    run_id: str,
    data_path: str,
    system_metals: list[tuple[str, list[float]]],
    docking_settings: dict | None = None,
) -> tuple[list[dict], str, str]:
    """
    Generate feature rows for all docked poses of one system-ligand pair.

    Uses CHELATE's pose reader, metal extraction and contact feature engine.
    Computes reference-dependent RMSD labels via ``docking_utils.get_docking_data``
    using the same parsed poses.

    Parameters
    ----------
    system_id : str
        PLINDER system identifier.
    target_system : str
        Target (apo) system identifier used to resolve the apo receptor path.
    ligand_id : str
        Ligand identifier (``instance.asym_id`` format).
    split_name : str
        Dataset split: ``"train"``, ``"val"``, or ``"test"``.
    method : str
        Docking program: ``"SMINA"``, ``"VINA"``, or ``"GNINA"``.
    run_type : str
        Docking run type (e.g. ``"apo_docked_FH"`` or ``"redocked_FH"``).
    run_id : str
        Docking job identifier.
    data_path : str
        Root directory for per-system input files.
    system_metals : list[tuple[str, list[float]]]
        Metal list for this system from the curated JSON; each entry is
        ``[symbol, [x, y, z]]``.
    docking_settings : dict, optional
        ``exhaustiveness``, ``num_modes`` and ``energy_range`` of the docking run,
        used to build the docked-pose filename.

    Returns
    -------
    tuple[list[dict], str, str]
        ``(rows, status, message)`` where *status* is ``"ok"`` on success or a short
        error tag (e.g. ``"missing_receptor"``, ``"docking_data_error"``) on failure,
        and *message* is a human-readable description.
    """

    # Keep the extra docking dependencies (pymol, openbabel and biotite) out of
    # module-level imports; the shared pose and feature helpers do not need them.
    import docking_utils

    if run_type.startswith("apo_docked"):
        receptor_path_in = f"{data_path}/{system_id}/{system_id}__{target_system}_merged_{ligand_id}.pdbqt"
        receptor_path = f"{data_path}/{system_id}/{system_id}__{target_system}_merged_{ligand_id}.pdb"

    else:
        receptor_path_in = f"{data_path}/{system_id}/receptor_merged_{ligand_id}.pdbqt"
        receptor_path = f"{data_path}/{system_id}/receptor_merged_{ligand_id}.pdb"

    try:
        ensure_receptor_pdb(Path(receptor_path_in), Path(receptor_path))
    except (OSError, ValueError, RuntimeError) as exc:
        return [], "obabel_error", f"Receptor conversion failed: {exc}"

    if not os.path.exists(receptor_path):
        return (
            [],
            "missing_receptor",
            f"Receptor file not found after conversion: {receptor_path}",
        )

    ligand_path = f"{data_path}/{system_id}/{ligand_id}.sdf"
    docked_path: str = _get_docked_path(
        data_path,
        system_id,
        ligand_id,
        method,
        run_type,
        run_id,
        **(docking_settings or {}),
    )

    if not os.path.exists(ligand_path):
        return [], "missing_ligand", f"Ligand file not found: {ligand_path}"
    if not os.path.exists(docked_path):
        return [], "missing_docked", f"Docked file not found: {docked_path}"

    try:
        docked_mols: list[Chem.rdchem.Mol]
        docked_mols, docking_scores = read_poses(docked_path, method)
    except Exception as exc:
        return (
            [],
            "parse_docked_error",
            f"Could not parse docked file for {system_id}: {exc}",
        )

    if len(docked_mols) == 0:
        warnings.warn(
            f"No docked poses found in {docked_path} for {system_id}__{ligand_id}.",
            stacklevel=2,
        )

    if not system_metals:
        warnings.warn(
            f"No metals in metal_ligand_data for {system_id}; metal-binding-group RMSD will be unavailable.",
            stacklevel=2,
        )

    metal_coords = [m[1] for m in system_metals]
    try:
        docking_df: pd.DataFrame
        docking_df = docking_utils.get_docking_data(
            docked_path,
            ligand_path,
            metal_coords,
            method,
            docked_mols=docked_mols,
            docking_scores=docking_scores,
        )
    except Exception as exc:
        return (
            [],
            "docking_data_error",
            f"Could not compute docking data for {system_id}: {exc}",
        )
    pose_by_rank = {int(row["pose"]): row.to_dict() for _, row in docking_df.iterrows()}

    structure_rdkit = load_receptor_pdb(receptor_path)
    if structure_rdkit is None:
        return [], "parse_receptor_error", f"Could not parse receptor for {system_id}"

    receptor_metals: list[tuple[str, np.ndarray]] = extract_metals(structure_rdkit)
    if len(receptor_metals) == 0:
        return [], "missing_metals", f"No receptor metal atoms found for {system_id}"
    receptor_charges = compute_gasteiger_charges(structure_rdkit)

    n_poses = len(docked_mols)
    score_col = get_score_col_for_method(method)
    rows: list[dict] = []
    for pose_idx, mol in enumerate(docked_mols, start=1):
        pose_data = pose_by_rank.get(pose_idx)
        if pose_data is None:
            continue

        try:
            feats: pd.Series = compute_contact_features(
                structure_rdkit,
                mol,
                receptor_metals,
                pose_idx=pose_idx,
                n_poses=n_poses,
                receptor_charges=receptor_charges,
            )
        except Exception as exc:
            return (
                [],
                "feature_error",
                f"Feature calc failed ({system_id} pose {pose_idx}): {exc}",
            )

        row_dict = {
            "system_id": system_id,
            "ligand_id": ligand_id,
            "split": split_name,
            "method": method,
            "pose": pose_idx,
            "symm_rmsd": pose_data.get("rmsd"),
            "close_rmsd_3.5_hetero": pose_data.get("close_rmsd_3.5_hetero"),
            **feats.to_dict(),
        }

        if method == "GNINA":
            row_dict["minimizedAffinity"] = pose_data.get("minimizedAffinity")
            row_dict["CNNaffinity"] = pose_data.get("CNNaffinity")
        row_dict[score_col] = pose_data.get(score_col)

        rows.append(row_dict)

    if n_poses > 0 and len(rows) == 0:
        warnings.warn(
            f"All {n_poses} poses were skipped for {system_id}__{ligand_id} "
            "(pose index not found in docking_df).",
            stacklevel=2,
        )

    return rows, "ok", f"Processed {system_id}__{ligand_id} ({len(rows)} rows)"


def main():
    """
    Entry point for feature generation.

    Reads the PLINDER dataset CSV, resolves train/val/test splits, iterates over all
    protein-ligand pairs, and dispatches ``_process_one_system`` - either sequentially or
    via a ``ProcessPoolExecutor``. Results are concatenated and written to a single CSV
    under ``features/``.

    CLI arguments
    -------------
    --method : str
        Docking program (``SMINA``, ``VINA``, or ``GNINA``).
    --id : str
        Docking job identifier.
    --type : str
        Docking run type (e.g. ``apo_docked_FH``).
    --split : str
        Data split to process (``train``, ``val``, or ``test``).
    --workers : int
        Number of parallel worker processes (default: 1 = sequential).
    --max-systems : int
        If > 0, limit processing to this many unique systems.
    --dataset-csv : str
        Curated PLINDER dataset CSV (default: ``docking_params.DATASET_CSV``).
    --repr-csv : str
        Cluster-representative CSV (default: ``docking_params.CLUSTER_REPRESENTATIVES_CSV``).
    --metal-ligand-data : str
        Per-system metal/ligand JSON (default: ``docking_params.METAL_LIGAND_DATA_JSON``).
    --exhaustiveness, --num-modes, --energy-range
        Settings of the docking run, used to locate its output files (defaults match
        ``run_docking.py``).
    """
    parser = argparse.ArgumentParser(
        description="Generate CHELATE features with optional multiprocessing"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SMINA",
        choices=["SMINA", "VINA", "GNINA"],
        help="Docking method used",
    )
    parser.add_argument("--id", type=str, default="3", help="Docking job identifier")
    parser.add_argument(
        "--type",
        type=str,
        default="apo_docked_FH",
        help="Docking type (e.g., apo_docked_FH or redocked_FH)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-system parallelism",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=0,
        help="If >0, process only the first N unique systems",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=DATASET_CSV,
        help="Curated PLINDER dataset CSV (systems, ligands, splits)",
    )
    parser.add_argument(
        "--repr-csv",
        type=str,
        default=CLUSTER_REPRESENTATIVES_CSV,
        help="Cluster-representative CSV used to subsample the train split",
    )
    parser.add_argument(
        "--metal-ligand-data",
        type=str,
        default=METAL_LIGAND_DATA_JSON,
        help="Per-system metal/ligand coordinate JSON",
    )
    parser.add_argument(
        "--systems-root",
        default=os.environ.get("SYSTEMS_ROOT", DEFAULT_SYSTEMS_ROOT),
        help=(
            "Directory containing the train_systems, val_systems, and test_systems "
            "folders (default: $SYSTEMS_ROOT or repository/docking)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DEFAULT_SYSTEMS_ROOT, "features"),
        help="Directory in which to write the generated feature CSV",
    )
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=100)
    parser.add_argument("--energy-range", type=float, default=20.0)
    args = parser.parse_args()
    docking_settings = {
        "exhaustiveness": args.exhaustiveness,
        "num_modes": args.num_modes,
        "energy_range": args.energy_range,
    }

    method = args.method
    run_id = args.id
    run_type = args.type
    split_name = args.split
    workers = max(1, args.workers)
    max_systems = max(0, args.max_systems)

    dataset_csv = args.dataset_csv
    repr_csv = args.repr_csv
    metal_ligand_data_path = args.metal_ligand_data
    data_path = os.path.join(args.systems_root, f"{split_name}_systems")
    out_name = os.path.join(
        args.output_dir,
        f"features_{split_name}_{method}_{run_type}_id={run_id}.csv",
    )

    # "NA" is a valid ligand chain ID and must not be parsed as a missing value.
    plinder_df: pd.DataFrame = pd.read_csv(dataset_csv, keep_default_na=False)
    plinder_df["ligand_id"] = (
        plinder_df["ligand_instance"].astype(str)
        + "."
        + plinder_df["ligand_asym_id"].astype(str)
    )
    plinder_df = plinder_df[plinder_df["split"] == split_name]

    if split_name == "train":
        repr_df: pd.DataFrame = pd.read_csv(repr_csv, keep_default_na=False)
        plinder_df = plinder_df[
            plinder_df["system_id"].isin(repr_df["representative_system_id"])
        ]

    with open(metal_ligand_data_path) as f:
        metal_ligand_data = json.load(f)

    tasks: list[tuple] = []
    for _, row in plinder_df.iterrows():
        system_id = row["system_id"]
        target_system = row["target_system"]
        ligand_id = row["ligand_id"]
        tasks.append(
            (
                system_id,
                target_system,
                ligand_id,
                split_name,
                method,
                run_type,
                run_id,
                data_path,
                metal_ligand_data.get(system_id, {}).get("metals", []),
                docking_settings,
            )
        )

    if max_systems > 0:
        selected = set()
        limited_tasks: list[tuple] = []
        for task in tasks:
            system_id = task[0]
            if system_id not in selected:
                if len(selected) >= max_systems:
                    continue
                selected.add(system_id)
            limited_tasks.append(task)
        tasks = limited_tasks
        print(
            f"Limiting run to first {max_systems} unique systems ({len(tasks)} system-ligand tasks)"
        )

    total = len(tasks)
    print(f"Processing {total} systems with workers={workers}")

    rows_accum: list[dict] = []
    failed_cnt = 0
    processed_cnt = 0

    if workers == 1:
        for task in tasks:
            processed_cnt += 1
            rows: list[dict]
            status: str
            msg: str
            rows, status, msg = _process_one_system(*task)
            if status == "ok":
                rows_accum.extend(rows)
                print(f"[{processed_cnt}/{total}] ok: {msg}")
            else:
                failed_cnt += 1
                print(f"[{processed_cnt}/{total}] {status}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_process_one_system, *task): (task[0], task[2])
                for task in tasks
            }
            for future in as_completed(future_map):
                processed_cnt += 1
                system_id, ligand_id = future_map[future]
                try:
                    rows, status, msg = future.result()
                except Exception as exc:
                    failed_cnt += 1
                    print(
                        f"[{processed_cnt}/{total}] worker_error: {system_id}__{ligand_id} -> {exc}"
                    )
                    continue

                if status == "ok":
                    rows_accum.extend(rows)
                    print(f"[{processed_cnt}/{total}] ok: {msg}")
                else:
                    failed_cnt += 1
                    print(f"[{processed_cnt}/{total}] {status}: {msg}")

    global_df = pd.DataFrame(rows_accum)
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    global_df.to_csv(out_name, index=False)
    print(f"Saved {len(global_df)} rows to {out_name}")
    print(f"Failed systems: {failed_cnt}")


if __name__ == "__main__":
    main()
