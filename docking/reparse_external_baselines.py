"""Parse saved AutoDock, GPDOCK, and MetalloDock test poses into result CSVs.

For every method, scenario, and run, this script reads the saved pose files and
computes symmetry-corrected full-ligand and metal-binding-group RMSDs and the
other per-pose metrics with ``docking.docking_utils.get_docking_data``.  It does
not run any docking program.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "docking"))

from docking import docking_utils  # noqa: E402

METHODS = ("AUTODOCK", "GPDOCK", "METALLODOCK")
SCENARIOS = ("redocked", "apo_docked")
RUN_IDS = (1, 2, 3)


@dataclass(frozen=True)
class Task:
    method: str
    scenario: str
    run_id: int
    system_id: str
    ligand_id: str
    target_system: str
    pose_path: str
    ligand_path: str
    metal_coords: tuple[tuple[float, float, float], ...]


def pose_path(
    systems_dir: Path,
    method: str,
    scenario: str,
    run_id: int,
    system_id: str,
    ligand_id: str,
) -> Path:
    base = systems_dir / system_id
    if method == "AUTODOCK":
        name = (
            f"AUTODOCK_{scenario}_FH_id={run_id}_nrun=100_boxExt=4.0_"
            f"{system_id}_lig={ligand_id}.dlg"
        )
    elif method == "GPDOCK":
        name = f"GPDOCK_{scenario}_FH_id={run_id}_HOH=0_lig={ligand_id}_out.pdb"
    elif method == "METALLODOCK":
        name = (
            f"METALLODOCK_none_leak_{scenario}_FH_"
            f"no_leak_max_plus_4_ball_multi_pose_id={run_id}_lig={ligand_id}.sdf"
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return base / name


def output_name(method: str, scenario: str, run_id: int) -> str:
    if method == "AUTODOCK":
        return f"AUTODOCK_{scenario}_FH_id={run_id}_nrun=100_boxExt=4.0.csv"
    if method == "GPDOCK":
        return f"GPDOCK_{scenario}_FH_id={run_id}_HOH=0.csv"
    if method == "METALLODOCK":
        return (
            f"METALLODOCK_{scenario}_FH_"
            f"no_leak_max_plus_4_ball_multi_pose_id={run_id}.csv"
        )
    raise ValueError(f"Unsupported method: {method}")


def process_task(task: Task) -> tuple[Task, pd.DataFrame | None, str | None]:
    try:
        result = docking_utils.get_docking_data(
            task.pose_path,
            task.ligand_path,
            list(task.metal_coords),
            docking_type=task.method.lower(),
        )
        if task.scenario == "redocked":
            result["system_id"] = task.system_id
        else:
            result["alt_system_id"] = f"{task.system_id}__{task.target_system}"
        result["ligand_id"] = task.ligand_id
        return task, result, None
    except Exception:
        return task, None, traceback.format_exc()


def main() -> None:
    # Relative defaults and arguments are resolved from the repository root. The
    # working directory changes here, not at import time.
    os.chdir(REPO_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Folder holding test_systems/ with the saved external-baseline outputs.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path(
            "data_curation/cached_data/"
            "final_dataset_distTresh=4.0_mwThresh=750.0_"
            "goodApoThresh=70.0_resplit.csv"
        ),
    )
    parser.add_argument(
        "--metal-json",
        type=Path,
        default=Path(
            "data_curation/cached_data/"
            "metal_ligand_data_distTresh=4.0_mwThresh=750.0_"
            "goodApoThresh=70.0_resplit.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docking/results/test"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()

    systems_dir = args.source_root / "test_systems"
    dataset = pd.read_csv(args.dataset_csv, keep_default_na=False)
    dataset = dataset[dataset["split"] == "test"].copy()
    dataset["ligand_id"] = (
        dataset["ligand_instance"].astype(str)
        + "."
        + dataset["ligand_asym_id"].astype(str)
    )
    with args.metal_json.open() as handle:
        metal_data = json.load(handle)

    tasks: list[Task] = []
    missing: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in SCENARIOS:
            for run_id in RUN_IDS:
                for row in dataset.itertuples(index=False):
                    system_id = str(row.system_id)
                    ligand_id = str(row.ligand_id)
                    path = pose_path(
                        systems_dir,
                        method,
                        scenario,
                        run_id,
                        system_id,
                        ligand_id,
                    )
                    ligand_path = systems_dir / system_id / f"{ligand_id}.sdf"
                    metals = metal_data.get(system_id, {}).get("metals", [])
                    metal_coords = tuple(tuple(metal[1]) for metal in metals)
                    if (
                        not path.exists()
                        or not ligand_path.exists()
                        or not metal_coords
                    ):
                        missing.append(
                            {
                                "method": method,
                                "scenario": scenario,
                                "run_id": run_id,
                                "system_id": system_id,
                                "ligand_id": ligand_id,
                                "pose_exists": path.exists(),
                                "ligand_exists": ligand_path.exists(),
                                "has_metal_coordinates": bool(metal_coords),
                                "pose_path": str(path),
                            }
                        )
                        continue
                    tasks.append(
                        Task(
                            method=method,
                            scenario=scenario,
                            run_id=run_id,
                            system_id=system_id,
                            ligand_id=ligand_id,
                            target_system=str(row.target_system),
                            pose_path=str(path),
                            ligand_path=str(ligand_path),
                            metal_coords=metal_coords,
                        )
                    )

    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing).to_csv(args.output_dir / "missing_inputs.csv", index=False)
    print(
        f"Prepared {len(tasks)} tasks; {len(missing)} expected inputs are unavailable."
    )

    grouped: dict[tuple[str, str, int], list[pd.DataFrame]] = {}
    failures: list[dict[str, object]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for future in as_completed(futures):
            task, result, error = future.result()
            completed += 1
            if error is None and result is not None:
                grouped.setdefault(
                    (task.method, task.scenario, task.run_id), []
                ).append(result)
            else:
                failures.append(
                    {
                        "method": task.method,
                        "scenario": task.scenario,
                        "run_id": task.run_id,
                        "system_id": task.system_id,
                        "ligand_id": task.ligand_id,
                        "pose_path": task.pose_path,
                        "error": error,
                    }
                )
            if completed % 50 == 0 or completed == len(tasks):
                print(
                    f"Completed {completed}/{len(tasks)} tasks; "
                    f"failures={len(failures)}",
                    flush=True,
                )

    pd.DataFrame(failures).to_csv(args.output_dir / "parse_failures.csv", index=False)
    for method in METHODS:
        for scenario in SCENARIOS:
            for run_id in RUN_IDS:
                frames = grouped.get((method, scenario, run_id), [])
                if not frames:
                    continue
                output = pd.concat(frames, ignore_index=True)
                output.to_csv(
                    args.output_dir / output_name(method, scenario, run_id),
                    index=False,
                )
                print(
                    f"Wrote {output_name(method, scenario, run_id)}: "
                    f"{len(output)} poses, "
                    f"{output.iloc[:, -2:].drop_duplicates().shape[0]} systems/ligands",
                    flush=True,
                )


if __name__ == "__main__":
    main()
