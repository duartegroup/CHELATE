"""Parse the saved leakage-control ("unseen systems") poses into pose CSVs.

Figs. S13 and S14 compare GPDOCK and MetalloDock, re-docked on test systems absent
from their training data, with Vina and smina docked on the same systems. This
script computes the whole-molecule and MBG RMSDs of the saved poses in
``<poses_dir>/<system_id>/`` with ``docking.docking_utils.get_docking_data``. No
docking program is run, and no existing file is modified.

The output CSVs use the file names that ``build_benchmark_results.py`` reads from
its ``--unseen_dir``.

Usage (plinder_env)::

    python reparse_unseen_systems.py --poses_dir <unseen_systems> \
        --output_dir unseen_systems --workers 32
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METAL_JSON = (
    REPO
    / "data_curation"
    / "cached_data"
    / "metal_ligand_data_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.json"
)

# (method, csv stem template, pose-file regex). The regex captures run id and ligand id.
SPECS = [
    (
        "GPDOCK",
        "GPDOCK_redocked_FHUNSEEN_ONLY_id={rid}_HOH=0",
        r"^GPDOCK_redocked_FHUNSEEN_ONLY_id=(\d)_HOH=0_lig=(.+)_out\.pdb$",
    ),
    (
        "METALLODOCK",
        "METALLODOCK_redocked_FH_no_leak_max_plus_4_ball_multi_pose_UNSEEN_ONLY_id={rid}_min_method=uff",
        r"^METALLODOCK_none_leak_redocked_FH_no_leak_max_plus_4_ball_multi_pose_UNSEEN_ONLY_id=(\d)_lig=(.+)_min_method=uff\.sdf$",
    ),
]
for _tool in ("GPDOCK", "METALLODOCK"):
    SPECS.append(
        (
            "VINA",
            f"VINA_redocked_FHUNSEEN_ONLY_{_tool}_id={{rid}}_exhaust=32_numModes=100_eRange=20.0",
            rf"^VINA_redocked_FHUNSEEN_ONLY_{_tool}_id=(\d)_exhaust=32_numModes=100_eRange=20\.0_lig=(.+)\.sdf$",
        )
    )
    SPECS.append(
        (
            "SMINA",
            f"SMINA_redocked_FHUNSEEN_ONLY_{_tool}_id={{rid}}_exhaust=32_numModes=100_eRange=20.0",
            rf"^SMINA_redocked_FHUNSEEN_ONLY_{_tool}_id=(\d)_exhaust=32_numModes=100_eRange=20\.0_lig=(.+)\.pdbqt$",
        )
    )

_METALS: dict = {}


def _init(metal_json: str) -> None:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "docking"))
    with open(metal_json) as f:
        _METALS.update(json.load(f))


def _work(task: tuple) -> tuple:
    stem, method, system_id, ligand_id, pose_path = task
    from docking import docking_utils  # imported in the worker after sys.path setup

    try:
        metals = _METALS.get(system_id, {}).get("metals", [])
        if not metals:
            return task, None, "system has no metal coordinates in metal_ligand_data"
        # The reference ligand sits next to the poses, in <poses_dir>/<system_id>/.
        ligand_sdf = Path(pose_path).parent / f"{ligand_id}.sdf"
        if not ligand_sdf.exists():
            return task, None, f"missing reference ligand {ligand_sdf}"
        dtype = {
            "VINA": "vina",
            "SMINA": "smina",
            "GPDOCK": "gpdock",
            "METALLODOCK": "metallodock",
        }[method]
        with contextlib.redirect_stdout(io.StringIO()):
            df = docking_utils.get_docking_data(
                str(pose_path),
                str(ligand_sdf),
                [m[1] for m in metals],
                docking_type=dtype,
            )
        if df.empty:
            return task, None, "no poses parsed"
        df["system_id"] = system_id
        df["ligand_id"] = ligand_id
        return task, df, None
    except Exception:
        return task, None, traceback.format_exc(limit=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses_dir",
        type=Path,
        required=True,
        help="Folder with one <system_id>/ subfolder of saved unseen-system poses.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for system_dir in sorted(p for p in args.poses_dir.iterdir() if p.is_dir()):
        names = os.listdir(system_dir)
        for method, stem_t, pattern in SPECS:
            rx = re.compile(pattern)
            for name in names:
                m = rx.match(name)
                if not m:
                    continue
                if (system_dir / name).stat().st_size == 0:
                    continue
                rid, lig = int(m.group(1)), m.group(2)
                tasks.append(
                    (
                        stem_t.format(rid=rid),
                        method,
                        system_dir.name,
                        lig,
                        system_dir / name,
                    )
                )
    print(f"{len(tasks)} pose files to reparse", flush=True)

    grouped: dict[str, list[pd.DataFrame]] = {}
    failures = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init, initargs=(str(METAL_JSON),)
    ) as ex:
        futures = [ex.submit(_work, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            task, df, err = fut.result()
            if err is None:
                grouped.setdefault(task[0], []).append(df)
            else:
                failures.append(
                    {
                        "csv": task[0],
                        "method": task[1],
                        "system_id": task[2],
                        "ligand_id": task[3],
                        "pose_path": str(task[4]),
                        "error": err,
                    }
                )
            if i % 200 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done, failures={len(failures)}", flush=True)

    pd.DataFrame(failures).to_csv(args.output_dir / "parse_failures.csv", index=False)
    for stem, frames in sorted(grouped.items()):
        out = pd.concat(frames, ignore_index=True).sort_values(
            ["system_id", "ligand_id", "pose"], kind="stable"
        )
        out.to_csv(args.output_dir / f"{stem}.csv", index=False)
        n_sys = out[["system_id", "ligand_id"]].drop_duplicates().shape[0]
        n_nan = (
            out.groupby(["system_id", "ligand_id"])["close_rmsd_3.5_hetero"]
            .apply(lambda s: s.isna().all())
            .sum()
        )
        print(
            f"Wrote {stem}.csv: {len(out)} poses, {n_sys} systems, {n_nan} all-NaN MBG"
        )


if __name__ == "__main__":
    main()
