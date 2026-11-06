"""Export the four molecular examples used in Figure 5.

For each case in ``current_cases.json`` the exporter writes the cross-docking receptor,
the crystal ligand, the docking program's top-ranked pose, CHELATE's pick and the best
sampled pose, plus ``case.json`` with their MBG and whole-molecule RMSDs and
predictions. The files are written outside Git; pass ``--output-dir`` to choose where.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from chelate import predict_poses

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from model.prediction_cutoffs import DEFAULT_PATH, cutoff_tag, load_prediction_cutoffs

DEFAULT_SYSTEMS_ROOT = (
    Path(os.environ.get("SYSTEMS_ROOT", REPO_ROOT / "docking")) / "test_systems"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".runtime" / "case_studies_fig5"
DEFAULT_CASE_MANIFEST = Path(__file__).with_name("current_cases.json")
DEFAULT_METALS = (
    REPO_ROOT
    / "data_curation/cached_data/metal_ligand_data_distTresh=4.0_mwThresh=750.0_goodApoThresh=70.0_resplit.json"
)

RUN_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems-root", type=Path, default=DEFAULT_SYSTEMS_ROOT)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pred-cutoffs", type=Path, nargs="?", const=DEFAULT_PATH)
    parser.add_argument("--run-id", type=int, choices=(1, 2, 3), default=RUN_ID)
    parser.add_argument(
        "--pymol-settings",
        type=Path,
        default=None,
        help="Optional PyMOL settings script copied once beside the four case folders",
    )
    return parser.parse_args()


def docking_filename(method: str, ligand_id: str, run_id: int) -> str:
    if method == "GNINA":
        return (
            f"GNINA_apo_docked_FH_id={run_id}_exhaust=32_numModes=100_"
            f"prepPH=7.4_lig={ligand_id}.sdf.gz"
        )
    return (
        f"VINA_apo_docked_FH_id={run_id}_exhaust=32_numModes=100_"
        f"eRange=20_lig={ligand_id}.sdf"
    )


def resolve_docking_file(
    source: Path, method: str, ligand_id: str, run_id: int
) -> Path:
    """Find the pose file; Vina's energy range may be written as ``20`` or ``20.0``."""
    path = source / docking_filename(method, ligand_id, run_id)
    if not path.is_file() and method == "VINA":
        path = source / path.name.replace("eRange=20_", "eRange=20.0_")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def metric_record(row: pd.Series) -> dict[str, int | float]:
    return {
        "pose": int(row["pose"]),
        "mbg_rmsd_A": round(float(row["reference_mbg_rmsd"]), 3),
        "whole_molecule_rmsd_A": round(float(row["reference_rmsd"]), 3),
        "predicted_mbg_rmsd_A": round(float(row["chelate_prediction"]), 3),
    }


def summarize_run(
    receptor: Path,
    crystal: Path,
    poses: Path,
    method: str,
    pred_cutoff: float | None = None,
    reference_metal_coords: list | None = None,
) -> tuple[object, dict]:
    if pred_cutoff is None:
        pred_cutoff = load_prediction_cutoffs()[method]
    result = predict_poses(receptor=receptor, poses=poses, method=method)
    result.add_reference_rmsd(crystal)
    result.add_reference_mbg_rmsd(
        crystal, reference_metal_coords=reference_metal_coords
    )
    baseline = result.baseline_top1()
    best = result.features.loc[result.features["reference_mbg_rmsd"].idxmin()]
    survivors = result.features[result.features["chelate_prediction"] <= pred_cutoff]
    chosen = result.top1(cutoff=pred_cutoff)
    summary = {
        "number_of_poses": len(result.features),
        "number_passing_cutoff": len(survivors),
        "pred_cutoff": pred_cutoff,
        "top1_original": metric_record(baseline),
        "best_available_pose": metric_record(best),
        "chelate_top1": metric_record(chosen) if chosen is not None else None,
        "minimum_predicted_mbg_rmsd_A": float(
            result.features["chelate_prediction"].min()
        ),
    }
    return result, summary


def copy_receptor_with_provenance(source: Path, destination: Path) -> None:
    """Keep validated conversion connectivity authoritative after exporting a PDB."""
    proof = source.with_suffix(source.suffix + ".provenance.json")
    if proof.is_file():
        record = json.loads(proof.read_text())
        if (
            record.get("schema") != "chelate-receptor-pdb-v1"
            or record.get("pdb_sha256")
            != hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            raise ValueError("Receptor conversion provenance does not match its PDB")
    shutil.copy2(source, destination)
    if proof.is_file():
        shutil.copy2(
            proof, destination.with_suffix(destination.suffix + ".provenance.json")
        )


def export_case(
    case: dict,
    systems_root: Path,
    output_root: Path,
    pred_cutoff: float | None = None,
    run_id: int = RUN_ID,
) -> dict:
    if pred_cutoff is None:
        pred_cutoff = load_prediction_cutoffs()[case["method"]]
    source = systems_root / case["system_id"]
    folder = f"{case['folder']}_predCutoff={cutoff_tag(pred_cutoff)}"
    if run_id != RUN_ID:
        folder += f"_run={run_id}"
    destination = output_root / folder
    destination.mkdir(parents=True, exist_ok=True)

    apo = f"{case['apo_pdb_id'].lower()}_{case['apo_chain']}"
    receptor_source = source / (
        f"{case['system_id']}__{apo}_merged_{case['ligand_id']}.pdb"
    )
    crystal_source = source / f"{case['ligand_id']}.sdf"
    poses_source = resolve_docking_file(
        source, case["method"], case["ligand_id"], run_id
    )
    for path in (receptor_source, crystal_source, poses_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    receptor = destination / "receptor.pdb"
    crystal = destination / "crystal.sdf"
    copy_receptor_with_provenance(receptor_source, receptor)
    shutil.copy2(crystal_source, crystal)

    with DEFAULT_METALS.open() as handle:
        reference_metals = json.load(handle)[case["system_id"]]["metals"]
    reference_metal_coords = [coordinate for _, coordinate in reference_metals]
    result, run_summary = summarize_run(
        receptor,
        crystal,
        poses_source,
        case["method"],
        pred_cutoff,
        reference_metal_coords,
    )

    baseline = result.baseline_top1()
    chelate = result.top1(cutoff=pred_cutoff)
    best = result.features.loc[result.features["reference_mbg_rmsd"].idxmin()]

    result.write_sdf(destination / "top1_original.sdf", baseline)
    if chelate is not None:
        result.write_sdf(destination / "chelate_top1.sdf", chelate)
    else:
        (destination / "chelate_top1.sdf").unlink(missing_ok=True)
    result.write_sdf(destination / "best_available_pose.sdf", best)

    files = {
        "crystal.sdf": {
            "mbg_rmsd_A": 0.0,
            "whole_molecule_rmsd_A": 0.0,
            "predicted_mbg_rmsd_A": None,
            "prediction_note": (
                "Not applicable: the crystal structure has no docking score or "
                "docking rank, both of which are model inputs."
            ),
        },
        "top1_original.sdf": metric_record(baseline),
        "chelate_top1.sdf": metric_record(chelate) if chelate is not None else None,
        "best_available_pose.sdf": metric_record(best),
    }
    metadata = {
        "reference_mbg_definition": "Original holo metal coordinates, N/O/S within 3.5 A, matching benchmark labels",
        "docking_program": case["method"],
        "docking_run_id": run_id,
        "docking_pose_file": poses_source.name,
        "holo_pdb_id": case["pdb_id"],
        "apo_pdb_id": case["apo_pdb_id"],
        "apo_chain": case["apo_chain"],
        "system_id": case["system_id"],
        "ligand_id": case["ligand_id"],
        "metal": case["metal"],
        "resolution_A": case["resolution_A"],
        "protein_name": case["protein_name"],
        "pred_cutoff": pred_cutoff,
        "number_of_poses": run_summary["number_of_poses"],
        "number_passing_cutoff": run_summary["number_passing_cutoff"],
        "files": files,
    }
    for key in ("ligand_ccd_code", "metadata_sources"):
        if key in case:
            metadata[key] = case[key]
    # Record each GNINA repeat at the fixed cutoff, including changes in the case effect.
    if case["method"] == "GNINA":
        metadata["all_runs"] = []
        for repeat in (1, 2, 3):
            if repeat == run_id:
                summary = run_summary
            else:
                run_poses = resolve_docking_file(
                    source, case["method"], case["ligand_id"], repeat
                )
                _, summary = summarize_run(
                    receptor,
                    crystal,
                    run_poses,
                    case["method"],
                    pred_cutoff,
                    reference_metal_coords,
                )
            metadata["all_runs"].append({"run_id": repeat, **summary})

    with (destination / "case.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pymol_settings is not None:
        shutil.copy2(args.pymol_settings, args.output_dir / "pymol_settings.py")

    cutoffs = load_prediction_cutoffs(args.pred_cutoffs)
    for case in json.loads(args.case_manifest.read_text())["cases"]:
        metadata = export_case(
            case,
            args.systems_root,
            args.output_dir,
            cutoffs[case["method"]],
            args.run_id,
        )
        files = metadata["files"]
        print(
            case["pdb_id"],
            f"poses={metadata['number_of_poses']}",
            f"cutoff={metadata['pred_cutoff']}",
            f"pass={metadata['number_passing_cutoff']}",
            f"top1={files['top1_original.sdf']['mbg_rmsd_A']:.3f}",
            f"best={files['best_available_pose.sdf']['mbg_rmsd_A']:.3f}",
        )


if __name__ == "__main__":
    main()
