"""
Description
-----------
This module defines the variables and parameters used in the pose re-ranking pipeline,
mirroring the role of [docking_params.py](../docking/docking_params.py) for the docking
pipeline. Local paths and training parameters are configured here; shared chemical
constants and feature definitions live in the `chelate` package.

Paths
-----
- `MODEL_DIR`: Directory containing the model scripts and their default outputs.
- `DATASET_CSV`: Curated PLINDER dataset CSV (systems, ligands, splits).
- `METAL_LIGAND_DATA_PATH`: Per-system metal/ligand coordinate JSON.
- `FEATURES_DIR`: Directory holding the feature CSVs written by `generate_features.py`.
- `FEATURE_CSV_TEMPLATE`: Filename template of those feature CSVs.
- `DOCK_TYPE_SLUG`: Docking-type labels mapped to run types used in feature filenames.
- `DOCKING_RESULTS_BASE`: Root of the compiled docking results.
- `BASELINE_RESULTS_HETERO_DIR`: Baseline docking results for the test split.

Training / evaluation
---------------------
- `SEED`: Global random seed.
- `SR_THRESHOLDS`: RMSD thresholds (Angstrom) at which success rates are reported.
- `MBG_RMSD_THRESHOLD`: Metal-binding-group RMSD threshold counting a pose as correct.
- `FULL_RMSD_THRESHOLD`: Full-ligand RMSD threshold counting a pose as correct.
- `METAL_COORD_MATCH_TOL`: Tolerance (Angstrom) for matching a predicted metal position
  to a crystal metal position.
- `EXTRA_BASELINE_METHODS`: Docking programs benchmarked in addition to VINA/SMINA/GNINA.
"""

import os
from pathlib import Path

MODEL_DIR: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Curated dataset files (produced by the data_curation pipeline)
# ---------------------------------------------------------------------------
CACHED_DATA_PATH: str = str(MODEL_DIR.parent / "data_curation" / "cached_data")
DATASET_CSV: str = (
    f"{CACHED_DATA_PATH}/final_dataset_distTresh=4.0_mwThresh=750.0_"
    "goodApoThresh=70.0_resplit.csv"
)
METAL_LIGAND_DATA_PATH: str = (
    f"{CACHED_DATA_PATH}/metal_ligand_data_distTresh=4.0_mwThresh=750.0_"
    "goodApoThresh=70.0_resplit.json"
)

# ---------------------------------------------------------------------------
# Docking pipeline outputs consumed by the re-ranking models
# ---------------------------------------------------------------------------
FEATURES_DIR: str = str(MODEL_DIR.parent / "docking" / "features")
# Formatted with split, method and run id, e.g. features_train_VINA_redocked_FH_id=1.csv
FEATURE_CSV_TEMPLATE: str = "features_{split}_{method}_{run_type}_id={run_id}.csv"
DOCK_TYPE_SLUG: dict[str, str] = {
    "redocked": "redocked_FH",
    "crossdocked": "apo_docked_FH",
}
TRAIN_RUN_TYPE: str = "redocked_FH"
TRAIN_RUN_ID: str = "1"
# Docking run ids evaluated on the test split (repeat runs of the same setting).
TEST_RUN_IDS: list[int] = [1, 2, 3]

DOCKING_RESULTS_BASE: str = str(MODEL_DIR.parent / "docking" / "results")
BASELINE_RESULTS_HETERO_DIR: str = os.path.join(DOCKING_RESULTS_BASE, "test")

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
SEED: int = 42

SR_THRESHOLDS: list[float] = [0.5, 1, 1.5, 2]
MBG_RMSD_THRESHOLD: float = 1.0
FULL_RMSD_THRESHOLD: float = 2.0
METAL_COORD_MATCH_TOL: float = 2.0

EXTRA_BASELINE_METHODS: list[str] = ["AUTODOCK", "GPDOCK", "METALLODOCK"]
