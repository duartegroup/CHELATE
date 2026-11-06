"""Run the fixed-hyperparameter training-fraction sweep without Optuna tuning.

The committed validation-selected per-engine cutoffs are used by default. Pass
--pred-cutoffs PATH to read another explicit cutoff file. The output filename and
CSV record the cutoffs actually used. Final committed model files are never written.

Fraction models are trained on the GPU by default, like the final models. The
100% point loads the committed final model without fitting. CPU training of the
smaller fractions uses different CatBoost defaults and gives different models.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from model.model_params import FEATURES_DIR
from model.prediction_cutoffs import DEFAULT_PATH, load_prediction_cutoffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pred-cutoffs", type=Path, nargs="?", const=DEFAULT_PATH)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--features-dir", type=Path, default=Path(FEATURES_DIR))
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    args = parser.parse_args()
    from model import train_model as tm

    tm._TASK_TYPE = args.task_type
    cutoffs = load_prediction_cutoffs(args.pred_cutoffs)
    t0 = time.time()
    tm.run_train_fraction_sweep(
        output_folder=str(args.output_dir),
        methods=["VINA", "SMINA", "GNINA"],
        target="close_rmsd_3.5_hetero",
        opt_rmsd_cutoff=1.0,
        model_dir=str(REPO / "model" / "params"),
        fractions=tm._parse_fraction_grid("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"),
        pred_cutoff=cutoffs,
        mbg_sr_threshold=1.0,
        keep_best_pose=False,
        subset_seed=tm.model_utils.SEED,
        threads=args.threads,
        features_dir=str(args.features_dir),
    )
    print(f"[DONE] sweep took {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
