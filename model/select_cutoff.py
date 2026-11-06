"""Choose prediction cutoffs on validation using fresh train-only fits.

No test feature table, test prediction or train-plus-validation final model is
read here. Each engine uses its saved hyperparameters, smooth weights and seed
42. Retained poses keep the docking engine's ranking, as in test_eval without
keep_best_pose. Final committed models are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from model import model_utils
from model.model_params import FEATURES_DIR, MODEL_DIR, SEED

TARGET = "close_rmsd_3.5_hetero"
METHODS = ("VINA", "SMINA", "GNINA")
# Exactly test_eval.main's prediction-cutoff grid; a test checks both literals.
CUTOFF_GRID = [
    0.2,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
]
RULE = (
    "Highest validation top-1 MBG success among retained systems, subject to "
    "retained fraction >= 0.5; exact ties go to the larger prediction cutoff."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_curve(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    method: str,
    cutoffs: list[float] = CUTOFF_GRID,
) -> pd.DataFrame:
    """Use the retained-system evaluator's filters, native ranks and denominators."""
    score = "CNNscore" if method == "GNINA" else "docking_score"
    data = frame.copy()
    data["pred"] = np.asarray(predictions)
    if not np.isfinite(data["pred"]).all():
        raise ValueError("Validation predictions must be finite")
    for column in dict.fromkeys(["docking_score", score]):
        if column in data:
            data = data.loc[np.isfinite(data[column])].copy()
    data = data.loc[data[TARGET].notna()].copy()
    data["system_id__ligand_id"] = data.system_id + "__" + data.ligand_id
    total = data["system_id__ligand_id"].nunique()
    if not total:
        raise ValueError("No eligible validation systems")
    native = model_utils.rerank_poses(data, score)
    baseline = native.loc[native.pose == 1, TARGET].le(1.0).mean()
    rows = []
    for cutoff in cutoffs:
        retained = data.loc[data["pred"] <= cutoff]
        top = model_utils.rerank_poses(retained, score)
        top = top.loc[top.pose == 1]
        count, success = len(top), int(top[TARGET].le(1.0).sum())
        rows.append(
            {
                "method": method,
                "pred_cutoff": float(cutoff),
                "n_systems": int(total),
                "n_retained": count,
                "n_successful": success,
                "retained_fraction": count / total,
                "sr_retained": success / count if count else np.nan,
                "sr_all_systems": success / total,
                "baseline_sr": float(baseline),
            }
        )
    return pd.DataFrame(rows)


def choose_cutoff(curve: pd.DataFrame) -> dict:
    """Apply ``RULE``; fail rather than invent a low-coverage fallback."""
    eligible = curve.loc[curve.retained_fraction >= 0.5]
    if eligible.empty:
        raise ValueError(
            "No cutoff retains at least 50% of validation systems; ask the user."
        )
    records = eligible.to_dict("records")
    return max(
        records,
        key=lambda row: (
            Fraction(int(row["n_successful"]), int(row["n_retained"])),
            row["pred_cutoff"],
        ),
    )


def fit_train_only(
    train: pd.DataFrame, method: str, parameters: dict, task_type: str, threads: int
) -> CatBoostRegressor:
    """Match train_model's saved hyperparameters, smooth weights and random seed."""
    names = model_utils.get_feature_list(docking_program=method)
    weights = np.exp(
        -parameters["smooth_k"]
        * np.abs(train[TARGET].to_numpy()) ** parameters["smooth_p"]
    )
    model = CatBoostRegressor(
        iterations=int(parameters["iterations"]),
        learning_rate=float(parameters["learning_rate"]),
        depth=int(parameters["depth"]),
        random_seed=SEED,
        loss_function="RMSE",
        task_type=task_type,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train[names], train[TARGET], sample_weight=weights)
    return model


def run(args: argparse.Namespace) -> None:
    root = MODEL_DIR.parent
    runtime = args.runtime_dir.resolve()
    if root / "model" / "params" in runtime.parents:
        raise ValueError("Runtime models must be stored outside model/params")
    runtime.mkdir(parents=True, exist_ok=True)
    model_utils.seed()
    curves, chosen, provenance = [], {}, {}
    final_models = {
        str(path.relative_to(root)): sha256(path)
        for directory in [MODEL_DIR / "params", root / "chelate" / "params"]
        for path in directory.glob("*_best_model.cbm")
    }
    (runtime / "protected_final_models.json").write_text(
        json.dumps(final_models, indent=2) + "\n"
    )
    methods = tuple(getattr(args, "methods", METHODS))
    for method in methods:
        parameter_path = (
            MODEL_DIR / "params" / f"{method}_rmsd=1.0_best_hyperparameters.json"
        )
        parameters = json.loads(parameter_path.read_text())
        train, validation = model_utils.load_train_val_for_method(
            method, TARGET, features_dir=str(args.features_dir)
        )
        for split, frame in [("train", train), ("val", validation)]:
            if set(frame.split) != {split}:
                raise ValueError(f"Unexpected {method} {split} split labels")
            if not np.isfinite(frame[TARGET]).all():
                raise ValueError("Non-finite training or validation targets")
        train_pdbs = set(train.system_id.str.split("__").str[0].str.lower())
        validation_pdbs = set(validation.system_id.str.split("__").str[0].str.lower())
        if train_pdbs & validation_pdbs:
            raise ValueError("Training and validation share PDB entries")
        print(f"{method}: fitting train only ({len(train)} poses)", flush=True)
        model = fit_train_only(train, method, parameters, args.task_type, args.threads)
        path = runtime / f"{method}_train_only.cbm"
        model.save_model(str(path))
        names = model_utils.get_feature_list(docking_program=method)
        predictions = model.predict(validation[names], thread_count=args.threads)
        curve = validation_curve(validation, predictions, method)
        selection = choose_cutoff(curve)
        chosen[method] = selection
        curves.append(curve)
        curve.to_csv(runtime / f"{method}_validation_curve.csv", index=False)
        columns = ["system_id", "ligand_id", "pose_rank", TARGET]
        validation[columns].assign(prediction=predictions).to_csv(
            runtime / f"{method}_validation_predictions.csv", index=False
        )
        provenance[method] = {
            "seed": SEED,
            "task_type": args.task_type,
            "train_pairs": int(train.system_id__ligand_id.nunique()),
            "validation_pairs": int(validation.system_id__ligand_id.nunique()),
            "train_only_model_sha256": sha256(path),
            "hyperparameters_sha256": sha256(parameter_path),
            "feature_sha256": {
                split: sha256(
                    args.features_dir
                    / f"features_{split}_{method}_redocked_FH_id=1.csv"
                )
                for split in ["train", "val"]
            },
        }
        (runtime / f"{method}_complete.json").write_text(
            json.dumps(
                {"selected": selection, "provenance": provenance[method]}, indent=2
            )
            + "\n"
        )
        print(f"{method}: {selection}", flush=True)
    for relative, expected in final_models.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Protected final model changed: {relative}")
    curve_path = root / "figures/model/csv_files/cutoff_selection_validation.csv"
    cutoff_path = MODEL_DIR / "params/pred_cutoffs.json"
    if curve_path.exists() or cutoff_path.exists():
        raise FileExistsError(
            "Cutoff outputs already exist; do not overwrite frozen choices"
        )
    pd.concat(curves, ignore_index=True).to_csv(curve_path, index=False)
    record = {method: chosen[method]["pred_cutoff"] for method in methods}
    record.update(
        rule=RULE,
        validation=chosen,
        provenance=provenance,
        test_used=False,
        final_models_changed=False,
    )
    cutoff_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (runtime / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cutoffs": {m: record[m] for m in methods},
                "curve_sha256": sha256(curve_path),
                "cutoffs_sha256": sha256(cutoff_path),
                "test_used": False,
                "final_models_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--features-dir", type=Path, default=Path(FEATURES_DIR))
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
