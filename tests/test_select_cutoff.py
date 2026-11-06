"""Check validation-only cutoff rules and retained-system metric semantics."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model import model_utils, select_cutoff


def test_exact_grid_matches_evaluator_source():
    source = Path(select_cutoff.__file__).with_name("test_eval.py")
    assignments = [
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "rmsd_cutoffs"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == select_cutoff.CUTOFF_GRID


def test_high_precision_below_half_coverage_is_ineligible():
    curve = pd.DataFrame(
        {
            "pred_cutoff": [0.2, 0.5, 1.0],
            "retained_fraction": [0.25, 0.5, 1.0],
            "n_successful": [1, 1, 1],
            "n_retained": [1, 2, 4],
        }
    )
    assert select_cutoff.choose_cutoff(curve)["pred_cutoff"] == 0.5


def test_exact_precision_ties_choose_larger_cutoff_even_when_unsorted():
    curve = pd.DataFrame(
        {
            "pred_cutoff": [0.9, 0.5, 0.7],
            "retained_fraction": [0.75, 0.5, 0.625],
            "n_successful": [3, 2, 2],
            "n_retained": [6, 4, 5],
        }
    )
    assert select_cutoff.choose_cutoff(curve)["pred_cutoff"] == 0.9


def test_no_feasible_cutoff_requires_a_user_decision():
    curve = pd.DataFrame({"retained_fraction": [0.49]})
    with pytest.raises(ValueError, match="ask the user"):
        select_cutoff.choose_cutoff(curve)


@pytest.mark.parametrize("method", ["VINA", "GNINA"])
def test_filter_then_native_ranking_matches_project_metrics(method):
    score = "CNNscore" if method == "GNINA" else "docking_score"
    values = np.array([-10.0, -9.0, -10.0, -9.0, -10.0])
    if method == "GNINA":
        values = -values
    frame = pd.DataFrame(
        {
            "system_id": ["A", "A", "B", "B", "C"],
            "ligand_id": "L",
            select_cutoff.TARGET: [3.0, 0.5, 0.5, 3.0, 0.5],
            score: values,
            "system_id__ligand_id": ["A__L", "A__L", "B__L", "B__L", "C__L"],
        }
    )
    prediction = np.array([0.5, 0.4, 0.2, 0.9, 0.9])
    row = select_cutoff.validation_curve(frame, prediction, method, [0.6]).iloc[0]
    retained = frame.loc[prediction <= 0.6]
    assert row.n_systems == 3 and row.n_retained == 2 and row.n_successful == 1
    assert (
        row.sr_retained
        == model_utils.calculate_success_rate(
            retained,
            rerank_col=score,
            sr_col=select_cutoff.TARGET,
            sr_threshold=1.0,
            top_n=[1],
        )[1]
    )
    assert (
        row.sr_all_systems
        == model_utils.calculate_success_rate_static(
            retained,
            all_ids=frame.system_id__ligand_id.unique().tolist(),
            rerank_col=score,
            sr_col=select_cutoff.TARGET,
            sr_threshold=1.0,
            top_n=[1],
        )[1]
    )


def test_train_only_fit_uses_saved_weights_and_seed(monkeypatch):
    calls = {}

    class Regressor:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit(self, x, y, sample_weight):
            calls.update(rows=len(x), targets=y.tolist(), weights=sample_weight)

    monkeypatch.setattr(select_cutoff, "CatBoostRegressor", Regressor)
    monkeypatch.setattr(model_utils, "get_feature_list", lambda **_: ["feature"])
    train = pd.DataFrame({"feature": [1.0, 2.0], select_cutoff.TARGET: [0.5, 2.0]})
    parameters = {
        "iterations": 10,
        "depth": 4,
        "learning_rate": 0.1,
        "smooth_k": 2.0,
        "smooth_p": 1.5,
    }
    select_cutoff.fit_train_only(train, "VINA", parameters, "CPU", 2)
    assert calls["rows"] == 2 and calls["targets"] == [0.5, 2.0]
    assert calls["random_seed"] == 42 and calls["loss_function"] == "RMSE"
    np.testing.assert_allclose(
        calls["weights"], np.exp(-2 * np.array([0.5, 2.0]) ** 1.5)
    )


def test_selected_method_reads_only_its_training_inputs(tmp_path, monkeypatch):
    import argparse
    import json

    root = tmp_path / "workspace"
    params = root / "model/params"
    params.mkdir(parents=True)
    (root / "chelate/params").mkdir(parents=True)
    (root / "figures/model/csv_files").mkdir(parents=True)
    features = root / "docking/features"
    features.mkdir(parents=True)
    for split in ("train", "val"):
        (features / f"features_{split}_GNINA_redocked_FH_id=1.csv").write_text(
            "fixture input\n"
        )
    (params / "GNINA_rmsd=1.0_best_hyperparameters.json").write_text("{}\n")
    monkeypatch.setattr(select_cutoff, "MODEL_DIR", root / "model")
    monkeypatch.setattr(model_utils, "seed", lambda: None)
    monkeypatch.setattr(model_utils, "get_feature_list", lambda **_: ["feature"])
    loaded = []

    def load(method, target, features_dir):
        loaded.append(method)
        assert method == "GNINA"
        frames = []
        for split, pdb in [("train", "1aaa"), ("val", "2bbb")]:
            frames.append(
                pd.DataFrame(
                    {
                        "system_id": [pdb, pdb],
                        "ligand_id": ["L", "L"],
                        "system_id__ligand_id": [pdb + "__L"] * 2,
                        "split": [split, split],
                        "pose_rank": [1, 2],
                        "feature": [1.0, 2.0],
                        "CNNscore": [0.9, 0.8],
                        target: [0.5, 2.0],
                    }
                )
            )
        return frames

    class FittedModel:
        def save_model(self, path):
            Path(path).write_text("fitted on train only\n")

        def predict(self, frame, thread_count):
            return np.array([0.8, 1.0])

    monkeypatch.setattr(model_utils, "load_train_val_for_method", load)
    monkeypatch.setattr(select_cutoff, "fit_train_only", lambda *args: FittedModel())
    select_cutoff.run(
        argparse.Namespace(
            methods=["GNINA"],
            runtime_dir=tmp_path / "fits",
            features_dir=features,
            task_type="CPU",
            threads=1,
        )
    )
    assert loaded == ["GNINA"]
    record = json.loads((params / "pred_cutoffs.json").read_text())
    assert set(record["validation"]) == {"GNINA"}
    assert record["test_used"] is False
    curve = pd.read_csv(
        root / "figures/model/csv_files/cutoff_selection_validation.csv"
    )
    assert set(curve.method) == {"GNINA"}
