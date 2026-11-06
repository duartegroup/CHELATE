"""The public sweep command must use the explicitly staged comparison inputs."""

import sys
from types import SimpleNamespace

from figures.model.scripts import run_sweep


def test_sweep_cli_routes_the_frozen_feature_directory(tmp_path, monkeypatch):
    calls = []
    trainer = SimpleNamespace(
        _TASK_TYPE=None,
        _parse_fraction_grid=lambda grid: [float(x) for x in grid.split(",")],
        run_train_fraction_sweep=lambda **kwargs: calls.append(kwargs),
        model_utils=SimpleNamespace(SEED=42),
    )
    import model

    monkeypatch.setitem(sys.modules, "model.train_model", trainer)
    monkeypatch.setattr(model, "train_model", trainer, raising=False)
    features = tmp_path / "comparison_features"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sweep.py", str(tmp_path / "sweep"), "--features-dir", str(features)],
    )
    run_sweep.main()
    assert len(calls) == 1
    assert calls[0]["features_dir"] == str(features)
    assert calls[0]["fractions"] == [i / 10 for i in range(1, 11)]
    assert calls[0]["subset_seed"] == 42
    # Like the final models, so the 100% point reproduces them.
    assert trainer._TASK_TYPE == "GPU"
