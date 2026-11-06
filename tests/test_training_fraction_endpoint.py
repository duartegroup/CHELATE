"""The full-data sweep endpoint must evaluate the frozen final model."""

import hashlib

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from model import train_model as tm


def test_full_fraction_loads_final_model_without_fitting(tmp_path, monkeypatch):
    x = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    target = [0.1, 0.3, 1.2, 2.1]
    expected = CatBoostRegressor(
        iterations=2,
        depth=1,
        verbose=False,
        thread_count=1,
        allow_writing_files=False,
    )
    expected.fit(x, target)
    path = tmp_path / "GNINA_rmsd=1.0_best_model.cbm"
    expected.save_model(str(path))
    fingerprint = hashlib.sha256(path.read_bytes()).digest()
    data = x.assign(system_id=["a", "a", "b", "c"], target=target)
    monkeypatch.setattr(
        tm.model_utils,
        "load_train_val_for_method",
        lambda **kwargs: (data.iloc[:3].copy(), data.iloc[3:].copy()),
    )
    monkeypatch.setattr(
        tm.model_utils, "get_feature_list", lambda **kwargs: ["feature"]
    )

    def forbid_fit(*args, **kwargs):
        raise AssertionError("The 100% point must use the committed model")

    monkeypatch.setattr(CatBoostRegressor, "fit", forbid_fit)
    monkeypatch.setattr(tm, "_load_saved_hyperparams", forbid_fit)
    actual, features, meta = tm._train_fraction_model_from_saved_hparams(
        method="GNINA",
        target="target",
        opt_rmsd_cutoff=1.0,
        model_dir=str(tmp_path),
        fraction=1.0,
        subset_seed=42,
        features_dir=str(tmp_path),
        threads=1,
    )
    np.testing.assert_array_equal(actual.predict(x), expected.predict(x))
    assert features == ["feature"]
    assert meta == {
        "fraction": 1.0,
        "systems_used": 3,
        "systems_total": 3,
        "rows_used": 4,
        "rows_total": 4,
    }
    assert hashlib.sha256(path.read_bytes()).digest() == fingerprint
