"""Check loading of the committed and explicitly given prediction cutoffs."""

import json
from pathlib import Path

import pytest

from model.prediction_cutoffs import cutoff_map_tag, load_prediction_cutoffs


def test_default_reads_committed_validation_cutoffs():
    path = Path(__file__).resolve().parents[1] / "model/params/pred_cutoffs.json"
    committed = json.loads(path.read_text())
    expected = {method: committed[method] for method in ("VINA", "SMINA", "GNINA")}
    assert load_prediction_cutoffs() == expected


def test_explicit_cutoff_file_ignores_provenance_metadata(tmp_path):
    path = tmp_path / "cutoffs.json"
    path.write_text(
        json.dumps({"VINA": 0.65, "SMINA": 0.9, "GNINA": 1.1, "rule": "fixed"})
    )
    result = load_prediction_cutoffs(path)
    assert (
        cutoff_map_tag(result, ["VINA", "SMINA", "GNINA"])
        == "VINA-0p65_SMINA-0p9_GNINA-1p1"
    )
    assert cutoff_map_tag({"VINA": 0.8, "GNINA": 0.8}, ["VINA", "GNINA"]) == "0p8"


@pytest.mark.parametrize("value", [None, True, -1, float("inf")])
def test_invalid_cutoffs_fail_instead_of_falling_back(tmp_path, value):
    path = tmp_path / "cutoffs.json"
    path.write_text(json.dumps({"VINA": value, "SMINA": 0.8, "GNINA": 0.85}))
    with pytest.raises(ValueError):
        load_prediction_cutoffs(path)
