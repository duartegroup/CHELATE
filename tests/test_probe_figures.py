"""Probe plots must display the full ranked candidate space."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from chelate.constants import EXPLICIT_METALS
from figures.metal_identity.make_metal_identity_figures import probe_confusion_table


def test_candidate_absent_from_true_labels_stays_in_the_heatmap():
    selected = pd.DataFrame(
        [
            {
                "system_id": "a",
                "ligand_id": "1.B",
                "true_metal": "Zn",
                "candidate_metal": "Fe",
            },
            {
                "system_id": "b",
                "ligand_id": "1.B",
                "true_metal": "Zn",
                "candidate_metal": "Zn",
            },
        ]
    )
    table = probe_confusion_table(selected, ["Zn"])
    assert set(table.columns) == set(EXPLICIT_METALS)
    assert table.index.tolist() == ["Zn"]
    assert table.loc["Zn", "Fe"] == 50
    assert table.loc["Zn", "Zn"] == 50
    assert table.loc["Zn"].sum() == 100


def test_top_three_uses_pairs_as_denominator_and_omits_the_zero_control():
    selected = pd.DataFrame(
        [
            {
                "system_id": "a",
                "ligand_id": "1.B",
                "true_metal": "Zn",
                "candidate_metal": metal,
            }
            for metal in ["Fe", "Zn", "Mg", "ALL_ZERO"]
        ]
    )
    table = probe_confusion_table(selected, ["Zn"])
    assert "ALL_ZERO" not in table.columns
    assert table.loc["Zn"].sum() == pytest.approx(300)


def test_probe_table_import_does_not_require_optional_seaborn(monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)
    path = (
        Path(__file__).resolve().parents[1]
        / "figures/metal_identity/make_metal_identity_figures.py"
    )
    spec = importlib.util.spec_from_file_location("probe_without_seaborn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = pd.DataFrame(
        [
            {
                "system_id": "a",
                "ligand_id": "1.B",
                "true_metal": "Zn",
                "candidate_metal": "Zn",
            }
        ]
    )
    assert module.probe_confusion_table(selected, ["Zn"]).loc["Zn", "Zn"] == 100
