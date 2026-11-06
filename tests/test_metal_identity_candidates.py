"""The probe ranks ten metals; its zero-metal control is diagnostic only."""

import pandas as pd
import pytest

from model import predict_metal_identity
from model.predict_metal_identity import (
    _summarize_system_predictions,
    candidate_protocol,
)


def test_probe_primary_chance_is_one_in_ten():
    protocol = candidate_protocol()
    assert protocol["ranked_candidate_count"] == 10
    assert len(set(protocol["ranked_metals"])) == 10
    assert protocol["uniform_top1_chance"] == 0.1
    assert protocol["ALL_ZERO_is_ranked_in_primary_metric"] is False


def test_zero_control_cannot_win_the_primary_metal_ranking():
    candidates = candidate_protocol()["ranked_metals"]
    rows = [
        {
            "system_id": "system",
            "ligand_id": "1.X",
            "true_metal": "Zn",
            "candidate_metal": metal,
            "pred_mbg_rmsd": 0.2 if metal == "Zn" else 1.0,
        }
        for metal in candidates
    ]
    rows.append(
        {
            "system_id": "system",
            "ligand_id": "1.X",
            "true_metal": "Zn",
            "candidate_metal": "ALL_ZERO",
            "pred_mbg_rmsd": 0.0,
        }
    )
    summary = _summarize_system_predictions(pd.DataFrame(rows))
    assert summary.loc[0, "correct"] == 1
    assert summary.loc[0, "true_rank"] == 1


def test_empty_cached_ligand_is_rejected_without_overwriting(tmp_path, monkeypatch):
    ligand = tmp_path / "1.X.sdf"
    cached = tmp_path / "1.X_crystal.pdbqt"
    cached.write_text("")

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("An existing artifact must be preserved for diagnosis")

    monkeypatch.setattr(predict_metal_identity.subprocess, "run", unexpected_conversion)
    with pytest.raises(RuntimeError, match="contains no atoms"):
        predict_metal_identity._ensure_ligand_crystal_pdbqt(ligand, "1.X")
    assert cached.read_bytes() == b""


def test_zero_exit_conversion_with_empty_output_is_rejected(tmp_path, monkeypatch):
    ligand = tmp_path / "1.X.sdf"
    cached = tmp_path / "1.X_crystal.pdbqt"

    def empty_conversion(*args, **kwargs):
        cached.write_text("REMARK no molecule converted\n")
        return predict_metal_identity.subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(predict_metal_identity.subprocess, "run", empty_conversion)
    with pytest.raises(RuntimeError, match="contains no atoms"):
        predict_metal_identity._ensure_ligand_crystal_pdbqt(ligand, "1.X")


def test_valid_cached_ligand_is_reused(tmp_path):
    ligand = tmp_path / "1.X.sdf"
    cached = tmp_path / "1.X_crystal.pdbqt"
    cached.write_text("ATOM      1  C   LIG A   1       0.000   0.000   0.000\n")
    assert predict_metal_identity._ensure_ligand_crystal_pdbqt(ligand, "1.X") == cached
