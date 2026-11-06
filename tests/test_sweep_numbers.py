"""The paper's sweep numbers use the committed cutoff and all three docking repeats."""

import pandas as pd
import pytest

from figures.model.scripts.sweep_numbers import render_sweep


def test_sweep_rejects_other_cutoffs_and_missing_repeats(tmp_path):
    path = tmp_path / "mbg_sr_vs_training_fraction_predCutoff=VINA-0p85.csv"
    frame = pd.DataFrame(
        {
            "method": ["VINA"] * 3,
            "dock_key": ["redocked"] * 3,
            "fraction": [1.0] * 3,
            "pred_cutoff": [0.85] * 3,
            "mbg_sr_threshold": [1.0] * 3,
            "rep_id": [1, 2, 3],
            "systems_used": [100] * 3,
            "systems_total": [100] * 3,
            "systems_count": [10] * 3,
            "mbg_sr_top1": [0.3, 0.4, 0.5],
        }
    )
    frame.to_csv(path, index=False)
    assert "40.0 ± 10.0" in render_sweep(tmp_path, {"VINA": 0.85})
    with pytest.raises(ValueError, match="committed cutoff"):
        render_sweep(tmp_path, {"VINA": 0.8})
    frame.iloc[:2].to_csv(path, index=False)
    with pytest.raises(ValueError, match="three distinct runs"):
        render_sweep(tmp_path, {"VINA": 0.85})
