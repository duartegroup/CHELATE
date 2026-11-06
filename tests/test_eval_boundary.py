"""Prediction-quality availability uses the same inclusive RMSD boundary as success."""

import numpy as np
import pandas as pd

from model.test_eval import collect_prediction_quality_inputs


class ConstantPredictor:
    def predict(self, frame):
        return np.full(len(frame), 0.5)


def test_exact_one_angstrom_pose_is_successful_in_quality_tables():
    target = "close_rmsd_3.5_hetero"
    frame = pd.DataFrame(
        {
            "system_id": ["boundary", "inaccurate"],
            "ligand_id": ["1.L", "1.L"],
            "pose": [1, 1],
            "docking_score": [-8.0, -7.0],
            "feature": [0.0, 0.0],
            target: [1.0, 2.0],
        }
    )
    _, _, table = collect_prediction_quality_inputs(
        ConstantPredictor(),
        target,
        [frame],
        ["feature"],
        "VINA",
        "redocked",
        [0.75],
        False,
    )
    row = table.iloc[0]
    assert row["successful_systems_total"] == 1
    assert row["successful_systems_retained"] == 1
    assert row["frac_retained_with_success"] == 1.0
    assert row["theoretical_sr_top1"] == 0.5
