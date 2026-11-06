"""A stray empty delimiter block cannot discard valid native poses."""

from pathlib import Path

import numpy as np
import pytest

from chelate.poses import read_poses

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["VINA", "SMINA"])
def test_empty_pdbqt_blocks_preserve_all_real_poses_and_scores(tmp_path, method):
    if method == "SMINA":
        source = ROOT / "model/example/8j3c__1__2.A__2.B_2.C/SMINA_crossdocked.pdbqt"
    else:
        source = (
            ROOT
            / "docking/example_output/2pjc__3__1.C__1.H_1.I/VINA_redocked_FH_id=1_exhaust=32_numModes=100_eRange=20_lig=1.I.pdbqt"
        )
    groups = source.read_text().split("ENDMDL")[:2]
    first, second = [group + "ENDMDL\n" for group in groups]
    clean = tmp_path / "clean.pdbqt"
    clean.write_text(first + second)
    expected, scores = read_poses(str(clean), method)
    dirty = tmp_path / "empty_blocks.pdbqt"
    dirty.write_text(first + " 2\nENDMDL\n" + second + "MODEL 999\nENDMDL\n")
    original = dirty.read_bytes()
    with pytest.warns(UserWarning, match="empty PDBQT model block"):
        actual, actual_scores = read_poses(str(dirty), method)
    assert len(actual) == len(expected) == 2
    np.testing.assert_array_equal(actual_scores, scores)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            observed.GetConformer().GetPositions(),
            reference.GetConformer().GetPositions(),
        )
    assert dirty.read_bytes() == original
