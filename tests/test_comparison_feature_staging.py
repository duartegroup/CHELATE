"""Comparison staging must retain numeric text and catch duplicate pose identities."""

import pytest

from model.stage_comparison_features import copy_pose_rows


def test_staging_preserves_numeric_precision_and_quoted_fields(tmp_path):
    header = b"system_id,ligand_id,pose,value,description\r\n"
    keep = b'example,1.NA,2,1.234567890123456789e-04,"metal, ligand"\r\n'
    discard = b"other,1.A,1,-9.0000000000000000,other\r\n"
    source, output = tmp_path / "source.csv", tmp_path / "output.csv"
    source.write_bytes(header + discard + keep)
    copy_pose_rows(source, output, {("example", "1.NA", 2)})
    assert output.read_bytes() == header + keep
    assert source.read_bytes() == header + discard + keep


def test_duplicate_selected_poses_are_rejected(tmp_path):
    source, output = tmp_path / "source.csv", tmp_path / "output.csv"
    source.write_text("system_id,ligand_id,pose\nexample,1.A,1\nexample,1.A,1\n")
    with pytest.raises(ValueError, match="Duplicate selected pose"):
        copy_pose_rows(source, output, {("example", "1.A", 1)})
