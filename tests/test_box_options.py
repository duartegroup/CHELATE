"""Explicit box arguments preserve GNINA's default rotation expansion."""

import pytest

from docking.box_options import fixed_box_options

BOX = {
    "center_x": 1,
    "center_y": 2,
    "center_z": 3,
    "size_x": 8,
    "size_y": 10,
    "size_z": 12,
}


@pytest.mark.parametrize("method", ["VINA", "SMINA", "GNINA"])
def test_engine_receives_reference_box_with_default_rotation_handling(method):
    options = fixed_box_options(method, BOX)
    values = dict(zip(options[::2], options[1::2], strict=True))
    assert {key: float(values["--" + key]) for key in BOX} == BOX
    assert "--autobox_ligand" not in values
    assert "--autobox_extend" not in values
    assert "--autobox_add" not in values


def test_invalid_method_does_not_get_a_partially_defined_box():
    with pytest.raises(ValueError, match="Unknown docking method"):
        fixed_box_options("other", BOX)
