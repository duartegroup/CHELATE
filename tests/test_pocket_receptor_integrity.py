"""The pocket exclusion rule uses a fixed distance to the docking box."""

import numpy as np
import pytest

from docking.receptor_integrity import compare_source_atoms, distances_to_docking_box


def source_atom(
    name,
    xyz,
    *,
    chain="protein",
    element="C",
    polymer=True,
    residue="ALA",
    number="1",
    alt=".",
):
    return {
        "name": name,
        "xyz": list(xyz),
        "chain": chain,
        "auth_chain": chain,
        "element": element,
        "polymer": polymer,
        "residue": residue,
        "number": number,
        "insertion": "?",
        "model": "1",
        "alt": alt,
    }


def receptor_atom(xyz, element="C"):
    return {"xyz": list(xyz), "element": element}


BOX = {
    "center_x": 0,
    "center_y": 0,
    "center_z": 0,
    "size_x": 2,
    "size_y": 2,
    "size_z": 2,
}


def audit(source, receptor):
    return compare_source_atoms(
        source, receptor, target_ligand="target", drop_nonpolymer=False, docking_box=BOX
    )


def test_box_distance_is_to_surface_and_uses_euclidean_corners():
    assert distances_to_docking_box(
        [[0, 0, 0], [1, 1, 1], [9, 0, 0], [7, 7, 0]], BOX
    ) == pytest.approx([0, 0, 8, np.sqrt(72)])


@pytest.mark.parametrize(("x", "excluded"), [(8.999, True), (9, True), (9.001, False)])
def test_fixed_eight_angstrom_boundary(x, excluded):
    result = audit(
        [source_atom("CA", (0, 0, 0)), source_atom("CB", (x, 0, 0))],
        [receptor_atom((0, 0, 0))],
    )
    assert result["defective"] is excluded
    assert result["strict_whole_protein_defective"]
    assert result["missing_protein_atoms"] == 1
    assert result["binding_site_missing_atoms"] == int(excluded)
    assert result["missing_atoms_outside_binding_site"] == int(not excluded)
    assert result["nearest_missing_atom_distance_to_box"] == pytest.approx(x - 1)


def test_missing_corner_atom_outside_spherical_cutoff_is_recorded():
    result = audit(
        [source_atom("CA", (0, 0, 0)), source_atom("CB", (7, 7, 0))],
        [receptor_atom((0, 0, 0))],
    )
    assert not result["defective"]
    assert result["nearest_outside_missing_atom_distance_to_box"] == pytest.approx(
        np.sqrt(72)
    )


def test_target_ligand_contamination_excludes_even_far_outside_the_box():
    result = audit(
        [
            source_atom("CA", (0, 0, 0)),
            source_atom(
                "C1", (100, 0, 0), chain="target", polymer=False, residue="LIG"
            ),
        ],
        [receptor_atom((0, 0, 0)), receptor_atom((100, 0, 0))],
    )
    assert result["defective"]
    assert result["target_ligand_atoms_present"] == 1
    assert result["binding_site_missing_atoms"] == 0


def test_one_missing_alternate_position_near_box_is_sufficient():
    result = audit(
        [
            source_atom("CA", (0, 0, 0)),
            source_atom("CB", (50, 0, 0), alt="A"),
            source_atom("CB", (2, 0, 0), alt="B"),
        ],
        [receptor_atom((0, 0, 0))],
    )
    assert result["defective"]
    assert result["binding_site_missing_atoms"] == 1
    assert result["nearest_missing_atom_distance_to_box"] == 1


@pytest.mark.parametrize("field", ["center_x", "size_y"])
def test_nonfinite_box_is_rejected_even_for_a_complete_receptor(field):
    with pytest.raises(ValueError, match="finite"):
        compare_source_atoms(
            [source_atom("CA", (0, 0, 0))],
            [receptor_atom((0, 0, 0))],
            target_ligand="target",
            drop_nonpolymer=False,
            docking_box={**BOX, field: np.nan},
        )
