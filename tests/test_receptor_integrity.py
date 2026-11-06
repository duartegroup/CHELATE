"""Tests for missing source atoms and incorrect ligand provenance."""

import pytest

from docking.receptor_integrity import (
    compare_source_atoms,
    read_receptor_atoms,
    read_source_atoms,
)


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


def compare(source, receptor, **kwargs):
    return compare_source_atoms(
        source,
        receptor,
        target_ligand=kwargs.pop("target_ligand", "target"),
        drop_nonpolymer=kwargs.pop("drop_nonpolymer", False),
        **kwargs,
    )


def test_missing_protein_atom_is_not_hidden_by_retained_metals():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom("CB", (1, 0, 0)),
        source_atom(
            "MG", (2, 0, 0), chain="metal", element="Mg", polymer=False, residue="MG"
        ),
    ]
    result = compare(source, [receptor_atom((0, 0, 0)), receptor_atom((2, 0, 0), "Mg")])
    assert result["defective"]
    assert result["missing_protein_atoms"] == 1
    assert result["missing_metal_atoms"] == 0
    assert result["missing_atoms"][0]["name"] == "CB"


def test_nonpolymer_removal_never_excuses_missing_source_metal():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom(
            "MG", (2, 0, 0), chain="metal", element="Mg", polymer=False, residue="MG"
        ),
    ]
    result = compare(source, [receptor_atom((0, 0, 0))], drop_nonpolymer=True)
    assert result["missing_metal_atoms"] == 1


def test_target_ligand_contamination_is_found_without_chain_labels():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom("C1", (2, 0, 0), chain="target", polymer=False, residue="LIG"),
    ]
    result = compare(source, [receptor_atom((0, 0, 0)), receptor_atom((2, 0, 0))])
    assert result["missing_protein_atoms"] == 0
    assert result["target_ligand_atoms_present"] == 1
    assert result["defective"]


def test_original_source_is_filtered_for_this_ligand_not_another_ligand():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom("C1", (2, 0, 0), chain="ligand_O", polymer=False, residue="LIG"),
        source_atom("C1", (4, 0, 0), chain="ligand_P", polymer=False, residue="LIG"),
    ]
    receptor = [receptor_atom((0, 0, 0)), receptor_atom((4, 0, 0))]
    assert not compare(source, receptor, target_ligand="ligand_O")["defective"]
    other = compare(source, receptor, target_ligand="ligand_P")
    assert other["defective"]
    assert other["target_ligand_atoms_present"] == 1


@pytest.mark.parametrize(("delta", "missing"), [(0.009, 0), (0.011, 1)])
def test_original_coordinate_tolerance_is_one_hundredth_angstrom(delta, missing):
    result = compare([source_atom("CA", (0, 0, 0))], [receptor_atom((delta, 0, 0))])
    assert result["missing_protein_atoms"] == missing


def test_caller_cannot_relax_the_coordinate_tolerance():
    with pytest.raises(ValueError, match=r"fixed at 0\.01"):
        compare([source_atom("CA", (0, 0, 0))], [receptor_atom((0, 0, 0))], tolerance=1)


def test_added_atoms_and_removed_hydrogens_waters_are_not_defects():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom("H", (1, 0, 0), element="H"),
        source_atom("O", (2, 0, 0), element="O", residue="HOH", polymer=False),
    ]
    result = compare(source, [receptor_atom((0, 0, 0)), receptor_atom((0, 1, 0))])
    assert not result["defective"]
    assert result["ignored_source_atoms"] == {"hydrogen": 1, "water": 1}


def test_fallback_removes_only_the_explicit_nonstandard_template_atoms():
    source = [
        source_atom("CA", (0, 0, 0), residue="SEP"),
        source_atom("P", (2, 0, 0), element="P", residue="SEP"),
    ]
    result = compare(source, [receptor_atom((0, 0, 0))], replacements={"SEP": {"CA"}})
    assert not result["defective"]
    assert result["ignored_source_atoms"]["replacement_atom"] == 1
    result = compare(
        source, [receptor_atom((2, 0, 0), "P")], replacements={"SEP": {"CA"}}
    )
    assert result["missing_protein_atoms"] == 1


def test_alternate_sites_require_one_retained_conformer():
    source = [
        source_atom("CA", (0, 0, 0), alt="A"),
        source_atom("CA", (1, 0, 0), alt="B"),
    ]
    result = compare(source, [receptor_atom((1, 0, 0))])
    assert not result["defective"]
    assert result["source_atom_groups_expected"] == 1


def test_duplicate_expected_atoms_cannot_share_one_receptor_atom():
    source = [source_atom("CA", (0, 0, 0)), source_atom("CA", (0, 0, 0), chain="other")]
    assert compare(source, [receptor_atom((0, 0, 0))])["missing_protein_atoms"] == 1


def test_autodock_acceptor_nitrogen_is_not_a_sodium_ion(tmp_path):
    path = tmp_path / "receptor.pdbqt"
    path.write_text(
        "ATOM      1   N1 ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 NA\n"
    )
    assert read_receptor_atoms(path)[0]["element"] == "N"
    source = [
        source_atom(
            "NA", (0, 0, 0), chain="metal", element="Na", residue="NA", polymer=False
        )
    ]
    assert compare(source, read_receptor_atoms(path))["missing_metal_atoms"] == 1


def test_source_reader_retains_complete_chain_ids_and_polymer_identity(tmp_path):
    path = tmp_path / "original.cif"
    path.write_text(
        "data_original\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\nATOM C CA ALA 1.A 1 0 0 0\nHETATM Mg MG MG 1.AA . 2 0 0\n"
    )
    atoms = read_source_atoms(path)
    assert [a["chain"] for a in atoms] == ["1.A", "1.AA"]
    assert [a["polymer"] for a in atoms] == [True, False]


@pytest.mark.parametrize("residue", ["ASP", "ALA", "DA", "U", "UNK", "N"])
def test_pdbfixer_keeps_standard_residues_even_when_marked_nonpolymer(residue):
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom(
            "C1", (2, 0, 0), chain="free_residue", residue=residue, polymer=False
        ),
    ]
    result = compare(source, [receptor_atom((0, 0, 0))], drop_nonpolymer=True)
    assert result["defective"]
    assert result["missing_other_atoms"] == 1
    assert result["ignored_source_atoms"].get("removed_nonpolymer", 0) == 0


def test_pdbfixer_still_removes_an_unrecognized_nonpolymer_component():
    source = [
        source_atom("CA", (0, 0, 0)),
        source_atom("C1", (2, 0, 0), chain="heterogen", residue="LIG", polymer=False),
    ]
    assert not compare(source, [receptor_atom((0, 0, 0))], drop_nonpolymer=True)[
        "defective"
    ]
