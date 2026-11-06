"""Dative metal donors must not acquire extra valence during charge calculation."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from docking.prepare_ligand import corrected_metal_charges, has_uncapped_dative_donor


@pytest.mark.parametrize("smiles", ["[Fe+2]<-n1ccccc1", "[O-][Fe+2]<-n1ccccc1"])
def test_dative_donor_charge_copy_preserves_the_original_molecule(smiles):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(molecule)
    before = Chem.MolToMolBlock(molecule)
    assert has_uncapped_dative_donor(molecule)
    charges = corrected_metal_charges(molecule)
    assert len(charges) == molecule.GetNumAtoms()
    assert np.isfinite(charges).all()
    assert sum(charges) == pytest.approx(Chem.GetFormalCharge(molecule), abs=1e-8)
    assert Chem.MolToMolBlock(molecule) == before


def test_covalent_metal_bond_keeps_the_original_cap_charge_protocol():
    molecule = Chem.AddHs(Chem.MolFromSmiles("C[Fe]"))
    assert not has_uncapped_dative_donor(molecule)
    # Removing Fe leaves a methyl fragment; its H cap's charge returns to Fe.
    charges = corrected_metal_charges(molecule)
    assert np.isfinite(charges).all()
    assert sum(charges) == pytest.approx(0, abs=1e-8)
    assert charges[1] > 0


def test_configured_meeko_interpreter_is_preserved(tmp_path):
    from docking.prepare_ligand import preparation_command

    preparer = tmp_path / "mk_prepare_ligand.py"
    preparer.write_text("#!/bin/sh\n")
    preparer.chmod(0o755)
    interpreter = tmp_path / "python"
    interpreter.touch()
    assert preparation_command(preparer, "-i", "input.sdf") == [
        str(interpreter),
        "-m",
        "docking.prepare_ligand",
        "-i",
        "input.sdf",
    ]


def test_fallback_runs_only_after_a_dative_metal_valence_failure(monkeypatch):
    from types import SimpleNamespace

    from meeko.molsetup import RDKitMoleculeSetup

    from docking.prepare_ligand import metal_cap_fallback

    def original(setup, assign_charges, coords):
        if setup.fail:
            raise Chem.AtomValenceException("metal cap made donor overvalent")
        return "original preparation"

    monkeypatch.setattr(RDKitMoleculeSetup, "init_atom", original)
    molecule = Chem.AddHs(Chem.MolFromSmiles("[Fe+2]<-n1ccccc1"))
    added = []
    setup = SimpleNamespace(
        mol=molecule, atoms=[], fail=False, add_atom=lambda **kw: added.append(kw)
    )
    coordinates = np.zeros((molecule.GetNumAtoms(), 3))
    with metal_cap_fallback():
        assert (
            RDKitMoleculeSetup.init_atom(setup, True, coordinates)
            == "original preparation"
        )
        assert added == []
        setup.fail = True
        RDKitMoleculeSetup.init_atom(setup, True, coordinates)
        assert len(added) == molecule.GetNumAtoms()
        assert all(np.isfinite(record["charge"]) for record in added)
        assert sum(record["charge"] for record in added) == pytest.approx(2)
        with pytest.raises(Chem.AtomValenceException):
            RDKitMoleculeSetup.init_atom(setup, False, coordinates)
    assert RDKitMoleculeSetup.init_atom is original


def test_fallback_passes_newer_meeko_arguments_through(monkeypatch):
    from types import SimpleNamespace

    from meeko.molsetup import RDKitMoleculeSetup

    from docking.prepare_ligand import metal_cap_fallback

    # Meeko >= 0.7 calls init_atom(charge_model, read_charges_from_prop, coords,
    # template_key=..., template_charge=...).
    def original(setup, charge_model, read_charges_from_prop, coords, **kwargs):
        if setup.fail:
            raise Chem.AtomValenceException("metal cap made donor overvalent")
        return (charge_model, read_charges_from_prop, kwargs)

    monkeypatch.setattr(RDKitMoleculeSetup, "init_atom", original)
    molecule = Chem.AddHs(Chem.MolFromSmiles("[Fe+2]<-n1ccccc1"))
    setup = SimpleNamespace(mol=molecule, atoms=[], fail=False)
    with metal_cap_fallback():
        assert RDKitMoleculeSetup.init_atom(
            setup, "gasteiger", None, [], template_key=None, template_charge=None
        ) == ("gasteiger", None, {"template_key": None, "template_charge": None})
        setup.fail = True
        with pytest.raises(Chem.AtomValenceException):
            RDKitMoleculeSetup.init_atom(
                setup, "gasteiger", None, [], template_key=None, template_charge=None
            )
