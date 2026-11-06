"""Prepared PDB connectivity must not be extended with nearby nonbonded atoms."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from chelate.receptor import load_receptor_pdb


def crowded_ammonium():
    molecule = Chem.RWMol()
    atoms = [("N", 1), *[("C", 0)] * 4, ("C", 0), ("C", 0)]
    for index, (element, charge) in enumerate(atoms):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{element}{index:3d}")
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        molecule.AddAtom(atom)
    for index in range(1, 5):
        molecule.AddBond(0, index, Chem.BondType.SINGLE)
    molecule.AddBond(5, 6, Chem.BondType.SINGLE)
    conformer = Chem.Conformer(len(atoms))
    coordinates = [
        [0, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -0.5, 0.866],
        [0, -0.5, -0.866],
        [1.7, 0, 0],
        [3.2, 0, 0],
    ]
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, xyz)
    molecule.AddConformer(conformer)
    return molecule.GetMol()


def test_complete_conect_prevents_false_overvalent_nitrogen(tmp_path):
    path = tmp_path / "prepared.pdb"
    Chem.MolToPDBFile(crowded_ammonium(), str(path))
    original = path.read_bytes()
    assert Chem.MolFromPDBFile(str(path)) is None
    parsed = load_receptor_pdb(path)
    assert parsed is not None
    assert parsed.GetNumAtoms() == 7
    assert parsed.GetNumBonds() == 5
    assert path.read_bytes() == original
    expected = Chem.MolFromPDBFile(str(path), proximityBonding=False)
    assert np.array_equal(
        parsed.GetConformer().GetPositions(), expected.GetConformer().GetPositions()
    )


def test_pdb_without_connectivity_uses_proximity_bonds(tmp_path):
    path = tmp_path / "no_conect.pdb"
    block = Chem.MolToPDBBlock(crowded_ammonium())
    # A simple separated C-C fragment has no ambiguous geometry.
    records = [
        line
        for line in block.splitlines()
        if line.startswith("HETATM") and int(line[6:11]) >= 6
    ]
    path.write_text("\n".join(records) + "\nEND\n")
    molecule = load_receptor_pdb(path)
    assert molecule is not None
    assert molecule.GetNumBonds() == 1


def test_partial_connectivity_does_not_disable_bond_perception(tmp_path):
    path = tmp_path / "partial.pdb"
    block = Chem.MolToPDBBlock(crowded_ammonium())
    atoms = [
        line
        for line in block.splitlines()
        if line.startswith("HETATM") and int(line[6:11]) >= 6
    ]
    # A CONECT record concerning another residue does not describe this fragment.
    path.write_text("\n".join(atoms) + "\nCONECT 9998 9999\nEND\n")
    molecule = load_receptor_pdb(path)
    assert molecule is not None
    assert molecule.GetNumBonds() == 1


def test_standard_glutamate_conversion_bond_order_is_repaired(tmp_path):
    molecule = Chem.MolFromFASTA("E")
    names = {
        atom.GetPDBResidueInfo().GetName().strip(): atom.GetIdx()
        for atom in molecule.GetAtoms()
    }
    molecule.GetBondBetweenAtoms(names["CG"], names["CD"]).SetBondType(
        Chem.BondType.DOUBLE
    )
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, [index * 2.0, 0, 0])
    molecule.AddConformer(conformer)
    path = tmp_path / "wrong_glutamate.pdb"
    Chem.MolToPDBFile(molecule, str(path))
    original = path.read_bytes()
    assert Chem.MolFromPDBFile(str(path), proximityBonding=False) is None
    with pytest.warns(UserWarning, match="standard amino-acid"):
        parsed = load_receptor_pdb(path)
    assert parsed is not None
    assert (
        parsed.GetBondBetweenAtoms(names["CG"], names["CD"]).GetBondType()
        == Chem.BondType.SINGLE
    )
    assert np.array_equal(
        parsed.GetConformer().GetPositions(), molecule.GetConformer().GetPositions()
    )
    assert path.read_bytes() == original


def test_validated_conversion_does_not_add_bonds_to_isolated_atoms(tmp_path):
    molecule = Chem.RWMol(crowded_ammonium())
    chloride = Chem.Atom("Cl")
    chloride.SetFormalCharge(-1)
    info = Chem.AtomPDBResidueInfo()
    info.SetName(" CL ")
    info.SetResidueName("LIG")
    info.SetResidueNumber(1)
    info.SetChainId("A")
    info.SetIsHeteroAtom(True)
    chloride.SetMonomerInfo(info)
    molecule.AddAtom(chloride)
    molecule.GetConformer().SetAtomPosition(7, [30, 0, 0])
    path = tmp_path / "converted.pdb"
    Chem.MolToPDBFile(molecule, str(path))
    path.with_suffix(".pdb.provenance.json").write_text(
        json.dumps(
            {
                "schema": "chelate-receptor-pdb-v1",
                "pdb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )
    assert Chem.MolFromPDBFile(str(path)) is None
    parsed = load_receptor_pdb(path)
    assert parsed is not None
    assert parsed.GetNumAtoms() == 8
    assert parsed.GetNumBonds() == 5


def test_openbabel_conversion_without_sidecar_matches_validated_conversion(tmp_path):
    # A user's own OpenBabel conversion has no provenance sidecar but must be
    # bonded exactly like the pipeline's validated conversion of the same file.
    example = Path(__file__).resolve().parents[1] / "model" / "example"
    source = example / "4hev__1__1.A__1.C_1.D" / "Apo_receptor_prepared.pdb"
    lines = source.read_text().splitlines()
    # Leave Ser 11 OG (serial 6) out of the CONECT records: partial connectivity.
    stripped = []
    for line in lines:
        if line.startswith("CONECT"):
            serials = [line[k : k + 5] for k in range(6, len(line), 5)]
            serials = [x for x in serials if x.strip() and int(x) != 6]
            if len(serials) < 2:
                continue
            line = "CONECT" + "".join(serials)
        stripped.append(line)
    text = "\n".join(stripped) + "\n"
    validated = tmp_path / "validated" / "receptor.pdb"
    user = tmp_path / "user" / "receptor.pdb"
    for path in (validated, user):
        path.parent.mkdir()
        path.write_text(text)
    validated.with_suffix(".pdb.provenance.json").write_text(
        json.dumps(
            {
                "schema": "chelate-receptor-pdb-v1",
                "pdb_sha256": hashlib.sha256(validated.read_bytes()).hexdigest(),
            }
        )
    )
    expected = load_receptor_pdb(validated)
    parsed = load_receptor_pdb(user)
    assert expected is not None and parsed is not None
    assert parsed.GetNumAtoms() == expected.GetNumAtoms()
    assert sorted(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
        for b in parsed.GetBonds()
    ) == sorted(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
        for b in expected.GetBonds()
    )
