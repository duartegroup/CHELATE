import copy

import pytest

from docking import free_residue_templates as module


@pytest.fixture
def library():
    return {
        "residue_templates": {
            "ALA": {
                "smiles": "C(=O)C([H])(N[H])C([H])([H])[H]",
                "atom_name": [
                    "C",
                    "O",
                    "CA",
                    "HA",
                    "N",
                    "H",
                    "CB",
                    "HB1",
                    "HB2",
                    "HB3",
                ],
                "link_labels": {"0": "C-term", "4": "N-term"},
            },
            "NALA": {
                "smiles": "C(=O)C([H])([N+]([H])([H])[H])C([H])([H])[H]",
                "atom_name": [
                    "C",
                    "O",
                    "CA",
                    "HA",
                    "N",
                    "H1",
                    "H2",
                    "H3",
                    "CB",
                    "HB1",
                    "HB2",
                    "HB3",
                ],
                "link_labels": {"0": "C-term"},
            },
            "CALA": {
                "smiles": "[O-]C(=O)C([H])(N[H])C([H])([H])[H]",
                "atom_name": [
                    "OXT",
                    "C",
                    "O",
                    "CA",
                    "HA",
                    "N",
                    "H",
                    "CB",
                    "HB1",
                    "HB2",
                    "HB3",
                ],
                "link_labels": {"5": "N-term"},
            },
        },
        "ambiguous": {"ALA": ["ALA", "NALA", "CALA"]},
    }


def test_combined_template_has_both_charged_terminal_states(library):
    result = module.build_extension(library)["residue_templates"]["FREE_ALA"]
    mol = module.parse_template(result)
    names = result["atom_name"]
    nitrogen = mol.GetAtomWithIdx(names.index("N"))
    oxygen = mol.GetAtomWithIdx(names.index("OXT"))
    assert nitrogen.GetFormalCharge() == 1
    assert sum(a.GetAtomicNum() == 1 for a in nitrogen.GetNeighbors()) == 3
    assert oxygen.GetFormalCharge() == -1
    assert result["link_labels"] == {}
    assert {
        name
        for atom, name in zip(mol.GetAtoms(), names, strict=True)
        if atom.GetAtomicNum() != 1
    } == {"N", "CA", "C", "O", "OXT", "CB"}


def test_existing_templates_and_default_order_remain_unchanged(library):
    frozen = copy.deepcopy(library)
    result = module.build_extension(library)
    assert library == frozen
    assert result["ambiguous"]["ALA"] == ["ALA", "NALA", "CALA", "FREE_ALA"]


def test_additional_covalent_links_are_not_removed(library):
    library["residue_templates"]["CALA"]["link_labels"]["7"] = "side-chain-link"
    result = module.build_extension(library)
    assert result["residue_templates"] == {}


def test_atom_order_supports_pinned_and_current_rdkit_formats():
    assert module.smiles_atom_order("[0,2,1,]", 3) == [0, 2, 1]
    assert module.smiles_atom_order("[0,2,1]", 3) == [0, 2, 1]


def test_incomplete_or_duplicate_atom_mapping_is_rejected():
    with pytest.raises(ValueError, match="complete atom permutation"):
        module.smiles_atom_order("[0,0,1]", 3)
