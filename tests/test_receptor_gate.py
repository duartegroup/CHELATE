"""A cached receptor with its metal intact must still fail if protein atoms vanished."""

import pytest

from docking.receptor_gate import require_complete_receptor


def test_direct_docking_rejects_cached_truncation_with_metal_retained(tmp_path):
    source = tmp_path / "system.cif"
    source.write_text(
        "data_original\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM C CA ALA 1.A 1 0 0 0\nATOM C CB ALA 1.A 1 1 0 0\n"
        "HETATM Mg MG MG 1.M . 2 0 0\n"
    )
    receptor = tmp_path / "cached.pdbqt"
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n",
        "HETATM    3 MG    MG M   1       2.000   0.000   0.000  1.00  0.00     2.000 Mg\n",
    ]
    receptor.write_text("".join(lines))
    with pytest.raises(ValueError, match="protein=1, metal=0"):
        require_complete_receptor(source, receptor, "1.L")
    lines.append(
        "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00     0.000 C\n"
    )
    receptor.write_text("".join(lines))
    assert not require_complete_receptor(source, receptor, "1.L")["defective"]


def test_pocket_gate_allows_only_distant_source_loss(tmp_path):
    source = tmp_path / "original.cif"
    source.write_text(
        "data_source\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM C CA ALA 1.A 1 0 0 0\nATOM C CB ALA 1.A 1 50 0 0\n"
    )
    receptor = tmp_path / "prepared.pdbqt"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
    )
    box = {
        "center_x": 0,
        "center_y": 0,
        "center_z": 0,
        "size_x": 2,
        "size_y": 2,
        "size_z": 2,
    }
    result = require_complete_receptor(source, receptor, "target", docking_box=box)
    assert not result["defective"]
    assert result["strict_whole_protein_defective"]
    with pytest.raises(ValueError, match="protein=1"):
        require_complete_receptor(source, receptor, "target")
