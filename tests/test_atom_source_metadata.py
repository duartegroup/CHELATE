"""Keep malformed descriptive metadata from hiding valid source coordinates."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docking"))

from cif_utils import prepare_ligand_receptor_source, read_atom_site_document


def cif_text(atom_row):
    return (
        'data_original\n_chem_comp.name "broken descriptive metadata\n#\n'
        "loop_\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n"
        "_atom_site.label_comp_id\n_atom_site.Cartn_x\n" + atom_row + "\n#\n"
    )


def test_valid_atom_loop_survives_malformed_noncoordinate_metadata(tmp_path):
    source = tmp_path / "original.cif"
    source.write_text(cif_text("1.A 1.A ALA 0\n1.L 1.L LIG 1"))
    frozen = source.read_bytes()
    document = read_atom_site_document(str(source))
    assert list(document.sole_block().find_values("_atom_site.Cartn_x")) == ["0", "1"]
    paths = prepare_ligand_receptor_source(str(source), "1.L", str(tmp_path / "pair"))
    assert list(
        read_atom_site_document(paths["sanitized"])
        .sole_block()
        .find_values("_atom_site.Cartn_x")
    ) == ["0"]
    assert source.read_bytes() == frozen


def test_fallback_never_silently_drops_a_malformed_atom_record(tmp_path):
    source = tmp_path / "original.cif"
    source.write_text(cif_text('1.A 1.A ALA "unterminated'))
    with pytest.raises(ValueError):
        read_atom_site_document(str(source))
