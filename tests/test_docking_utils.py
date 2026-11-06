"""Focused tests for receptor preparation helpers."""

import sys
from pathlib import Path

import gemmi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docking"))

from cif_utils import sanitize_asym_id_columns_in_cif  # noqa: E402


def test_sanitize_asym_ids_keeps_complete_ids_distinct(tmp_path):
    source = tmp_path / "input.cif"
    output = tmp_path / "output.cif"
    source.write_text(
        """data_chains
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_asym_id
_atom_site.label_asym_id
ATOM 1 AA AA
ATOM 2 BA BA
ATOM 3 AA AA
"""
    )

    assert sanitize_asym_id_columns_in_cif(source, output)

    block = gemmi.cif.read_file(str(output)).sole_block()
    labels = list(block.find_loop("_atom_site.label_asym_id"))
    authors = list(block.find_loop("_atom_site.auth_asym_id"))
    assert labels == authors
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert all(len(chain_id) == 1 for chain_id in labels)
