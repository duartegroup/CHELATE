"""Prevent preparing one ligand from replacing another ligand's source."""

import sys
from pathlib import Path

import gemmi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docking"))

from cif_utils import prepare_ligand_receptor_source


def original(path):
    path.write_text(
        "data_original\nloop_\n_atom_site.label_asym_id\n"
        "_atom_site.auth_asym_id\n_atom_site.label_comp_id\n"
        "_atom_site.Cartn_x\n"
        "1.A 1.A ALA 0\n1.O 1.O LIG 1\n1.P 1.P LIG 2\n"
        "1.AA 1.AA MG 3\n1.W 1.W HOH 4\n"
    )
    return path


def x_coordinates(path):
    return list(
        gemmi.cif.read_file(str(path)).sole_block().find_values("_atom_site.Cartn_x")
    )


def test_ligand_sources_are_independent_and_keep_the_other_ligand(tmp_path):
    source = original(tmp_path / "original.cif")
    original_bytes = source.read_bytes()
    first = prepare_ligand_receptor_source(str(source), "1.O", str(tmp_path / "1.O"))
    frozen = {name: Path(path).read_bytes() for name, path in first.items()}
    second = prepare_ligand_receptor_source(str(source), "1.P", str(tmp_path / "1.P"))
    assert source.read_bytes() == original_bytes
    assert {name: Path(path).read_bytes() for name, path in first.items()} == frozen
    assert x_coordinates(first["without_ligand"]) == ["0", "2", "3"]
    assert x_coordinates(second["without_ligand"]) == ["0", "1", "3"]
    assert x_coordinates(first["sanitized"]) == ["0", "2", "3"]
    block = gemmi.cif.read_file(first["sanitized"]).sole_block()
    assert len(set(block.find_values("_atom_site.label_asym_id"))) == 3


def test_absent_ligand_cannot_silently_prepare_the_wrong_receptor(tmp_path):
    source = original(tmp_path / "original.cif")
    destination = tmp_path / "wrong"
    with pytest.raises(ValueError, match="Target ligand chain is absent"):
        prepare_ligand_receptor_source(str(source), "1.X", str(destination))
    assert not destination.exists()


def test_metadata_lookup_preserves_literal_na_ligand_chain(tmp_path, monkeypatch):
    from model import model_utils

    dataset = tmp_path / "curated.csv"
    dataset.write_text(
        "system_id,ligand_instance,ligand_asym_id,target_system\nsystem,1,NA,apo_A\n"
    )
    monkeypatch.setattr(model_utils, "_DATASET_CSV", str(dataset))
    monkeypatch.setattr(model_utils, "_plinder_df", None)
    metadata = model_utils._get_plinder_df()
    assert metadata.loc[0, "ligand_id"] == "1.NA"
    assert metadata.loc[0, "system_id__ligand_id"] == "system__1.NA"
