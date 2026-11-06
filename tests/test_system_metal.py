"""Cross-docked metal lookup must not silently drop the metal features."""

import pytest

from model import model_utils


def test_missing_crossdocked_receptor_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Cross-docked receptor not found"):
        model_utils.get_system_metal(
            "missing__1__1.A__1.B",
            ligand_id="1.B",
            target_system="xxxx_A",
            split="test",
            docking_type="apo_docked_FH",
        )


def test_crossdocked_metals_come_from_the_receptor(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMS_ROOT", str(tmp_path))
    system = "present__1__1.A__1.B"
    folder = tmp_path / "test_systems" / system
    folder.mkdir(parents=True)
    (folder / f"{system}__yyyy_A_merged_1.B.pdb").write_text(
        "HETATM    1 ZN    ZN A 401       0.000   0.000   0.000  1.00  0.00          ZN\n"
        "END\n"
    )
    assert model_utils.get_system_metal(
        system,
        ligand_id="1.B",
        target_system="yyyy_A",
        split="test",
        docking_type="apo_docked_FH",
    ) == ["ZN"]
