"""Check general preparation repair rules and transactional output handling."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docking"))

import repair_receptors as repair  # noqa: E402


def atom(serial, name, element, xyz, chain="A", residue=1):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )


def source_cif(path, *, original_oxt=False):
    text = (
        "data_source\nloop_\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.label_atom_id\n"
        "_atom_site.type_symbol\n_atom_site.label_comp_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "A A 1 CA C ALA 0 0 0\nA A 1 C C ALA 1.5 0 0\n"
        "A A 1 O O ALA 2.1 1 0\nB B 2 ZN Zn ZN 2 0 0\n"
    )
    if original_oxt:
        text += "A A 1 OXT O ALA 2.1 1.05 0\n"
    path.write_text(text)
    return path


def test_deuterium_normalization_keeps_coordinates_and_atom_identity(tmp_path):
    source = tmp_path / "input.pdb"
    line = atom(1, "D1", "D", (1.234, -2.345, 3.456))
    source.write_text(line + "\n")
    output = tmp_path / "output.pdb"
    changes = repair.repair_normalized_pdb(
        source, source_cif(tmp_path / "source.cif"), output
    )
    result = output.read_text().splitlines()[0]
    assert changes["deuteriums_normalized"] == 1
    assert result[:76] == line[:76]
    assert result[76:78].strip() == "H"


def test_geometry_retry_rebuilds_added_oxt_and_keeps_observed_atoms(tmp_path):
    observed = [
        atom(1, "CA", "C", (0, 0, 0)),
        atom(2, "C", "C", (1.5, 0, 0)),
        atom(3, "O", "O", (2.1, 1, 0)),
    ]
    source = tmp_path / "input.pdb"
    source.write_text(
        "\n".join([*observed, atom(4, "OXT", "O", (2.1, 1.05, 0))]) + "\n"
    )
    output = tmp_path / "output.pdb"
    changes = repair.repair_normalized_pdb(
        source, source_cif(tmp_path / "source.cif"), output, rebuild_geometry=True
    )
    lines = output.read_text().splitlines()
    assert lines[:3] == observed
    position = [float(lines[3][start : start + 8]) for start in (30, 38, 46)]
    assert position == [2.1, -1.0, 0.0]
    assert math.dist(position, [1.5, 0, 0]) == pytest.approx(
        math.dist([2.1, 1, 0], [1.5, 0, 0])
    )
    assert len(changes["added_terminal_oxygens_rebuilt"]) == 1


def test_geometry_retry_preserves_original_oxt_and_only_removes_hydrogens(tmp_path):
    heavy = [
        atom(1, "CA", "C", (0, 0, 0)),
        atom(2, "C", "C", (1.5, 0, 0)),
        atom(3, "O", "O", (2.1, 1, 0)),
        atom(4, "OXT", "O", (2.1, 1.05, 0)),
    ]
    source = tmp_path / "input.pdb"
    source.write_text(
        "\n".join(
            [*heavy, atom(5, "H1", "H", (0, 0, 1)), atom(6, "D1", "D", (0, 1, 0))]
        )
        + "\n"
    )
    output = tmp_path / "output.pdb"
    changes = repair.repair_normalized_pdb(
        source,
        source_cif(tmp_path / "source.cif", original_oxt=True),
        output,
        rebuild_geometry=True,
    )
    assert output.read_text().splitlines() == heavy
    assert changes["hydrogens_removed"] == 2
    assert changes["added_terminal_oxygens_rebuilt"] == []


def test_receptor_audit_detects_lost_metals_and_wrong_chains(tmp_path):
    protein = tmp_path / "protein.pdbqt"
    unsupported = tmp_path / "unsupported.pdbqt"
    merged = tmp_path / "merged.pdbqt"
    carbon = atom(1, "CA", "C", (0, 0, 0))[:66] + "     0.000 C"
    zinc = atom(2, "ZN", "Zn", (2, 0, 0), chain="B", residue=2)[:66] + "     2.000 Zn"
    protein.write_text(carbon + "\n")
    unsupported.write_text(zinc + "\n")
    merged.write_text(carbon + "\n" + zinc + "\n")
    cif = source_cif(tmp_path / "source.cif")
    assert (
        repair.audit_repaired_receptor(cif, protein, unsupported, merged)[
            "unsupported_atoms"
        ]
        == 1
    )
    merged.write_text(carbon + "\n")
    with pytest.raises(ValueError, match="lost separately prepared"):
        repair.audit_repaired_receptor(cif, protein, unsupported, merged)
    bad_zinc = zinc[:21] + "Z" + zinc[22:]
    unsupported.write_text(bad_zinc + "\n")
    merged.write_text(carbon + "\n" + bad_zinc + "\n")
    with pytest.raises(ValueError, match="chain IDs differ"):
        repair.audit_repaired_receptor(cif, protein, unsupported, merged)


def test_failed_retry_preserves_existing_receptor_files(tmp_path, monkeypatch):
    sid = "example"
    base = tmp_path / "train_systems" / sid
    base.mkdir(parents=True)
    cif = source_cif(base / "system_wo_unsupported.cif")
    original_pdb = atom(1, "CA", "C", (0, 0, 0)) + "\n"
    (base / "system_wo_unsupported_normalized.pdb").write_text(original_pdb)
    protein = base / "system_wo_unsupported_normalized.pdbqt"
    merged = base / "receptor_merged_1.L.pdbqt"
    protein.write_text("old failed protein\n")
    merged.write_text("old merged artifact\n")
    monkeypatch.setattr(
        repair.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "known preparation failure"
        ),
    )
    with pytest.raises(RuntimeError, match="No valid Meeko receptor"):
        repair.repair_system(
            "train",
            sid,
            {"1.L"},
            tmp_path,
            Path("python"),
            Path("meeko"),
            retry_normalized=True,
        )
    assert protein.read_text() == "old failed protein\n"
    assert merged.read_text() == "old merged artifact\n"
    assert cif.is_file()


def test_supported_source_metal_is_restored_with_template_charge(tmp_path):
    cif = source_cif(tmp_path / "source.cif")
    protein = tmp_path / "protein.pdbqt"
    protein.write_text(atom(1, "CA", "C", (0, 0, 0))[:66] + "     0.000 C\n")
    existing_unsupported = tmp_path / "existing_unsupported.pdbqt"
    existing_unsupported.write_text("")
    restored = tmp_path / "restored.pdbqt"
    result = repair.retain_source_metals(cif, existing_unsupported, protein, restored)
    assert len(result["ions_added"]) == 1
    assert result["ions_added"][0]["charge"] == 2
    assert repair.pdbqt_metal_sites(restored) == [("Zn", "B", 2.0, 0.0, 0.0)]
    merged = tmp_path / "merged.pdbqt"
    repair.merge_pdbqt(protein, restored, merged)
    assert (
        repair.audit_repaired_receptor(cif, protein, restored, merged)["merged_atoms"]
        == 2
    )
    # An ion already present in the prepared protein must not be added twice.
    second = tmp_path / "second.pdbqt"
    assert (
        repair.retain_source_metals(cif, existing_unsupported, merged, second)[
            "ions_added"
        ]
        == []
    )


def test_audit_detects_source_metal_loss_even_when_unsupported_file_is_empty(tmp_path):
    cif = source_cif(tmp_path / "source.cif")
    protein = tmp_path / "protein.pdbqt"
    protein.write_text(atom(1, "CA", "C", (0, 0, 0))[:66] + "     0.000 C\n")
    empty = tmp_path / "unsupported.pdbqt"
    empty.write_text("")
    with pytest.raises(ValueError, match="lost a metal ion from the sanitizer output"):
        repair.audit_repaired_receptor(cif, protein, empty, protein)


def test_residue_template_retry_moves_only_added_heavy_atoms(tmp_path):
    observed = [
        atom(1, "CA", "C", (0, 0, 0)),
        atom(2, "C", "C", (1.5, 0, 0)),
        atom(3, "O", "O", (2.1, 1, 0)),
    ]
    source = tmp_path / "input.pdb"
    source.write_text("\n".join([*observed, atom(4, "CB", "C", (1.5, 0.1, 0))]) + "\n")
    templates = tmp_path / "templates"
    templates.mkdir()
    # The template is translated relative to the observed anchors.
    (templates / "ALA.pdb").write_text(
        "\n".join(
            [
                atom(1, "CA", "C", (10, 0, 0)),
                atom(2, "C", "C", (11.5, 0, 0)),
                atom(3, "O", "O", (12.1, 1, 0)),
                atom(4, "CB", "C", (10, 0, 1.5)),
            ]
        )
        + "\n"
    )
    output = tmp_path / "output.pdb"
    changes = repair.repair_added_heavy_atoms(
        source, source_cif(tmp_path / "source.cif"), output, templates
    )
    lines = output.read_text().splitlines()
    assert lines[:3] == observed
    assert [float(lines[3][i : i + 8]) for i in (30, 38, 46)] == [0, 0, 1.5]
    assert [a["atom"] for a in changes["atoms_rebuilt"]] == ["CB"]


def test_residue_template_retry_keeps_original_atom_after_rename(tmp_path):
    # SE can be renamed to SD when PDBFixer replaces MSE with MET. Coordinates
    # identify it as observed, so it must not be repositioned as a missing atom.
    cif = source_cif(tmp_path / "source.cif")
    with cif.open("a") as handle:
        handle.write("A A 1 SE Se MSE 0 0 1.5\n")
    source = tmp_path / "input.pdb"
    lines = [
        atom(1, "CA", "C", (0, 0, 0)),
        atom(2, "C", "C", (1.5, 0, 0)),
        atom(3, "O", "O", (2.1, 1, 0)),
        atom(4, "SD", "S", (0, 0, 1.5)),
    ]
    source.write_text("\n".join(lines) + "\n")
    output = tmp_path / "output.pdb"
    changes = repair.repair_added_heavy_atoms(source, cif, output, tmp_path)
    assert output.read_text().splitlines() == lines
    assert changes["atoms_rebuilt"] == []


def test_direct_retry_uses_fixed_cif_and_does_not_require_pdbfixer(
    tmp_path, monkeypatch
):
    base = tmp_path / "val_systems" / "example"
    base.mkdir(parents=True)
    source_cif(base / "system_wo_unsupported.cif")
    source_cif(base / "system_sanitized.cif")
    (base / "system_unsupported.pdbqt").write_text("")
    commands = []

    def prepare(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1]).with_suffix(".pdbqt")
        output.write_text(atom(1, "CA", "C", (0, 0, 0))[:66] + "     0.000 C\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(repair.subprocess, "run", prepare)
    _, detail = repair.repair_system(
        "val",
        "example",
        {"1.L"},
        tmp_path,
        Path("python"),
        Path("meeko"),
        retry_direct=True,
        stage_only=True,
    )
    assert commands[0][2] == "-i"
    assert not (base / "system_wo_unsupported_normalized.pdb").exists()
    assert not (base / "receptor_merged_1.L.pdbqt").exists()
    assert any(
        destination.endswith("system_wo_unsupported.pdbqt")
        for _, destination in detail["publish_files"]
    )
    assert len(detail["metal_retention"]["ions_added"]) == 1


def test_pdb_conversion_preserves_distinct_metal_chains_and_bonds(tmp_path):
    ions = [
        atom(7, "FE", "Fe", (1, 2, 3), chain="U"),
        atom(8, "FE", "Fe", (4, 5, 6), chain="V"),
    ]
    source = tmp_path / "ions.pdbqt"
    source.write_text("\n".join(line[:66] + "     3.000 Fe" for line in ions) + "\n")
    converted = [ions[0], ions[1][:21] + "U" + ions[1][22:], "CONECT    7    8"]
    output = tmp_path / "ions.pdb"
    output.write_text("\n".join(converted) + "\n")
    assert repair.preserve_pdbqt_atom_identity(source, output) == 1
    restored = output.read_text().splitlines()
    assert [line[:30] for line in restored[:2]] == [line[:30] for line in ions]
    assert [line[30:] for line in restored[:2]] == [line[30:] for line in converted[:2]]
    assert restored[2] == converted[2]


@pytest.mark.parametrize("damage", ["coordinates", "missing_atom", "duplicate_serial"])
def test_pdb_conversion_rejects_ambiguous_atom_matches(tmp_path, damage):
    lines = [
        atom(7, "FE", "Fe", (1, 2, 3), chain="U"),
        atom(8, "FE", "Fe", (4, 5, 6), chain="V"),
    ]
    source = tmp_path / "ions.pdbqt"
    original = [line[:66] + "     3.000 Fe" for line in lines]
    if damage == "duplicate_serial":
        original[1] = original[1][:6] + original[0][6:11] + original[1][11:]
    source.write_text("\n".join(original) + "\n")
    if damage == "coordinates":
        lines[1] = lines[1][:30] + " 999.000" + lines[1][38:]
    elif damage == "missing_atom":
        lines.pop()
    output = tmp_path / "ions.pdb"
    text = "\n".join(lines) + "\n"
    output.write_text(text)
    with pytest.raises(ValueError):
        repair.preserve_pdbqt_atom_identity(source, output)
    assert output.read_text() == text


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_pdb_conversion_accepts_unchanged_signed_zero_coordinates(tmp_path, axis):
    xyz = [1.0, 2.0, 3.0]
    xyz[axis] = -0.0
    original = atom(7, "FE", "Fe", xyz, chain="V")
    xyz[axis] = 0.0
    converted = atom(7, "FE", "Fe", xyz, chain="U")
    source = tmp_path / "ions.pdbqt"
    source.write_text(original[:66] + "     3.000 Fe\n")
    output = tmp_path / "ions.pdb"
    output.write_text(converted + "\n")

    assert repair.preserve_pdbqt_atom_identity(source, output) == 1
    restored = output.read_text().splitlines()[0]
    assert restored[:30] == original[:30]
    assert restored[30:] == converted[30:]


def test_pdb_conversion_rejects_one_thousandth_angstrom_movement(tmp_path):
    original = atom(7, "FE", "Fe", (0.0, 2.0, 3.0))
    source = tmp_path / "ions.pdbqt"
    source.write_text(original[:66] + "     3.000 Fe\n")
    output = tmp_path / "ions.pdb"
    converted = atom(7, "FE", "Fe", (0.001, 2.0, 3.0)) + "\n"
    output.write_text(converted)

    with pytest.raises(ValueError, match="coordinates differ"):
        repair.preserve_pdbqt_atom_identity(source, output)
    assert output.read_text() == converted
