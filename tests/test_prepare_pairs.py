"""Do not accept Meeko's successful exit when source protein atoms vanished."""

import subprocess
from pathlib import Path

import pytest

from docking import prepare_pairs, receptor_integrity


def atom(serial, name, x, *, chain="A", element="C"):
    return f"ATOM  {serial:5d} {name:>4s} ALA {chain}   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}"


@pytest.fixture
def original(tmp_path):
    p = tmp_path / "original.cif"
    p.write_text(
        "data_source\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n_atom_site.auth_asym_id\n"
        "_atom_site.label_seq_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM C CA ALA 1.A 1.A 1 1 0 0 0\n"
        "ATOM C CB ALA 1.A 1.A 1 1 1 0 0\n"
        "HETATM Mg MG MG 1.M 1.M . 2 2 0 0\n"
        "HETATM C C1 LIG 1.L 1.L . 3 5 0 0\n"
    )
    return p


def mock_meeko(monkeypatch, *, always_partial=False, contaminant=False):
    def run(command, **_kwargs):
        prefix = Path(command[command.index("-o") + 1])
        lines = [atom(1, "CA", 0)]
        if prefix.name != "direct" and not always_partial:
            lines.append(atom(2, "CB", 1))
        if contaminant:
            lines.append(atom(3, "C1", 5))
        prefix.with_suffix(".pdbqt").write_text(
            "\n".join(line[:66] + "     0.000 C" for line in lines) + "\n"
        )
        return subprocess.CompletedProcess(
            command, 0, "success despite omitted residue", ""
        )

    def normalize(_source, output):
        output.write_text(atom(1, "CA", 0) + "\n" + atom(2, "CB", 1) + "\n")

    monkeypatch.setattr(prepare_pairs.subprocess, "run", run)
    monkeypatch.setattr(prepare_pairs, "normalize_protein", normalize)
    monkeypatch.setattr(receptor_integrity, "pdbfixer_replacement_rules", dict)


def test_silent_source_loss_retries_and_metals_are_retained(
    original, tmp_path, monkeypatch
):
    mock_meeko(monkeypatch)
    old = tmp_path / "old_receptor.pdbqt"
    old.write_text("original receptor must remain archived and untouched")
    result = prepare_pairs.prepare_original_pair(
        original, "1.L", tmp_path / "work", python=Path("python"), meeko=Path("meeko")
    )
    assert result["policy"] == "pdbfixer"
    assert result["attempts"][0]["returncode"] == 0
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][0]["source_atom_audit"]["missing_protein_atoms"] == 1
    assert not result["source_atom_audit"]["defective"]
    assert result["source_atom_audit"]["source_atom_groups_retained"] == 3
    assert result["metal_retention"]["ions_added"][0]["charge"] == 2
    assert old.read_text() == "original receptor must remain archived and untouched"


@pytest.mark.parametrize("contaminant", [False, True])
def test_no_partial_or_contaminated_receptor_is_published(
    original, tmp_path, monkeypatch, contaminant
):
    mock_meeko(monkeypatch, always_partial=not contaminant, contaminant=contaminant)
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="No complete receptor"):
        prepare_pairs.prepare_original_pair(
            original, "1.L", work, python=Path("python"), meeko=Path("meeko")
        )
    assert not (work / "prepared.json").exists()


def test_prepared_record_cannot_hide_changed_inputs(original, tmp_path, monkeypatch):
    mock_meeko(monkeypatch)
    work = tmp_path / "work"
    prepare_pairs.prepare_original_pair(
        original, "1.L", work, python=Path("python"), meeko=Path("meeko")
    )
    original.write_text(original.read_text() + "# changed original\n")
    with pytest.raises(ValueError, match="Previously prepared input changed"):
        prepare_pairs.prepare_original_pair(
            original, "1.L", work, python=Path("python"), meeko=Path("meeko")
        )


def test_unsupported_ion_split_uses_column_names(tmp_path):
    source = tmp_path / "source.cif"
    source.write_text(
        "data_source\nloop_\n_atom_site.Cartn_z\n_atom_site.label_atom_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n_atom_site.type_symbol\n_atom_site.Cartn_y\n"
        "3 NA 7 1 NA B Na 2\n0 N 1 0 ALA A N 0\n"
    )
    protein, ions = tmp_path / "protein.cif", tmp_path / "ions.pdbqt"
    prepare_pairs.split_source_unsupported(source, protein, ions)
    line = ions.read_text().splitlines()[0]
    assert line.split()[-1] == "Na"
    assert line[21:22] == "B"
    assert [float(line[i : i + 8]) for i in [30, 38, 46]] == [1, 2, 3]
    assert (
        len(
            prepare_pairs.read_atom_site_document(str(protein))
            .sole_block()
            .find_values("_atom_site.type_symbol")
        )
        == 1
    )


def test_cached_preparation_is_checked_against_current_audit(
    original, tmp_path, monkeypatch
):
    mock_meeko(monkeypatch)
    work = tmp_path / "work"
    kwargs = {"python": Path("python"), "meeko": Path("meeko")}
    prepare_pairs.prepare_original_pair(original, "1.L", work, **kwargs)
    frozen = (work / "prepared.json").read_bytes()
    real_audit = prepare_pairs.audit_original_receptor

    def stricter_audit(*args, **kwargs):
        return {**real_audit(*args, **kwargs), "defective": True}

    monkeypatch.setattr(prepare_pairs, "audit_original_receptor", stricter_audit)
    with pytest.raises(ValueError, match="Cached receptor failed the current"):
        prepare_pairs.prepare_original_pair(original, "1.L", work, **kwargs)
    assert (work / "prepared.json").read_bytes() == frozen


def test_cached_preparation_cannot_be_used_for_another_ligand(
    original, tmp_path, monkeypatch
):
    mock_meeko(monkeypatch)
    work = tmp_path / "work"
    kwargs = {"python": Path("python"), "meeko": Path("meeko")}
    prepare_pairs.prepare_original_pair(original, "1.L", work, **kwargs)
    with pytest.raises(ValueError, match="different source or ligand"):
        prepare_pairs.prepare_original_pair(original, "1.M", work, **kwargs)


def test_free_residue_retry_still_requires_every_original_atom(
    original, tmp_path, monkeypatch
):
    mock_meeko(monkeypatch, always_partial=True)
    previous = prepare_pairs.subprocess.run
    observed = []

    def run(command, **kwargs):
        observed.append(command)
        prefix = Path(command[command.index("-o") + 1])
        if "--backend" not in command:
            return previous(command, **kwargs)
        prefix.with_suffix(".pdbqt").write_text(
            "\n".join(
                line[:66] + "     0.000 C"
                for line in [atom(1, "CA", 0), atom(2, "CB", 1)]
            )
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "complete free residue", "")

    def build(_python, output):
        output.write_text("{}")
        return {
            "extension": str(output),
            "extension_sha256": "synthetic-template-fixture",
        }

    monkeypatch.setattr(prepare_pairs.subprocess, "run", run)
    monkeypatch.setattr(prepare_pairs, "build_free_residue_overlay", build)
    result = prepare_pairs.prepare_original_pair(
        original, "1.L", tmp_path / "work", python=Path("python"), meeko=Path("meeko")
    )
    assert result["policy"] == "free_residue_template_geometry_retry"
    assert result["source_atom_audit"]["source_atom_groups_retained"] == 3
    assert not result["source_atom_audit"]["defective"]
    assert all(a["status"] == "failed" for a in result["attempts"][:-1])
    assert result["metal_retention"]["ions_added"][0]["element"] == "Mg"
    assert result["parent_pdb"].endswith("geometry_retry.pdb")
    assert observed[-1][observed[-1].index("--backend") + 2] == "--"
