"""Reject stale, incomplete and wrong-ligand receptor PDB caches."""

import subprocess
from pathlib import Path

import pytest

from docking import receptor_pdb_cache as cache


def atom(serial, name, element, x, chain="A"):
    return (
        f"ATOM  {serial:5d} {name:>4s} ALA {chain}{1:4d}    "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )


def source(path, x=1.0):
    lines = [atom(1, "CA", "C", x), atom(2, "FE", "Fe", 5.0, chain="V")]
    path.write_text(
        "\n".join(
            line[:66] + "     0.000 " + element
            for line, element in zip(lines, ["C", "Fe"], strict=True)
        )
        + "\n"
    )


@pytest.fixture
def converter(monkeypatch):
    calls = []

    def convert(command, **kwargs):
        calls.append(command)
        lines = []
        for line in Path(command[1]).read_text().splitlines():
            kind = line.split()[-1]
            element = cache.AUTODOCK_ELEMENTS.get(kind, kind.capitalize())
            lines.append(line[:66].ljust(76) + f"{element:>2s}")
        Path(command[3]).write_text("\n".join(lines) + "\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cache.subprocess, "run", convert)
    return calls


def test_cache_without_provenance_missing_a_metal_is_rebuilt_and_archived(
    tmp_path, converter
):
    qt, pdb = tmp_path / "receptor.pdbqt", tmp_path / "receptor.pdb"
    source(qt)
    existing = atom(1, "CA", "C", 1.0) + "\n"
    pdb.write_text(existing)
    assert cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive") == pdb
    cache.validate_converted_receptor(qt, pdb)
    assert len(converter) == 1
    archived = list((tmp_path / "archive").rglob("receptor.pdb"))
    assert len(archived) == 1 and archived[0].read_text() == existing


def test_matching_source_and_output_cache_does_not_reconvert(tmp_path, converter):
    qt, pdb = tmp_path / "receptor.pdbqt", tmp_path / "receptor.pdb"
    source(qt)
    cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive")
    cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive")
    assert len(converter) == 1


def test_changed_source_invalidates_even_an_existing_cache(tmp_path, converter):
    qt, pdb = tmp_path / "receptor.pdbqt", tmp_path / "receptor.pdb"
    source(qt)
    cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive")
    source(qt, x=9.0)
    cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive")
    assert len(converter) == 2
    assert float(pdb.read_text().splitlines()[0][30:38]) == 9.0


def test_failed_conversion_never_uses_the_existing_cache(tmp_path, monkeypatch):
    qt, pdb = tmp_path / "receptor.pdbqt", tmp_path / "receptor.pdb"
    source(qt)
    existing = atom(1, "CA", "C", 1.0) + "\n"
    pdb.write_text(existing)
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "failed"),
    )
    with pytest.raises(RuntimeError, match="conversion failed"):
        cache.ensure_receptor_pdb(qt, pdb, archive_dir=tmp_path / "archive")
    assert pdb.read_text() == existing


def test_probe_uses_the_receptor_for_the_exact_ligand(tmp_path, converter, monkeypatch):
    from model import predict_metal_identity as probe

    qt = tmp_path / "receptor_merged_1.L.pdbqt"
    source(qt)
    generic = tmp_path / "receptor_merged.pdb"
    generic.write_text(atom(1, "CA", "C", 99.0) + "\n")
    original = cache.ensure_receptor_pdb

    def local_cache(source_path, output_path):
        return original(source_path, output_path, archive_dir=tmp_path / "archive")

    monkeypatch.setattr(probe, "ensure_receptor_pdb", local_cache)
    result = probe._ensure_receptor_pdb(generic, qt)
    assert result == qt.with_suffix(".pdb")
    cache.validate_converted_receptor(qt, result)
    assert generic.read_text() == atom(1, "CA", "C", 99.0) + "\n"


def test_identical_pair_inputs_share_a_verified_conversion(tmp_path, converter):
    first = tmp_path / "receptor_merged_1.F.pdbqt"
    other = tmp_path / "receptor_merged_1.G.pdbqt"
    source(first)
    other.write_bytes(first.read_bytes())
    pdb = cache.ensure_receptor_pdb(
        first, first.with_suffix(".pdb"), archive_dir=tmp_path / "archive"
    )
    assert not other.with_suffix(".pdb").exists()
    assert (
        cache.validated_receptor_pdb(other, cache_by_sha256={cache.digest(first): pdb})
        == pdb
    )
    assert len(converter) == 1


def test_cached_alias_rejects_an_unrelated_conversion(tmp_path, converter):
    first, other = tmp_path / "first.pdbqt", tmp_path / "other.pdbqt"
    source(first)
    source(other, x=9.0)
    pdb = cache.ensure_receptor_pdb(
        first, first.with_suffix(".pdb"), archive_dir=tmp_path / "archive"
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        cache.validated_receptor_pdb(other, cache_by_sha256={cache.digest(other): pdb})


def test_cached_alias_checks_atom_identity_even_when_output_hash_matches(
    tmp_path, converter
):
    import json

    qt = tmp_path / "source.pdbqt"
    source(qt)
    pdb = cache.ensure_receptor_pdb(
        qt, qt.with_suffix(".pdb"), archive_dir=tmp_path / "archive"
    )
    pdb.write_text(pdb.read_text().replace("   1.000", "   9.000", 1))
    proof = pdb.with_suffix(".pdb.provenance.json")
    metadata = json.loads(proof.read_text())
    metadata["pdb_sha256"] = cache.digest(pdb)
    proof.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="coordinates differ"):
        cache.validated_receptor_pdb(qt)
