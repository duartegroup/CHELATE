"""Test nested archive publication and preservation of scientific inputs."""

import pytest

from docking import artifacts


def test_missing_archive_parents_are_created(tmp_path):
    source = tmp_path / "source.pdbqt"
    source.write_bytes(b"original receptor")
    destination = tmp_path / "missing" / "archived_inputs" / "source.pdbqt"
    artifacts.copy_immutable(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b"original receptor"


def test_an_existing_different_archive_is_never_replaced(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "archive"
    source.write_bytes(b"new input")
    destination.write_bytes(b"old input")
    with pytest.raises(FileExistsError):
        artifacts.copy_immutable(source, destination)
    assert destination.read_bytes() == b"old input"


def test_identical_archive_can_be_reused(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "archive"
    source.write_bytes(b"same")
    destination.write_bytes(b"same")
    artifacts.copy_immutable(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b"same"


def test_a_changing_source_is_not_published(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "archive"
    source.write_bytes(b"first version")
    copy = artifacts.shutil.copy2

    def changed(src, dst):
        result = copy(src, dst)
        src.write_bytes(b"second version")
        return result

    monkeypatch.setattr(artifacts.shutil, "copy2", changed)
    with pytest.raises(ValueError, match="Source changed while copying"):
        artifacts.copy_immutable(source, destination)
    assert not destination.exists()


def test_preparation_provenance_can_live_away_from_cached_receptor(tmp_path):
    import hashlib
    import json

    from docking.artifacts import verified_preparation_record

    receptor = tmp_path / "cached" / "receptor.pdbqt"
    receptor.parent.mkdir()
    receptor.write_text("immutable receptor")
    digest = hashlib.sha256(receptor.read_bytes()).hexdigest()
    record = tmp_path / "preparation.json"
    record.write_text(json.dumps({"receptor_sha256": digest}))
    prepared = {
        "prepared_record": str(record),
        "receptor": str(receptor),
        "receptor_sha256": digest,
    }
    assert verified_preparation_record(prepared) == record
    receptor.write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        verified_preparation_record(prepared)


def test_archive_stays_unchanged_when_the_editable_source_changes(tmp_path):
    source, destination = tmp_path / "source.py", tmp_path / "bundle/source.py"
    source.write_text("original source")
    artifacts.copy_immutable(source, destination)
    source.write_text("later source edit")
    assert destination.read_text() == "original source"


def test_existing_source_hardlink_is_not_accepted_as_an_independent_archive(tmp_path):
    import os

    source, destination = tmp_path / "source.py", tmp_path / "snapshot.py"
    source.write_text("shared source")
    os.link(source, destination)
    with pytest.raises(ValueError, match="shares the editable source inode"):
        artifacts.copy_immutable(source, destination)
    assert source.read_text() == destination.read_text() == "shared source"
