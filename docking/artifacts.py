"""Copy immutable scientific inputs without replacing an existing archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_immutable(source: Path, destination: Path) -> None:
    """Create missing archive parents and publish a verified copy atomically.

    The new archive has its own inode, so later source edits cannot change it.
    An identical independent archive can be reused. A different existing file is never
    replaced, including when another process creates it during the copy.
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _sha256(source)
    if destination.exists():
        if os.path.samefile(source, destination):
            raise ValueError("Archive shares the editable source inode")
        if _sha256(destination) != expected:
            raise FileExistsError(destination)
        return
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".archive_", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != expected or _sha256(source) != expected:
            raise ValueError(f"Source changed while copying: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256(destination) != expected:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def verified_preparation_record(prepared: dict) -> Path:
    """Use the declared provenance path even when a cached receptor lives elsewhere."""
    record = Path(prepared["prepared_record"])
    metadata = json.loads(record.read_text())
    receptor = Path(prepared["receptor"])
    expected = prepared["receptor_sha256"]
    if metadata["receptor_sha256"] != expected or _sha256(receptor) != expected:
        raise ValueError(
            f"Preparation provenance does not match the receptor: {record}"
        )
    return record
