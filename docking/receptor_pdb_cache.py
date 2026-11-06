"""Keep converted receptor PDB caches tied to their exact PDBQT source."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from docking.artifacts import copy_immutable
from docking.receptor_integrity import AUTODOCK_ELEMENTS
from docking.repair_receptors import preserve_pdbqt_atom_identity, validate_pdbqt

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / ".runtime/receptor_pdb_cache"
SCHEMA = "chelate-receptor-pdb-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_converted_receptor(pdbqt: Path, pdb: Path) -> None:
    """Require identical atom serials, finite coordinates, identities and elements."""
    validate_pdbqt(pdbqt)
    source = {}
    for line in pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial = int(line[6:11])
            if serial in source:
                raise ValueError("Duplicate source receptor atom serial")
            source[serial] = line
    seen = set()
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        serial = int(line[6:11])
        if serial not in source or serial in seen:
            raise ValueError("Converted receptor atom serials differ from source")
        original = source[serial]
        if original[:30] != line[:30]:
            raise ValueError("Converted receptor atom identity differs from source")
        for start in (30, 38, 46):
            if float(original[start : start + 8]) != float(line[start : start + 8]):
                raise ValueError("Converted receptor coordinates differ from source")
        kind = original.split()[-1]
        element = AUTODOCK_ELEMENTS.get(kind, kind.capitalize())
        if element != line[76:78].strip().capitalize():
            raise ValueError("Converted receptor element differs from source")
        seen.add(serial)
    if seen != set(source):
        raise ValueError("Receptor atoms lost during PDB conversion")


def ensure_receptor_pdb(
    pdbqt: Path, pdb: Path, *, archive_dir: Path = DEFAULT_ARCHIVE
) -> Path:
    """Reuse only a validated cache with matching source and output hashes.

    A PDB without a matching source fingerprint is regenerated. An existing PDB
    and its provenance are archived before replacement. Failed conversions leave
    the existing cache untouched and raise rather than using it as a fallback.
    """
    pdbqt, pdb, archive_dir = Path(pdbqt), Path(pdb), Path(archive_dir)
    source_hash = digest(pdbqt)
    identity = hashlib.sha256(str(pdb.absolute()).encode()).hexdigest()
    directory = archive_dir / identity
    directory.mkdir(parents=True, exist_ok=True)
    provenance = pdb.with_suffix(pdb.suffix + ".provenance.json")
    with (directory / "conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pdb.is_file() and provenance.is_file():
            try:
                previous = json.loads(provenance.read_text())
                if (
                    previous.get("schema") == SCHEMA
                    and previous.get("source_pdbqt_sha256") == source_hash
                    and previous.get("pdb_sha256") == digest(pdb)
                ):
                    validate_converted_receptor(pdbqt, pdb)
                    if digest(pdbqt) != source_hash:
                        raise ValueError("Receptor source changed while reading cache")
                    return pdb
            except (OSError, ValueError, TypeError):
                pass
        pdb.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".receptor_pdb_", dir=pdb.parent
        ) as work:
            converted = Path(work) / pdb.name
            command = ["obabel", str(pdbqt), "-O", str(converted)]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode or not converted.is_file():
                raise RuntimeError(
                    "Receptor conversion failed: " + result.stderr.strip()
                )
            preserve_pdbqt_atom_identity(pdbqt, converted)
            validate_converted_receptor(pdbqt, converted)
            if digest(pdbqt) != source_hash:
                raise ValueError("Receptor source changed during conversion")
            metadata = {
                "schema": SCHEMA,
                "source_pdbqt": str(pdbqt.absolute()),
                "source_pdbqt_sha256": source_hash,
                "pdb_sha256": digest(converted),
                "command": command,
            }
            if pdb.exists():
                copy_immutable(pdb, directory / digest(pdb) / pdb.name)
            if provenance.exists():
                copy_immutable(
                    provenance, directory / digest(provenance) / provenance.name
                )
            os.replace(converted, pdb)
            temporary = Path(work) / provenance.name
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            os.replace(temporary, provenance)
    return pdb


def validated_receptor_pdb(
    pdbqt: Path, *, cache_by_sha256: Mapping[str, Path] | None = None
) -> Path:
    """Resolve a conversion by exact source content, independent of ligand filename.

    Several pair-specific PDBQT paths can have identical bytes. A conversion made
    once for that content can be shared, while its original provenance is retained.
    Verify both fingerprints and atom identity before returning any cached path.
    """
    pdbqt = Path(pdbqt)
    source_hash = digest(pdbqt)
    pdb = Path((cache_by_sha256 or {}).get(source_hash, pdbqt.with_suffix(".pdb")))
    provenance = pdb.with_suffix(pdb.suffix + ".provenance.json")
    metadata = json.loads(provenance.read_text())
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("source_pdbqt_sha256") != source_hash
        or metadata.get("pdb_sha256") != digest(pdb)
    ):
        raise ValueError("Cached receptor conversion provenance does not match")
    validate_converted_receptor(pdbqt, pdb)
    if digest(pdbqt) != source_hash:
        raise ValueError("Receptor source changed while resolving its conversion")
    return pdb
