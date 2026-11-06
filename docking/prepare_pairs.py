"""Prepare isolated, original-source receptors and reject silent atom loss."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from chelate.constants import METAL_ELEMENT_SYMBOLS
from docking.cif_utils import (
    prepare_ligand_receptor_source,
    read_atom_site_document,
    sanitize_asym_id_columns_in_cif,
)
from docking.docking_params import CHARGE_DICT
from docking.free_residue_templates import build_free_residue_overlay
from docking.receptor_integrity import audit_original_receptor
from docking.repair_receptors import (
    audit_repaired_receptor,
    merge_pdbqt,
    normalize_protein,
    repair_added_heavy_atoms,
    repair_normalized_pdb,
    retain_source_metals,
)


def split_source_unsupported(source: Path, protein: Path, unsupported: Path) -> None:
    """Apply the ``CHARGE_DICT`` unsupported-ion charge rules using CIF column names."""
    document = read_atom_site_document(str(source))
    table = document.sole_block().find_mmcif_category("_atom_site.")
    indices = {tag.split(".", 1)[1]: i for i, tag in enumerate(table.tags)}
    records = []
    removed = []
    for i, row in enumerate(table):

        def value(name, default="", atom_row=row):
            return (
                str(atom_row[indices[name]]).strip("\"'")
                if name in indices
                else default
            )

        symbol = value("type_symbol").upper()
        residue = value("label_comp_id")
        # The same charge rules apply to source elements and named ions (FE2).
        if symbol not in CHARGE_DICT and not (
            symbol.capitalize() in METAL_ELEMENT_SYMBOLS and residue in CHARGE_DICT
        ):
            continue
        charge = CHARGE_DICT[residue if residue in CHARGE_DICT else symbol]
        chain = value("auth_asym_id", value("label_asym_id"))
        if len(chain) != 1:
            raise ValueError(
                "Unsupported ions must use the sanitized single-character chain map"
            )
        number = value("auth_seq_id", value("label_seq_id", "0"))
        number = int(number) if number not in {".", "?", ""} else 0
        x, y, z = [float(value("Cartn_" + axis)) for axis in "xyz"]
        atom = value("label_atom_id")
        records.append(
            f"HETATM{len(records) + 1:5d} {atom:>4s} {residue:>3s} {chain}{number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"    {charge:6.3f} {symbol.capitalize():<2s}"
        )
        removed.append(i)
    for i in reversed(removed):
        table.remove_row(i)
    document.write_file(str(protein))
    unsupported.write_text("\n".join(records) + ("\n" if records else ""))


def prepare_pair_source(original: Path, ligand: str | None, work: Path) -> Path:
    """Filter the exact target chain for redocking, or water only for an apo CIF."""
    work.mkdir(parents=True, exist_ok=True)
    if ligand is not None:
        paths = prepare_ligand_receptor_source(str(original), ligand, str(work))
        return Path(paths["sanitized"])
    document = read_atom_site_document(str(original))
    table = document.sole_block().find_mmcif_category("_atom_site.")
    indices = {tag.split(".", 1)[1]: i for i, tag in enumerate(table.tags)}
    for i in reversed(range(len(table))):
        if table[i][indices["label_comp_id"]] in {"HOH", "WAT", "H2O", "TIP", "SOL"}:
            table.remove_row(i)
    cleaned = work / "apo_wo_solvent.cif"
    document.write_file(str(cleaned))
    fixed = work / "system_sanitized.cif"
    sanitize_asym_id_columns_in_cif(str(cleaned), str(fixed))
    return fixed


def prepare_original_pair(
    original: Path,
    ligand: str | None,
    work: Path,
    *,
    python: Path,
    meeko: Path,
    residue_templates_dir: Path | None = None,
) -> dict:
    """Retry within the pipeline, accepting only a complete original-source audit.

    Meeko's successful exit with --allow_bad_res is not sufficient: that option
    can omit source protein residues. All candidates must pass both the chain/
    metal audit and the original heavy-atom/target-ligand audit before use.
    Existing receptors and poses are never written by this staging function.
    """
    work.mkdir(parents=True, exist_ok=True)
    record = work / "prepared.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if (
            Path(previous["original_source"]).resolve() != original.resolve()
            or previous.get("target_ligand") != ligand
        ):
            raise ValueError(
                "Cached preparation belongs to a different source or ligand"
            )
        for path, expected in [
            (original, previous["original_source_sha256"]),
            (Path(previous["receptor"]), previous["receptor_sha256"]),
        ]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Previously prepared input changed: {path}")
        current = audit_original_receptor(
            original,
            Path(previous["receptor"]),
            target_ligand=ligand,
            pdbfixer_used=previous["policy"] != "direct",
        )
        if current["defective"]:
            raise ValueError(
                "Cached receptor failed the current original-source audit; "
                f"retry in a new isolated directory: {record}"
            )
        audit_repaired_receptor(
            Path(previous["fixed_source"]),
            Path(previous["protein"]),
            Path(previous["unsupported"]),
            Path(previous["receptor"]),
        )
        return {
            **previous,
            "source_atom_audit": current,
            "prepared_record": str(record),
        }
    fixed = prepare_pair_source(original, ligand, work)
    supported = work / "system_wo_unsupported.cif"
    unsupported = work / "system_unsupported.pdbqt"
    split_source_unsupported(fixed, supported, unsupported)
    attempts = []
    normalized = work / "pdbfixer_original.pdb"
    selected = None
    policies = ["direct", "pdbfixer", "geometry_retry"]
    if residue_templates_dir is not None:
        policies.append("residue_template_retry")
    # Append combined-terminal templates only after all other preparation policies.
    # Each retry reuses an existing PDB; source heavy-atom coordinates stay fixed.
    free_parents = [name for name in reversed(policies) if name != "direct"]
    policies.extend("free_residue_template_" + name for name in free_parents)
    overlay_provenance = None
    for policy in policies:
        attempt = {"policy": policy}
        attempts.append(attempt)
        try:
            free_parent = (
                policy.removeprefix("free_residue_template_")
                if policy.startswith("free_residue_template_")
                else None
            )
            if free_parent is not None:
                prepared = work / f"{free_parent}.pdb"
                if not prepared.is_file():
                    raise ValueError(
                        "No prepared PDB is available for the free-residue retry"
                    )
                if overlay_provenance is None:
                    overlay_provenance = build_free_residue_overlay(
                        python, work / "free_residue_templates.json"
                    )
                attempt["free_residue_templates"] = overlay_provenance
                attempt["parent_pdb"] = str(prepared)
                attempt["parent_pdb_sha256"] = hashlib.sha256(
                    prepared.read_bytes()
                ).hexdigest()
            elif policy == "direct":
                prepared = supported
            else:
                if not normalized.exists():
                    normalize_protein(supported, normalized)
                prepared = work / f"{policy}.pdb"
                if policy == "residue_template_retry":
                    repair_added_heavy_atoms(
                        work / "geometry_retry.pdb",
                        supported,
                        prepared,
                        residue_templates_dir,
                    )
                else:
                    repair_normalized_pdb(
                        normalized,
                        supported,
                        prepared,
                        rebuild_geometry=policy == "geometry_retry",
                    )
            prefix = work / policy
            candidate = prefix.with_suffix(".pdbqt")
            argv = [
                str(python),
                str(meeko),
                "-i" if policy == "direct" else "--read_pdb",
                str(prepared),
                "-o",
                str(prefix),
                "-p",
                "--allow_bad_res",
            ]
            if free_parent is not None:
                argv = [
                    str(python),
                    str(Path(__file__).with_name("free_residue_templates.py")),
                    "--backend",
                    overlay_provenance["extension"],
                    "--",
                    *argv[2:],
                ]
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                env={**os.environ, "MPLCONFIGDIR": "/tmp/chelate_mpl"},
            )
            log = work / f"{policy}.log"
            log.write_text(result.stdout + result.stderr)
            attempt.update(returncode=result.returncode, command=argv, log=str(log))
            if result.returncode or not candidate.is_file():
                raise RuntimeError("Meeko did not produce a valid receptor")
            ions = work / f"{policy}_unsupported.pdbqt"
            retention = retain_source_metals(fixed, unsupported, candidate, ions)
            merged = work / f"{policy}_merged.pdbqt"
            merge_pdbqt(candidate, ions, merged)
            attempt["chain_metal_audit"] = audit_repaired_receptor(
                fixed, candidate, ions, merged
            )
            full = audit_original_receptor(
                original, merged, target_ligand=ligand, pdbfixer_used=policy != "direct"
            )
            attempt["source_atom_audit"] = full
            if full["defective"]:
                raise ValueError(
                    "Candidate lost required source atoms or contains the target ligand"
                )
            attempt["status"] = "passed"
            selected = {
                "receptor": str(merged),
                "protein": str(candidate),
                "unsupported": str(ions),
                "fixed_source": str(fixed),
                "policy": policy,
                "source_atom_audit": full,
                "chain_metal_audit": attempt["chain_metal_audit"],
                "metal_retention": retention,
            }
            if free_parent is not None:
                selected["free_residue_templates"] = overlay_provenance
                selected["parent_pdb"] = attempt["parent_pdb"]
                selected["parent_pdb_sha256"] = attempt["parent_pdb_sha256"]
            break
        except Exception as error:
            attempt.update(status="failed", error=str(error))
        finally:
            (work / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    if selected is None:
        raise RuntimeError(
            f"No complete receptor for {original}, ligand {ligand}; see {work / 'attempts.json'}"
        )
    selected.update(
        prepared_record=str(record),
        original_source=str(original),
        target_ligand=ligand,
        original_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        receptor_sha256=hashlib.sha256(
            Path(selected["receptor"]).read_bytes()
        ).hexdigest(),
        attempts=attempts,
    )
    record.write_text(json.dumps(selected, indent=2) + "\n")
    return selected
