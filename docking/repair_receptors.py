"""Recover redocking receptors that failed in the CIF-to-Meeko path.

The failed structures contain nonstandard polymer residues (most commonly MSE)
or small non-receptor heterogens that make Meeko abort.  This script converts
MSE to MET without moving coordinates, removes non-polymer residues from the
protein input, runs Meeko, and merges back the unsupported atoms already
extracted by the docking pipeline (including the metal ions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import string
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem

from chelate.constants import METAL_ELEMENT_SYMBOLS

REPO = Path(__file__).resolve().parents[1]
# Folder holding the {split}_systems folders, as for run_docking.py.
DEFAULT_DATA_ROOT = Path(os.environ.get("SYSTEMS_ROOT", REPO / "docking"))
DEFAULT_RUNTIME = REPO / ".runtime" / "receptor_repair"
DEFAULT_MEEKO = Path(
    os.environ.get(
        "MK_PREPARE_RECEPTOR_PATH",
        REPO / ".envs/docking_env/bin/mk_prepare_receptor.py",
    )
)
DEFAULT_PYTHON = Path(
    os.environ.get("DOCKING_PYTHON", REPO / ".envs/docking_env/bin/python")
)
DEFAULT_SHARDS = REPO / ".runtime" / "docking_shards"
DEFAULT_ERROR_FILENAME = (
    "errors_GNINA_redocked_FH_id=1_exhaust=32_numModes=100_prepPH=7.4.csv"
)


def failed_tasks(
    split: str,
    data_root: Path,
    shards_dir: Path,
    error_filename: str,
) -> dict[str, set[str]]:
    systems_dir = data_root / f"{split}_systems"
    error_path = systems_dir / error_filename
    errors = pd.read_csv(error_path, keep_default_na=False)
    failed_ids = set(
        errors.loc[
            errors["error_message"].str.startswith(
                "Error preparing receptor with Meeko"
            ),
            "system_id",
        ]
    )
    rows = pd.concat(
        [
            pd.read_csv(path, keep_default_na=False)
            for path in sorted(shards_dir.glob(f"{split}_shard*.csv"))
        ],
        ignore_index=True,
    )
    tasks: dict[str, set[str]] = {}
    for row in rows.itertuples(index=False):
        if row.system_id in failed_ids:
            ligand = f"{row.ligand_instance}.{row.ligand_asym_id}"
            output = systems_dir / row.system_id / f"receptor_merged_{ligand}.pdbqt"
            output_is_valid = False
            if output.is_file() and output.stat().st_size > 0:
                try:
                    validate_pdbqt(output)
                    output_is_valid = True
                except (OSError, ValueError):
                    pass
            if not output_is_valid:
                tasks.setdefault(str(row.system_id), set()).add(ligand)
    return tasks


def normalize_protein(source: Path, output: Path) -> None:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(source))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Sanitizer output already has unique PDB-compatible chain IDs. Keep
    # them so protein and separately prepared metal records use the same map.
    chain_ids = string.ascii_uppercase + string.ascii_lowercase + string.digits
    chains = list(fixer.topology.chains())
    if len(chains) > len(chain_ids):
        raise ValueError(f"Too many chains for PDB output: {len(chains)}")
    existing_ids = [str(chain.id) for chain in chains]
    can_preserve = len(set(existing_ids)) == len(existing_ids) and all(
        len(chain_id) == 1 and chain_id in chain_ids for chain_id in existing_ids
    )
    if not can_preserve:
        for chain, chain_id in zip(chains, chain_ids, strict=False):
            chain.id = chain_id

    with output.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)


def repair_normalized_pdb(
    source: Path, source_cif: Path, output: Path, *, rebuild_geometry: bool = False
) -> dict:
    """Normalize hydrogen isotopes and optionally rebuild hydrogen/OXT input.

    The optional retry removes supplied hydrogens so Meeko assigns them from its
    standard templates. It reflects only added terminal OXT atoms across the
    CA-C axis, keeping the observed carbonyl bond length and peptide plane.
    Original heavy atoms, including an original OXT, are never moved.
    """
    block = gemmi.cif.read_file(str(source_cif)).sole_block()
    chains = list(block.find_values("_atom_site.auth_asym_id"))
    numbers = list(block.find_values("_atom_site.auth_seq_id"))
    names = list(block.find_values("_atom_site.label_atom_id"))
    insertions = list(block.find_values("_atom_site.pdbx_PDB_ins_code"))
    if not chains or len(chains) != len(numbers) or len(chains) != len(names):
        raise ValueError("Source CIF lacks complete atom identity columns")
    original = set()
    for index, (chain, number, name) in enumerate(
        zip(chains, numbers, names, strict=True)
    ):
        insertion = insertions[index] if insertions else ""
        if insertion in {".", "?"}:
            insertion = ""
        original.add((chain, number + insertion, name.strip("\"'")))

    lines = source.read_text().splitlines()
    changes = {
        "deuteriums_normalized": 0,
        "hydrogens_removed": 0,
        "added_terminal_oxygens_rebuilt": [],
    }
    residues = {}
    for index, line in enumerate(lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip()
        if element == "D":
            changes["deuteriums_normalized"] += 1
            lines[index] = line[:76] + " H" + line[78:]
        key = line[21:22], line[22:27].strip()
        residues.setdefault(key, {})[line[12:16].strip()] = index

    if rebuild_geometry:
        for key, atoms in residues.items():
            if "OXT" not in atoms or (*key, "OXT") in original:
                continue
            if not all(name in atoms for name in ("CA", "C", "O")):
                raise ValueError(f"Added OXT lacks CA/C/O anchors: {key}")

            def xyz(name, atom_indices=atoms):
                line = lines[atom_indices[name]]
                return [float(line[start : start + 8]) for start in (30, 38, 46)]

            ca, carbon, oxygen = xyz("CA"), xyz("C"), xyz("O")
            axis = [carbon[i] - ca[i] for i in range(3)]
            bond = [oxygen[i] - carbon[i] for i in range(3)]
            norm = sum(value * value for value in axis)
            if norm < 1e-12:
                raise ValueError(f"Degenerate CA-C geometry: {key}")
            scale = sum(axis[i] * bond[i] for i in range(3)) / norm
            position = [carbon[i] + 2 * scale * axis[i] - bond[i] for i in range(3)]
            input_position = xyz("OXT")
            index = atoms["OXT"]
            lines[index] = (
                lines[index][:30]
                + "".join(f"{value:8.3f}" for value in position)
                + lines[index][54:]
            )
            changes["added_terminal_oxygens_rebuilt"].append(
                {
                    "chain": key[0],
                    "residue": key[1],
                    "before": input_position,
                    "after": position,
                }
            )
        kept = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "H":
                changes["hydrogens_removed"] += 1
            else:
                kept.append(line)
        lines = kept
    output.write_text("\n".join(lines) + "\n")
    return changes


def audit_repaired_receptor(
    sanitized_cif: Path, protein: Path, unsupported: Path, merged: Path
) -> dict:
    """Check chain IDs and every separately prepared unsupported record."""
    validate_pdbqt(protein)
    validate_pdbqt(merged)
    block = gemmi.cif.read_file(str(sanitized_cif)).sole_block()
    allowed = {
        str(value)
        for tag in ("_atom_site.label_asym_id", "_atom_site.auth_asym_id")
        for value in block.find_values(tag)
        if value not in {".", "?"}
    }
    if not allowed or any(len(chain) != 1 for chain in allowed):
        raise ValueError("Sanitizer output has invalid chain IDs")

    def records(path):
        return [
            line
            for line in path.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]

    protein_lines = records(protein)
    unsupported_lines = records(unsupported)
    merged_lines = records(merged)
    observed = {
        line[21:22] for line in protein_lines + unsupported_lines + merged_lines
    }
    if not observed <= allowed:
        raise ValueError(
            f"Receptor chain IDs differ from sanitizer output: {observed - allowed}"
        )

    def signature(line):
        return line[12:16], line[17:20], line[21:27], line[30:54], line.split()[-1]

    missing = Counter(signature(line) for line in unsupported_lines) - Counter(
        signature(line) for line in merged_lines
    )
    if missing:
        raise ValueError(
            "Merged receptor lost separately prepared metal/unsupported records"
        )
    remaining = Counter(
        metal_site_signature(site) for site in source_metal_ions(sanitized_cif)
    ) - Counter(pdbqt_metal_sites(merged))
    if remaining:
        raise ValueError("Merged receptor lost a metal ion from the sanitizer output")
    return {
        "protein_atoms": len(protein_lines),
        "unsupported_atoms": len(unsupported_lines),
        "merged_atoms": len(merged_lines),
        "chains": sorted(observed),
    }


def repair_added_heavy_atoms(
    source: Path, source_cif: Path, output: Path, template_dir: Path
) -> dict:
    """Fit standard residue templates to observed atoms and rebuild added atoms.

    Only atoms absent from the source coordinates can move. This also preserves
    original atoms renamed during nonstandard-residue replacement, such as SE
    renamed to SD. At least three non-collinear observed anchors are required.
    """
    block = gemmi.cif.read_file(str(source_cif)).sole_block()
    table = block.find_mmcif_category("_atom_site.")
    columns = {tag.split(".", 1)[1]: index for index, tag in enumerate(table.tags)}
    original = {}
    for row in table:
        if row[columns["type_symbol"]] in {"H", "D"}:
            continue
        insertion = (
            row[columns["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in columns else ""
        )
        if insertion in {".", "?"}:
            insertion = ""
        key = row[columns["auth_asym_id"]], row[columns["auth_seq_id"]] + insertion
        original.setdefault(key, set()).add(
            tuple(round(float(row[columns["Cartn_" + axis]]), 3) for axis in "xyz")
        )

    def xyz(line):
        return np.array([float(line[start : start + 8]) for start in (30, 38, 46)])

    lines = source.read_text().splitlines()
    residues = {}
    for index, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            key = line[21:22], line[22:27].strip(), line[17:20].strip()
            residues.setdefault(key, {})[line[12:16].strip()] = index
    changes = []
    template_hashes = {}
    skipped = []
    for (chain, number, name), atoms in residues.items():
        added = [
            atom
            for atom, index in atoms.items()
            if tuple(xyz(lines[index])) not in original.get((chain, number), set())
            and atom != "OXT"
            and lines[index][76:78].strip() not in {"H", "D"}
        ]
        if not added:
            continue
        template = template_dir / f"{name}.pdb"
        if not template.is_file():
            skipped.append([chain, number, name, "No standard residue template"])
            continue
        positions = {
            line[12:16].strip(): xyz(line)
            for line in template.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM"))
        }
        anchors = [
            atom
            for atom in atoms
            if atom in positions and atom not in added and atom != "OXT"
        ]
        if len(anchors) < 3:
            skipped.append([chain, number, name, "Fewer than three observed anchors"])
            continue
        reference = np.array([positions[atom] for atom in anchors])
        observed = np.array([xyz(lines[atoms[atom]]) for atom in anchors])
        reference_center, observed_center = reference.mean(0), observed.mean(0)
        reference -= reference_center
        observed -= observed_center
        if np.linalg.matrix_rank(reference) < 2 or np.linalg.matrix_rank(observed) < 2:
            skipped.append([chain, number, name, "Collinear observed anchors"])
            continue
        left, _, right = np.linalg.svd(reference.T @ observed)
        orientation = np.eye(3)
        orientation[2, 2] = np.linalg.det(left @ right)
        rotation = left @ orientation @ right
        template_hashes[str(template)] = hashlib.sha256(
            template.read_bytes()
        ).hexdigest()
        for atom in added:
            if atom not in positions:
                continue
            index = atoms[atom]
            before = xyz(lines[index])
            after = (positions[atom] - reference_center) @ rotation + observed_center
            lines[index] = (
                lines[index][:30]
                + "".join(f"{value:8.3f}" for value in after)
                + lines[index][54:]
            )
            changes.append(
                {
                    "chain": chain,
                    "residue": number,
                    "residue_name": name,
                    "atom": atom,
                    "before": before.tolist(),
                    "after": after.tolist(),
                    "anchors": anchors,
                }
            )
    output.write_text("\n".join(lines) + "\n")
    return {
        "atoms_rebuilt": changes,
        "template_sha256": template_hashes,
        "skipped": skipped,
    }


def source_metal_ions(sanitized_cif: Path) -> list[dict]:
    """Read monatomic metal residues from the sanitized, nonperiodic CIF frame."""
    block = gemmi.cif.read_file(str(sanitized_cif)).sole_block()
    table = block.find_mmcif_category("_atom_site.")
    columns = {tag: index for index, tag in enumerate(table.tags)}
    required = [
        "type_symbol",
        "auth_asym_id",
        "auth_seq_id",
        "label_atom_id",
        "label_comp_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]
    if not all("_atom_site." + name in columns for name in required):
        raise ValueError("Source CIF lacks metal identity or coordinate columns")
    residues = {}
    for row in table:

        def value(name, atom_row=row):
            return str(atom_row[columns["_atom_site." + name]]).strip("\"'")

        symbol = value("type_symbol").capitalize()
        if symbol in {"H", "D"}:
            continue
        key = value("auth_asym_id"), value("auth_seq_id"), value("label_comp_id")
        residues.setdefault(key, []).append(
            {
                "element": symbol,
                "chain": key[0],
                "residue_number": key[1],
                "residue_name": key[2],
                "atom_name": value("label_atom_id"),
                "xyz": [float(value("Cartn_" + axis)) for axis in "xyz"],
                "formal_charge": value("pdbx_formal_charge")
                if "_atom_site.pdbx_formal_charge" in columns
                else "?",
            }
        )
    return [
        atoms[0]
        for atoms in residues.values()
        if len(atoms) == 1 and atoms[0]["element"] in METAL_ELEMENT_SYMBOLS
    ]


def metal_site_signature(site: dict) -> tuple:
    return site["element"], site["chain"], *(round(value, 3) for value in site["xyz"])


def pdbqt_metal_sites(path: Path) -> list[tuple]:
    """Use case-sensitive AutoDock atom types, distinguishing NA from sodium Na."""
    sites = []
    for line in path.read_text().splitlines():
        if (
            line.startswith(("ATOM", "HETATM"))
            and line.split()[-1] in METAL_ELEMENT_SYMBOLS
        ):
            sites.append(
                (
                    line.split()[-1],
                    line[21:22],
                    *(
                        round(float(line[start : start + 8]), 3)
                        for start in (30, 38, 46)
                    ),
                )
            )
    return sites


def retain_source_metals(
    sanitized_cif: Path,
    source_unsupported: Path,
    protein: Path,
    output: Path,
    *,
    template_json: Path | None = None,
) -> dict:
    """Restore supported metal ions removed by PDBFixer's removeHeterogens.

    Existing unsupported-ion records retain their charges. Missing supported ions
    use the same Meeko residue-template formal charge, with explicit source-CIF
    formal charge as a fallback. No guessed oxidation states are introduced.
    """
    if template_json is None:
        import meeko

        template_json = Path(meeko.__file__).parent / "data/residue_chem_templates.json"
    templates = json.loads(template_json.read_text())["residue_templates"]
    normalize_unsupported(source_unsupported, output)
    present = Counter(pdbqt_metal_sites(protein) + pdbqt_metal_sites(output))
    lines = output.read_text().splitlines()
    added = []
    for site in source_metal_ions(sanitized_cif):
        key = metal_site_signature(site)
        if present[key]:
            present[key] -= 1
            continue
        template = templates.get(site["residue_name"])
        molecule = Chem.MolFromSmiles(template["smiles"]) if template else None
        if (
            molecule is not None
            and molecule.GetNumAtoms() == 1
            and molecule.GetAtomWithIdx(0).GetSymbol() == site["element"]
        ):
            charge = molecule.GetAtomWithIdx(0).GetFormalCharge()
            charge_source = "Meeko residue template " + site["residue_name"]
        elif site["formal_charge"] not in {".", "?", ""}:
            charge = float(site["formal_charge"])
            charge_source = "Explicit source CIF formal charge"
        else:
            raise ValueError(f"No charge source for missing metal ion: {site}")
        try:
            number = int(site["residue_number"])
        except ValueError:
            number = 0
        x, y, z = site["xyz"]
        serial = len(lines) + 1
        lines.append(
            f"HETATM{serial:5d} {site['atom_name']:>4s} {site['residue_name']:>3s} "
            f"{site['chain']}{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"    {charge:6.3f} {site['element']:<2s}"
        )
        added.append(dict(site, charge=charge, charge_source=charge_source))
    output.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {
        "ions_added": added,
        "metal_template_json": str(template_json),
        "metal_template_sha256": hashlib.sha256(template_json.read_bytes()).hexdigest(),
    }


def publish_staged_repair(details: dict) -> None:
    """Install the files of an audited repair in the system folder.

    The source CIF must be unchanged since the repair was audited.
    """
    if (
        hashlib.sha256(Path(details["source_cif"]).read_bytes()).hexdigest()
        != details["source_cif_sha256"]
    ):
        raise ValueError("Source CIF changed after repair staging")
    for source, destination in details["publish_files"]:
        publish_repair(Path(source), Path(destination))


def publish_repair(source: Path, destination: Path) -> None:
    """Install a validated file, keeping any existing file with a timestamp suffix."""
    temporary = destination.with_name(destination.name + f".repair_{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    if destination.exists():
        destination.rename(
            destination.with_name(destination.name + f".before_repair_{time.time_ns()}")
        )
    os.replace(temporary, destination)


def normalize_unsupported(source: Path, output: Path) -> None:
    lines: list[str] = []
    if source.is_file():
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 13 or fields[0] not in {"ATOM", "HETATM"}:
                continue
            record, serial, atom, residue, chain, residue_number = fields[:6]
            x, y, z, occupancy, temperature, charge = map(float, fields[6:12])
            atom_type = fields[12]
            lines.append(
                f"{record:<6}{int(serial):5d} {atom:>4s} {residue:>3s} "
                f"{chain[:1]:1s}{int(residue_number):4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temperature:6.2f}"
                f"    {charge:6.3f} {atom_type:<2s}\n"
            )
    output.write_text("".join(lines))


def merge_pdbqt(protein: Path, unsupported: Path, output: Path) -> None:
    protein_lines = protein.read_text().splitlines()
    unsupported_lines = unsupported.read_text().splitlines()
    merged = [line for line in protein_lines if line.strip()]
    next_serial = sum(line.startswith(("ATOM", "HETATM")) for line in merged) + 1
    for line in unsupported_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        merged.append(f"{line[:6]}{next_serial:5d}{line[11:]}")
        next_serial += 1
    output.write_text("\n".join(merged) + "\n")


def preserve_pdbqt_atom_identity(pdbqt: Path, pdb: Path) -> int:
    """Keep source atom identity when OpenBabel joins identical ion residues.

    This is serialization only: coordinates, elements and inferred bonds stay as
    written by OpenBabel. Each output atom must match its unique source serial
    and coordinates before its original PDB identity fields can be copied.
    """
    source = {}
    for line in pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial = int(line[6:11])
            if serial in source:
                raise ValueError("Duplicate source receptor atom serial")
            source[serial] = line
    output, seen, changed = [], set(), 0
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial = int(line[6:11])
            if serial not in source or serial in seen:
                raise ValueError("Converted receptor atom serials differ from source")
            original = source[serial]
            # OpenBabel normalizes -0.000 to 0.000 without moving an atom.
            # Require exact numeric equality, including at 0.001 A precision.
            source_xyz = tuple(float(original[i : i + 8]) for i in (30, 38, 46))
            converted_xyz = tuple(float(line[i : i + 8]) for i in (30, 38, 46))
            if source_xyz != converted_xyz or not all(
                math.isfinite(value) for value in source_xyz + converted_xyz
            ):
                raise ValueError("Converted receptor coordinates differ from source")
            seen.add(serial)
            changed += line[21:22] != original[21:22]
            line = original[:30] + line[30:]
        output.append(line)
    if seen != set(source):
        raise ValueError("Receptor atoms lost during PDB conversion")
    pdb.write_text("\n".join(output) + "\n")
    return changed


def validate_pdbqt(path: Path) -> None:
    atoms = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
            try:
                if not all(
                    math.isfinite(float(line[start : start + 8]))
                    for start in (30, 38, 46)
                ):
                    raise ValueError("Non-finite receptor coordinates")
            except ValueError as error:
                raise ValueError(
                    f"Invalid coordinates in {path}, line {line_number}: {line}"
                ) from error
    if atoms == 0:
        raise ValueError(f"No receptor atoms in {path}")


def repair_system(
    split: str,
    system_id: str,
    ligands: set[str],
    data_root: Path,
    python: Path,
    meeko: Path,
    *,
    retry_normalized: bool = False,
    retry_direct: bool = False,
    repair_geometry: bool = False,
    work_dir: Path | None = None,
    retain_metals_only: bool = False,
    stage_only: bool = False,
    metal_template_json: Path | None = None,
    residue_templates_dir: Path | None = None,
) -> tuple[str, dict]:
    system_dir = data_root / f"{split}_systems" / system_id
    source = system_dir / "system_wo_unsupported.cif"
    normalized_pdb = system_dir / "system_wo_unsupported_normalized.pdb"
    protein_pdbqt = system_dir / "system_wo_unsupported_normalized.pdbqt"
    unsupported = system_dir / "system_unsupported.pdbqt"
    normalized_unsupported = system_dir / "system_unsupported_normalized.pdbqt"
    if retry_direct:
        protein_pdbqt = system_dir / "system_wo_unsupported.pdbqt"

    root = work_dir if work_dir is not None else system_dir / "receptor_repair_work"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{split}_{system_id}_", dir=root))
    if retry_direct:
        original_pdb = source
    elif retry_normalized or retain_metals_only:
        if not normalized_pdb.is_file():
            raise FileNotFoundError(f"No existing PDBFixer output: {normalized_pdb}")
        original_pdb = normalized_pdb
    else:
        original_pdb = work / "pdbfixer_original.pdb"
        normalize_protein(source, original_pdb)
    attempts = []
    selected_pdb = None
    selected_protein = None
    if retain_metals_only:
        validate_pdbqt(protein_pdbqt)
        selected_pdb = original_pdb
        selected_protein = protein_pdbqt
    policies = ["isotope_normalization"]
    if repair_geometry:
        policies.append("geometry_retry")
        if residue_templates_dir is not None:
            policies.append("residue_template_retry")
    if retain_metals_only:
        policies = []
    if retry_direct:
        policies = ["direct_retry"]
    for name in policies:
        prepared_pdb = work / f"{name}.pdb"
        if name == "direct_retry":
            prepared_pdb = prepared_pdb.with_suffix(".cif")
            shutil.copy2(source, prepared_pdb)
            changes = {"source_cif": str(source)}
        elif name == "residue_template_retry":
            changes = repair_added_heavy_atoms(
                work / "geometry_retry.pdb", source, prepared_pdb, residue_templates_dir
            )
        else:
            changes = repair_normalized_pdb(
                original_pdb,
                source,
                prepared_pdb,
                rebuild_geometry=name == "geometry_retry",
            )
        candidate = work / f"{name}.pdbqt"
        result = subprocess.run(
            [
                str(python),
                str(meeko),
                "-i" if retry_direct else "--read_pdb",
                str(prepared_pdb),
                "-o",
                str(candidate.with_suffix("")),
                "-p",
                "--allow_bad_res",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "MPLCONFIGDIR": "/tmp"},
        )
        log = work / f"{name}.log"
        log.write_text(result.stdout + result.stderr)
        attempt = {
            "policy": name,
            "changes": changes,
            "returncode": result.returncode,
            "log": str(log),
        }
        attempts.append(attempt)
        if result.returncode == 0 and candidate.is_file():
            try:
                validate_pdbqt(candidate)
            except ValueError as error:
                attempt["validation_error"] = str(error)
            else:
                selected_pdb, selected_protein = prepared_pdb, candidate
                break
    (work / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    if selected_protein is None:
        raise RuntimeError(
            f"No valid Meeko receptor for {split}/{system_id}; attempts: {work / 'attempts.json'}"
        )
    candidate_unsupported = work / "unsupported.pdbqt"
    metal_retention = retain_source_metals(
        system_dir / "system_sanitized.cif",
        unsupported,
        selected_protein,
        candidate_unsupported,
        template_json=metal_template_json,
    )
    audits = {}
    for ligand in sorted(ligands):
        output = work / f"receptor_merged_{ligand}.pdbqt"
        merge_pdbqt(selected_protein, candidate_unsupported, output)
        audits[ligand] = audit_repaired_receptor(
            system_dir / "system_sanitized.cif",
            selected_protein,
            candidate_unsupported,
            output,
        )
    details = {
        "split": split,
        "system": system_id,
        "ligands": sorted(ligands),
        "attempts": attempts,
        "audits": audits,
        "metal_retention": metal_retention,
        "source_cif": str(source),
        "source_cif_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "work": str(work),
    }
    publish_files = []
    if not retain_metals_only:
        if not retry_direct:
            publish_files.append((selected_pdb, normalized_pdb))
        publish_files.append((selected_protein, protein_pdbqt))
    publish_files.append((candidate_unsupported, normalized_unsupported))
    publish_files.extend(
        (
            work / f"receptor_merged_{ligand}.pdbqt",
            system_dir / f"receptor_merged_{ligand}.pdbqt",
        )
        for ligand in sorted(ligands)
    )
    details["publish_files"] = [
        [str(source), str(destination)] for source, destination in publish_files
    ]
    details["published"] = not stage_only
    (work / "repair_audit.json").write_text(json.dumps(details, indent=2) + "\n")
    if not stage_only:
        publish_staged_repair(details)
    return system_id, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--meeko", type=Path, default=DEFAULT_MEEKO)
    parser.add_argument("--error-filename", default=DEFAULT_ERROR_FILENAME)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=("train", "val")
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--tasks-json",
        type=Path,
        help="Explicit pair manifest containing split, system_id and ligand_id",
    )
    parser.add_argument(
        "--retry-normalized",
        action="store_true",
        help="Reuse the PDBFixer output saved by the failed attempt",
    )
    parser.add_argument(
        "--retry-direct",
        action="store_true",
        help="Repeat direct Meeko preparation from the current source CIF",
    )
    parser.add_argument(
        "--repair-input-geometry",
        action="store_true",
        help="After a failure, rebuild input hydrogens and added terminal OXT geometry",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Finish with a recorded failure list when some receptors remain blocked",
    )
    parser.add_argument(
        "--retain-metals-only",
        action="store_true",
        help="Keep the prepared protein and restore missing source metal ions",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Write audited repairs to the work directory without replacing existing receptors",
    )
    parser.add_argument(
        "--metal-template-json",
        type=Path,
        help="Meeko residue-template JSON used for metal charges",
    )
    parser.add_argument(
        "--repair-residue-templates",
        type=Path,
        help="After geometry retry fails, fit standard PDBFixer residue templates to observed atoms",
    )
    args = parser.parse_args()
    if args.repair_residue_templates and not args.repair_input_geometry:
        parser.error("--repair-residue-templates requires --repair-input-geometry")
    if args.retry_direct and (
        args.retry_normalized or args.retain_metals_only or args.repair_input_geometry
    ):
        parser.error("--retry-direct cannot be combined with PDBFixer retry options")

    tasks: list[tuple[str, str, set[str]]] = []
    for split in [] if args.tasks_json else args.splits:
        for system_id, ligands in failed_tasks(
            split,
            args.data_root,
            args.shards_dir,
            args.error_filename,
        ).items():
            tasks.append((split, system_id, ligands))
    if args.tasks_json:
        grouped = {}
        for row in json.loads(args.tasks_json.read_text()):
            if row["split"] in args.splits:
                grouped.setdefault((row["split"], row["system_id"]), set()).add(
                    row["ligand_id"]
                )
        tasks = [
            (split, system_id, ligands)
            for (split, system_id), ligands in sorted(grouped.items())
        ]

    failures: list[str] = []
    results = []
    repaired = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                repair_system,
                split,
                system_id,
                ligands,
                args.data_root,
                args.python,
                args.meeko,
                retry_normalized=args.retry_normalized,
                retry_direct=args.retry_direct,
                repair_geometry=args.repair_input_geometry,
                work_dir=args.runtime_dir / "receptor_repair_work",
                retain_metals_only=args.retain_metals_only,
                stage_only=args.stage_only,
                metal_template_json=args.metal_template_json,
                residue_templates_dir=args.repair_residue_templates,
            ): (
                split,
                system_id,
            )
            for split, system_id, ligands in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            split, system_id = futures[future]
            try:
                _, detail = future.result()
                results.append(dict(detail, status="success"))
                repaired += 1
            except Exception as error:
                failures.append(f"{split}/{system_id}: {error}")
                results.append(
                    {
                        "split": split,
                        "system": system_id,
                        "status": "failed",
                        "error": str(error),
                    }
                )
            if index % 25 == 0 or index == len(tasks):
                print(
                    f"Receptor recovery: {index}/{len(tasks)} checked, "
                    f"{repaired} repaired, {len(failures)} failed",
                    flush=True,
                )

    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.runtime_dir / "receptor_repair_failures.txt"
    failure_path.write_text("\n".join(failures) + ("\n" if failures else ""))
    (args.runtime_dir / "repair_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    if failures and not args.allow_failures:
        raise RuntimeError(
            f"{len(failures)} receptor repairs failed; see {failure_path}"
        )


if __name__ == "__main__":
    main()
