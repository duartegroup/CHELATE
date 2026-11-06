"""Reject incomplete cached or newly prepared receptors before a docking engine runs."""

from pathlib import Path

from docking.receptor_integrity import audit_original_receptor


def require_complete_receptor(
    original: str | Path,
    receptor: str | Path,
    target_ligand: str | None,
    *,
    docking_box: dict | None = None,
) -> dict:
    """Check the direct Meeko path used by run_docking, including existing caches.

    This entry point has no PDBFixer fallback, so its source comparison uses the
    direct-preparation keep rules. The audited fallback API in prepare_pairs
    records and checks its own explicit PDBFixer policy. Never assume that a
    successful Meeko exit status means that all source residues were retained.
    """
    result = audit_original_receptor(
        Path(original),
        Path(receptor),
        target_ligand=target_ligand,
        pdbfixer_used=False,
        docking_box=docking_box,
    )
    if result["defective"]:
        prefix = "binding_site_" if docking_box is not None else ""
        raise ValueError(
            "Original-source receptor check failed: "
            f"protein={result[prefix + 'missing_protein_atoms']}, "
            f"metal={result[prefix + 'missing_metal_atoms']}, "
            f"other={result[prefix + 'missing_other_atoms']}, "
            f"target_ligand={result['target_ligand_atoms_present']}. "
            "Prepare a complete isolated input with docking.prepare_pairs before docking; "
            "archive existing receptors and poses before installing a repaired input."
        )
    return result
