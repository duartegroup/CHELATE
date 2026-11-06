"""Small mmCIF helpers used during receptor preparation."""

import string

import gemmi


def sanitize_asym_id_columns_in_cif(input_cif: str, output_path: str) -> bool:
    """Map complete atom-site chain IDs to unique one-character identifiers.

    The mapping is shared by ``label_asym_id`` and ``auth_asym_id`` and follows the
    first appearance of each complete ID in the atom-site loop. Missing-value tokens
    are left unchanged. A PDB-compatible alphabet is used because the sanitized CIF
    is subsequently converted to PDB/PDBQT.
    """
    document = gemmi.cif.read_file(str(input_cif))
    block = document.sole_block()
    label_column = block.find_loop("_atom_site.label_asym_id")
    auth_column = block.find_loop("_atom_site.auth_asym_id")
    if not label_column or not auth_column:
        raise ValueError(
            "The atom-site loop must contain label_asym_id and auth_asym_id"
        )
    if label_column.get_loop() is not auth_column.get_loop():
        raise ValueError("The atom-site asym-ID columns are in different loops")

    complete_ids: list[str] = []
    seen: set[str] = set()
    for row_index in range(len(label_column)):
        for value in (label_column[row_index], auth_column[row_index]):
            if value not in {".", "?"} and value not in seen:
                seen.add(value)
                complete_ids.append(value)

    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if len(complete_ids) > len(alphabet):
        raise ValueError(
            f"Cannot map {len(complete_ids)} chain IDs to the "
            f"{len(alphabet)} available one-character identifiers"
        )
    asym_id_map = {value: alphabet[index] for index, value in enumerate(complete_ids)}
    for row_index in range(len(label_column)):
        label_value = label_column[row_index]
        auth_value = auth_column[row_index]
        if label_value in asym_id_map:
            label_column[row_index] = asym_id_map[label_value]
        if auth_value in asym_id_map:
            auth_column[row_index] = asym_id_map[auth_value]

    document.write_file(str(output_path))
    return True


def prepare_ligand_receptor_source(
    original_cif: str, ligand_id: str, output_dir: str
) -> dict[str, str]:
    """Build independent sanitizer inputs for exactly one ligand.

    The caller supplies a ligand-specific directory. Preparing another ligand
    must never replace this pair's source or change its chain map.
    """
    from pathlib import Path

    document = read_atom_site_document(str(original_cif))
    table = document.sole_block().find_mmcif_category("_atom_site.")
    indices = {tag.split(".", 1)[1]: i for i, tag in enumerate(table.tags)}
    if not {"label_asym_id", "label_comp_id"} <= indices.keys():
        raise ValueError("Original structure lacks ligand and residue identifiers")
    if ligand_id not in {row[indices["label_asym_id"]] for row in table}:
        raise ValueError(f"Target ligand chain is absent from original: {ligand_id}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "without_solvent": root / "system_wo_solvent.cif",
        "without_ligand": root / "system_wo_ligand.cif",
        "sanitized": root / "system_sanitized.cif",
    }
    for i in reversed(range(len(table))):
        if table[i][indices["label_comp_id"]] in {"HOH", "WAT", "H2O", "TIP", "SOL"}:
            table.remove_row(i)
    document.write_file(str(paths["without_solvent"]))
    for i in reversed(range(len(table))):
        if table[i][indices["label_asym_id"]] == ligand_id:
            table.remove_row(i)
    document.write_file(str(paths["without_ligand"]))
    sanitize_asym_id_columns_in_cif(
        str(paths["without_ligand"]), str(paths["sanitized"])
    )
    return {name: str(path) for name, path in paths.items()}


def read_atom_site_document(path: str) -> gemmi.cif.Document:
    """Read coordinates even when unrelated source metadata has broken quoting.

    Fall back only to the complete original atom-site loop. Its rows are still
    parsed strictly by Gemmi; malformed coordinates or atom identifiers are not
    repaired or skipped, and the source file is never changed.
    """
    try:
        return gemmi.cif.read_file(str(path))
    except ValueError as original_error:
        from pathlib import Path

        lines = Path(path).read_text().splitlines()
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if line.strip().startswith("_atom_site.")
            ),
            None,
        )
        if start is None:
            raise original_error
        end = start
        while end < len(lines) and lines[end].strip().startswith("_atom_site."):
            end += 1
        while end < len(lines):
            line = lines[end].strip()
            if line == "#" or line == "loop_" or line.startswith(("_", "data_")):
                break
            end += 1
        document = gemmi.cif.read_string(
            "data_original_atoms\nloop_\n" + "\n".join(lines[start:end]) + "\n"
        )
        table = document.sole_block().find_mmcif_category("_atom_site.")
        if not table or len(table) == 0:
            raise original_error
        return document
