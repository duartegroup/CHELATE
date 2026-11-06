"""Build free amino-acid templates from the pinned Meeko terminal templates.

Only terminal hydrogen/charge states are combined. Original heavy-atom
coordinates are never edited. The original template order remains first.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from rdkit import Chem


def smiles_atom_order(text, count):
    """RDKit versions may include a trailing comma in this internal list."""
    order = ast.literal_eval(text)
    if not isinstance(order, list) or sorted(order) != list(range(count)):
        raise ValueError("SMILES atom order is not a complete atom permutation")
    return order


def parse_template(template):
    settings = Chem.SmilesParserParams()
    settings.removeHs = False
    molecule = Chem.MolFromSmiles(template["smiles"], settings)
    if molecule is None or molecule.GetNumAtoms() != len(template["atom_name"]):
        raise ValueError("Invalid pinned residue template")
    return molecule


def combine_termini(n_terminal, c_terminal):
    """Use C-terminal carboxylate and the matching N-terminal amino group."""
    n_links = {
        n_terminal["atom_name"][int(i)]: v for i, v in n_terminal["link_labels"].items()
    }
    c_links = {
        c_terminal["atom_name"][int(i)]: v for i, v in c_terminal["link_labels"].items()
    }
    if n_links != {"C": "C-term"} or c_links != {"N": "N-term"}:
        raise ValueError("Template has additional or unexpected covalent links")
    n_molecule = parse_template(n_terminal)
    c_molecule = parse_template(c_terminal)
    n_names = n_terminal["atom_name"]
    names = list(c_terminal["atom_name"])
    n_atom = n_molecule.GetAtomWithIdx(n_names.index("N"))
    edit = Chem.RWMol(c_molecule)
    index = names.index("N")
    remove = [
        a.GetIdx()
        for a in edit.GetAtomWithIdx(index).GetNeighbors()
        if a.GetAtomicNum() == 1
    ]
    for i in sorted(remove, reverse=True):
        edit.RemoveAtom(i)
        names.pop(i)
    index = names.index("N")
    atom = edit.GetAtomWithIdx(index)
    atom.SetFormalCharge(n_atom.GetFormalCharge())
    atom.SetNumExplicitHs(n_atom.GetNumExplicitHs())
    atom.SetNoImplicit(True)
    for neighbor in n_atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 1:
            continue
        added = edit.AddAtom(Chem.Atom(neighbor))
        edit.AddBond(index, added, Chem.BondType.SINGLE)
        names.append(n_names[neighbor.GetIdx()])
    molecule = edit.GetMol()
    Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(molecule, canonical=False, allHsExplicit=True)
    order = smiles_atom_order(
        molecule.GetProp("_smilesAtomOutputOrder"), molecule.GetNumAtoms()
    )
    result = {
        "smiles": smiles,
        "atom_name": [names[i] for i in order],
        "link_labels": {},
    }
    check = parse_template(result)
    if any(a.GetTotalNumHs() for a in check.GetAtoms()):
        raise ValueError("Combined free residue still requires a covalent linker")
    return result


def build_extension(library):
    original = library["residue_templates"]
    added = {}
    mapping = {}
    for base in original:
        if "N" + base not in original or "C" + base not in original:
            continue
        n_template = original["N" + base]
        c_template = original["C" + base]
        n_links = {
            n_template["atom_name"][int(i)]: v
            for i, v in n_template["link_labels"].items()
        }
        c_links = {
            c_template["atom_name"][int(i)]: v
            for i, v in c_template["link_labels"].items()
        }
        if n_links != {"C": "C-term"} or c_links != {"N": "N-term"}:
            continue
        combined = combine_termini(n_template, c_template)
        key = "FREE_" + base
        if key in original:
            raise ValueError("Combined template name already exists")
        added[key] = combined
        mapping[base] = key
    ambiguous = {}
    for residue, choices in library["ambiguous"].items():
        new = [mapping[k] for k in choices if k in mapping]
        if new:
            ambiguous[residue] = list(choices) + new
    return {"residue_templates": added, "ambiguous": ambiguous}


def run_backend(overlay, arguments):
    """Extend defaults in this subprocess only; retain the pinned CLI backend."""
    import sys

    from meeko import ResidueChemTemplates
    from meeko.cli.mk_prepare_receptor import main

    original = ResidueChemTemplates.create_from_defaults.__func__
    payload = json.loads(Path(overlay).read_text())

    def defaults(cls):
        templates = original(cls)
        if set(payload["residue_templates"]) & set(templates.residue_templates):
            raise ValueError("Free-residue overlay would replace an existing template")
        for residue, choices in payload["ambiguous"].items():
            previous = templates.ambiguous.get(residue, [])
            if choices[: len(previous)] != previous:
                raise ValueError(
                    "Free-residue overlay would change the original template order"
                )
            if any(
                name not in payload["residue_templates"]
                for name in choices[len(previous) :]
            ):
                raise ValueError("Free-residue overlay adds an unknown template")
        templates.add_dict(copy.deepcopy(payload), overwrite=True)
        return templates

    original_factory = ResidueChemTemplates.__dict__["create_from_defaults"]
    previous_argv = sys.argv
    try:
        ResidueChemTemplates.create_from_defaults = classmethod(defaults)
        sys.argv = ["mk_prepare_receptor.py", *arguments]
        main()
    finally:
        ResidueChemTemplates.create_from_defaults = original_factory
        sys.argv = previous_argv


def build_free_residue_overlay(python, output):
    """Build in the requested Meeko/RDKit environment and record its provenance."""
    script = Path(__file__).resolve()
    result = subprocess.run(
        [str(python), str(script), "--build", str(output)],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if result.returncode:
        raise RuntimeError(
            "Cannot build free-residue templates: "
            + (result.stdout + result.stderr)[-3000:]
        )
    provenance = json.loads(result.stdout)
    if (
        hashlib.sha256(Path(output).read_bytes()).hexdigest()
        != provenance["extension_sha256"]
    ):
        raise ValueError("Free-residue template extension changed after building")
    return provenance


def write_extension(output):
    """Read the installed Meeko library without changing any installed file."""
    spec = importlib.util.find_spec("meeko")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Meeko is unavailable in the requested preparation environment"
        )
    source = Path(spec.origin).parent / "data" / "residue_chem_templates.json"
    original = source.read_bytes()
    extension = build_extension(json.loads(original))
    content = (json.dumps(extension, indent=2) + "\n").encode()
    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError(
                "Existing free-residue extension has different contents"
            )
    else:
        with output.open("xb") as handle:
            handle.write(content)
    return {
        "template_library": str(source),
        "template_library_sha256": hashlib.sha256(original).hexdigest(),
        "extension": str(output),
        "extension_sha256": hashlib.sha256(content).hexdigest(),
        "backend_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "combined_templates": len(extension["residue_templates"]),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        print(json.dumps(write_extension(Path(sys.argv[2]))))
    elif len(sys.argv) > 4 and sys.argv[1] == "--backend" and sys.argv[3] == "--":
        run_backend(sys.argv[2], sys.argv[4:])
    else:
        raise SystemExit("Use --build OUTPUT or --backend OVERLAY -- MEEKO_ARGUMENTS")
