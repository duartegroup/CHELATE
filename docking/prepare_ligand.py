"""Run Meeko with a fallback for metal-removal caps on dative donors."""

import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

REMOVED_METALS = (12, 20, 25, 26, 30)


def has_uncapped_dative_donor(molecule):
    """A donor loses no valence when its zero-contribution metal bond is removed."""
    return any(
        atom.GetAtomicNum() in REMOVED_METALS
        and neighbor.GetAtomicNum() not in REMOVED_METALS
        and molecule.GetBondBetweenAtoms(
            atom.GetIdx(), neighbor.GetIdx()
        ).GetValenceContrib(neighbor)
        == 0
        for atom in molecule.GetAtoms()
        for neighbor in atom.GetNeighbors()
    )


def corrected_metal_charges(molecule):
    """Use Meeko's auxiliary metal removal without hydrogenating dative donors.

    Only the charge-calculation copy changes. Original atoms, coordinates, bonds
    and formal charges stay intact. Covalent metal bonds retain Meeko's caps;
    their cap charges return to the removed metal, as in Meeko's own procedure.
    """
    removed = {
        atom.GetIdx(): atom.GetFormalCharge()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in REMOVED_METALS
    }
    caps = {}
    for index in removed:
        atom = molecule.GetAtomWithIdx(index)
        caps[index] = set()
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in removed:
                continue
            bond = molecule.GetBondBetweenAtoms(index, neighbor.GetIdx())
            if bond.GetValenceContrib(neighbor) > 0:
                caps[index].add(neighbor.GetIdx())
    capped_neighbors = set().union(*caps.values()) if caps else set()
    if sum(map(len, caps.values())) != len(capped_neighbors):
        raise ValueError("Multiple metal caps on one donor are not supported by Meeko")
    editable = Chem.RWMol(molecule)
    for index in sorted(removed, reverse=True):
        editable.RemoveAtom(index)
    auxiliary = editable.GetMol()
    indices = {
        index: index - sum(removed_index < index for removed_index in removed)
        for index in range(molecule.GetNumAtoms())
        if index not in removed
    }
    for index in capped_neighbors:
        atom = auxiliary.GetAtomWithIdx(indices[index])
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
    auxiliary.UpdatePropertyCache()
    Chem.SanitizeMol(auxiliary)
    auxiliary = Chem.AddHs(auxiliary)
    for atom in auxiliary.GetAtoms():
        if atom.GetAtomicNum() == 34:
            atom.SetAtomicNum(16)
    rdPartialCharges.ComputeGasteigerCharges(auxiliary)
    charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in auxiliary.GetAtoms()]
    if not np.isfinite(charges).all():
        raise ValueError("Nonfinite Gasteiger charges after auxiliary metal removal")
    first_added = molecule.GetNumAtoms() - len(removed)
    cap_charges = {}
    for atom in list(auxiliary.GetAtoms())[first_added:]:
        neighbors = atom.GetNeighbors()
        if atom.GetAtomicNum() != 1 or len(neighbors) != 1:
            raise ValueError("Unexpected auxiliary cap atom")
        index = neighbors[0].GetIdx()
        if index in cap_charges:
            raise ValueError("Multiple added hydrogen caps on one donor")
        cap_charges[index] = charges[atom.GetIdx()]
    result = []
    for index in range(molecule.GetNumAtoms()):
        if index in removed:
            result.append(
                removed[index] + sum(cap_charges[indices[n]] for n in caps[index])
            )
        else:
            result.append(charges[indices[index]])
    return result


@contextmanager
def metal_cap_fallback():
    """Keep every successful Meeko preparation on Meeko's unmodified code path."""
    from meeko.molsetup import RDKitMoleculeSetup
    from meeko.utils import rdkitutils

    original = RDKitMoleculeSetup.init_atom

    def init_atom(setup, *args, **kwargs):
        try:
            return original(setup, *args, **kwargs)
        except Chem.AtomValenceException:
            # The fallback reproduces Meeko 0.6.1's (assign_charges, coords) call,
            # the version pinned in docking_env.yml. Later Meeko versions pass other
            # arguments; they keep their own behaviour and error.
            if len(args) != 2 or kwargs:
                raise
            assign_charges, coords = args
            if not assign_charges or not has_uncapped_dative_donor(setup.mol):
                raise
            if setup.atoms:
                raise RuntimeError(
                    "Meeko modified atoms before its charge calculation failed"
                ) from None
            charges = corrected_metal_charges(setup.mol)
            for atom in setup.mol.GetAtoms():
                index = atom.GetIdx()
                setup.add_atom(
                    atom_index=index,
                    pdbinfo=rdkitutils.getPdbInfoNoNull(atom),
                    charge=charges[index],
                    coord=coords[index],
                    atomic_num=atom.GetAtomicNum(),
                    is_ignore=False,
                )

    RDKitMoleculeSetup.init_atom = init_atom
    try:
        yield
    finally:
        RDKitMoleculeSetup.init_atom = original


def preparation_command(preparer, *arguments):
    """Use the configured Meeko installation's interpreter when available."""
    resolved = shutil.which(str(preparer))
    sibling_python = Path(resolved).with_name("python") if resolved else None
    interpreter = (
        str(sibling_python)
        if sibling_python and sibling_python.is_file()
        else sys.executable
    )
    return [interpreter, "-m", "docking.prepare_ligand", *arguments]


def main():
    from meeko.cli.mk_prepare_ligand import main as prepare

    with metal_cap_fallback():
        return prepare()


if __name__ == "__main__":
    raise SystemExit(main())
