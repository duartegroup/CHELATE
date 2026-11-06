"""The worked cross-docking example must evaluate against its holo binding group."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from chelate.predict import PoseSet
from chelate.receptor import extract_metals, load_receptor_pdb

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "model/example/4hev__1__1.A__1.C_1.D"


def test_example_evaluation_counts_the_donor_missed_by_the_moved_apo_metal(monkeypatch):
    crystal = EXAMPLE / "1.D_crystal.sdf"
    reference_data = json.loads((EXAMPLE / "crystal_reference.json").read_text())
    assert (
        reference_data["crystal_sha256"]
        == hashlib.sha256(crystal.read_bytes()).hexdigest()
    )
    reference = Chem.RemoveHs(Chem.SDMolSupplier(str(crystal), removeHs=False)[0])
    pose = Chem.Mol(reference)
    xyz = reference.GetConformer().GetPositions()
    pose.GetConformer().SetAtomPosition(2, xyz[2] + np.array([3.0, 0.0, 0.0]))
    apo_metals = extract_metals(
        load_receptor_pdb(EXAMPLE / "Apo_receptor_prepared.pdb")
    )
    result = PoseSet(
        [pose],
        pd.DataFrame({"pose": [1]}),
        "VINA",
        [m for m, _ in apo_metals],
        [p for _, p in apo_metals],
    )
    indices = list(range(reference.GetNumAtoms()))
    monkeypatch.setattr(
        result,
        "_symmetry_corrected_matches",
        lambda _: (reference, [(0.0, (indices, indices))]),
    )
    result.add_reference_mbg_rmsd(crystal)
    assert result.features.reference_mbg_rmsd.iloc[0] == pytest.approx(0.0)
    notebook = json.loads((ROOT / "model/example/chelate.ipynb").read_text())
    cell = next(
        c
        for c in notebook["cells"]
        if "reference-mbg-benchmark" in c.get("metadata", {}).get("tags", [])
    )
    namespace = {
        "example_dir": str(EXAMPLE),
        "crystal_sdf": str(crystal),
        "result": result,
    }
    exec(compile("".join(cell["source"]), "example_reference_cell", "exec"), namespace)
    assert result.features.reference_mbg_rmsd.iloc[0] == pytest.approx(np.sqrt(3.0))
