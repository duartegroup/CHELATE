"""Tests for the installable ``chelate`` package, run on the bundled examples."""

import json
from pathlib import Path

import numpy as np
import pytest

from chelate import (
    DEFAULT_CUTOFFS,
    extract_metal_symbols_from_pdb,
    get_feature_list,
    load_model,
    predict_poses,
    read_poses,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "model" / "example" / "8j3c__1__2.A__2.B_2.C"
VINA_EXAMPLE = ROOT / "docking" / "example_output" / "2pjc__3__1.C__1.H_1.I"
VINA_STEM = "VINA_redocked_FH_id=1_exhaust=32_numModes=100_eRange=20_lig=1.I"

POSE_FILES = {
    "VINA": "VINA_crossdocked.sdf",
    "SMINA": "SMINA_crossdocked.pdbqt",
    "GNINA": "GNINA_crossdocked.sdf.gz",
}


@pytest.mark.parametrize("method", ["VINA", "SMINA", "GNINA"])
def test_schema_matches_bundled_model(method):
    features = get_feature_list(method)
    assert len(features) == 210
    assert list(load_model(method).feature_names_) == features


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        get_feature_list("AUTODOCK")


def _pdb_line(record, name, res, element):
    line = f"{record:<6}    1 {name:<4} {res:>3} A   1       0.000   0.000   0.000  1.00  0.00"
    return f"{line}          {element:>2}\n"


def test_metal_symbols_ignore_atom_names_of_non_metals(tmp_path):
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text(
        _pdb_line("HETATM", "CA", "MSE", "C")  # selenomethionine C-alpha
        + _pdb_line("HETATM", "NA", "HEM", "N")  # haem nitrogen
        + _pdb_line("HETATM", "SG", "CSO", "S")  # modified cysteine
        + _pdb_line("HETATM", "ZN", "ZN", "ZN")
        # Missing element field: trusted only for single-atom ion records.
        + _pdb_line("HETATM", "MG", "MG", "").rstrip()
        + "\n"
        + _pdb_line("HETATM", "CD", "LYS", "").rstrip()
        + "\n"
    )
    assert extract_metal_symbols_from_pdb(str(pdb)) == {"ZN", "MG"}


def test_vina_pdbqt_matches_exported_sdf():
    pdbqt_mols, pdbqt_scores = read_poses(
        str(VINA_EXAMPLE / f"{VINA_STEM}.pdbqt"), "VINA"
    )
    sdf_mols, sdf_scores = read_poses(str(VINA_EXAMPLE / f"{VINA_STEM}.sdf"), "VINA")
    assert len(pdbqt_mols) == len(sdf_mols) == 100
    np.testing.assert_allclose(pdbqt_scores, sdf_scores, atol=1e-3)
    for a, b in zip(pdbqt_mols, sdf_mols, strict=True):
        assert a.GetNumAtoms() == b.GetNumAtoms()
        np.testing.assert_allclose(
            a.GetConformer().GetPositions(), b.GetConformer().GetPositions(), atol=1e-3
        )


@pytest.fixture(scope="module", params=["VINA", "SMINA", "GNINA"])
def pose_set(request):
    method = request.param
    return predict_poses(
        receptor=str(EXAMPLE / "Apo_receptor_prepared.pdb"),
        poses=str(EXAMPLE / POSE_FILES[method]),
        method=method,
    )


def test_predict_poses_end_to_end(pose_set):
    assert pose_set.metals == ["Co"]
    preds = pose_set.features["chelate_prediction"]
    assert len(preds) == len(pose_set.poses) > 0
    assert np.isfinite(preds).all()

    kept = pose_set.reranked()
    assert pose_set.default_cutoff == DEFAULT_CUTOFFS[pose_set.method]
    assert (kept["chelate_prediction"] <= pose_set.default_cutoff).all()
    assert kept.index.equals(pose_set.reranked(cutoff=pose_set.default_cutoff).index)
    top1 = pose_set.top1()
    if top1 is None:
        assert kept.empty
    else:
        assert top1.name == kept.index[0]
        assert pose_set.mol_for(top1) is pose_set.poses[int(top1["pose"]) - 1]

    unfiltered = pose_set.top1(cutoff=None)
    assert unfiltered.name == pose_set.baseline_top1().name


def test_no_fallback_when_no_pose_passes(pose_set):
    # When CHELATE trusts no pose, it must say so rather than fall back to the
    # docking program's ranking.
    below_all = float(pose_set.features["chelate_prediction"].min()) - 0.01
    assert pose_set.reranked(cutoff=below_all).empty
    assert pose_set.top1(cutoff=below_all) is None


def test_reference_rmsds(pose_set):
    crystal = str(EXAMPLE / "2.C_crystal.sdf")
    pose_set.add_reference_rmsd(crystal)
    pose_set.add_reference_mbg_rmsd(crystal)
    whole = pose_set.features["reference_rmsd"]
    mbg = pose_set.features["reference_mbg_rmsd"]
    assert np.isfinite(whole).all()
    assert np.isfinite(mbg).all()
    assert (mbg >= 0).all()


def test_invalid_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        load_model("AUTODOCK")
    with pytest.raises(FileNotFoundError):
        load_model(path=tmp_path / "missing.cbm")
    with pytest.raises(ValueError):
        read_poses(str(EXAMPLE / POSE_FILES["VINA"]), "AUTODOCK")

    metal_free = tmp_path / "metal_free.pdb"
    metal_free.write_text(
        "".join(
            line
            for line in (EXAMPLE / "Apo_receptor_prepared.pdb")
            .read_text()
            .splitlines(keepends=True)
            if line[76:78].strip().upper() != "CO"
        )
    )
    with pytest.raises(ValueError, match="No metal atoms"):
        predict_poses(str(metal_free), str(EXAMPLE / POSE_FILES["VINA"]), "VINA")


def test_features_without_metals():
    from rdkit import Chem

    from chelate import compute_contact_features

    receptor = Chem.MolFromPDBFile(str(EXAMPLE / "Apo_receptor_prepared.pdb"))
    poses, _ = read_poses(str(EXAMPLE / POSE_FILES["VINA"]), "VINA")
    feats = compute_contact_features(receptor, poses[0], [], pose_idx=1, n_poses=1)
    assert len(feats) == 198  # every feature except the score and metal one-hot
    assert feats["cn_lig_3.5"] == 0
    with pytest.raises(ValueError):
        compute_contact_features(receptor, poses[0], [], pose_idx=2, n_poses=1)


def test_gnina_poses_match_crystal_graph():
    # GNINA keeps a stereo-defining amidine N-H here that Chem.RemoveHs retains;
    # the graph match must tolerate it for the symmetry-corrected RMSD to succeed.
    from rdkit import Chem

    gnina = VINA_EXAMPLE / (
        "GNINA_redocked_FH_id=1_exhaust=32_numModes=100_lig=1.I.sdf.gz"
    )
    poses, _ = read_poses(str(gnina), "GNINA")
    crystal = Chem.MolFromMolFile(str(VINA_EXAMPLE / "1.I.sdf"))
    assert all(p.GetNumAtoms() == crystal.GetNumAtoms() for p in poses)


@pytest.mark.parametrize("method", ["VINA", "SMINA", "GNINA"])
def test_shipped_models_match_training_output(method):
    # chelate/params ships copies of the models train_model.py writes to model/params.
    name = f"{method}_rmsd=1.0_best_model.cbm"
    shipped = ROOT / "chelate" / "params" / name
    trained = ROOT / "model" / "params" / name
    assert shipped.read_bytes() == trained.read_bytes()


def test_default_cutoffs_match_selected_cutoffs():
    # model/select_cutoff.py writes the validation-selected cutoffs the package ships.
    selected = json.loads((ROOT / "model" / "params" / "pred_cutoffs.json").read_text())
    assert {m: selected[m] for m in ("VINA", "SMINA", "GNINA")} == DEFAULT_CUTOFFS


def test_reference_mbg_uses_holo_metals_when_apo_site_moves(tmp_path):
    """Cross-docking labels must keep the reference group defined in the holo."""
    import pandas as pd
    from rdkit import Chem

    from chelate.predict import PoseSet

    reference = Chem.MolFromSmiles("NO")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (5.0, 0.0, 0.0))
    reference.AddConformer(conformer)
    path = tmp_path / "crystal.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(reference)
    pose = Chem.Mol(reference)
    pose.GetConformer().SetAtomPosition(0, (1.0, 0.0, 0.0))
    pose.GetConformer().SetAtomPosition(1, (7.0, 0.0, 0.0))
    result = PoseSet(
        [pose], pd.DataFrame({"pose": [1]}), "VINA", ["Zn"], [np.array([0.0, 0.0, 0.0])]
    )
    result.add_reference_mbg_rmsd(path)
    assert result.features.reference_mbg_rmsd.iloc[0] == pytest.approx(1.0)
    result.add_reference_mbg_rmsd(path, reference_metal_coords=[[5.0, 0.0, 0.0]])
    assert result.features.reference_mbg_rmsd.iloc[0] == pytest.approx(2.0)
    assert result.metal_coords[0].tolist() == [0.0, 0.0, 0.0]


def test_reference_rmsd_ignores_stereo_hydrogens(tmp_path):
    """Hydrogens kept by ``RemoveHs`` for stereochemistry must not break the match."""
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from chelate.predict import PoseSet

    imine = Chem.AddHs(Chem.MolFromSmiles("CC=N"))
    AllChem.EmbedMolecule(imine, randomSeed=1)
    Chem.AssignStereochemistryFrom3D(imine)
    assert Chem.RemoveHs(imine).GetNumAtoms() > Chem.RemoveAllHs(imine).GetNumAtoms()
    path = tmp_path / "crystal.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(imine)
    # Poses from ``read_poses`` are hydrogen-free.
    pose = Chem.RemoveAllHs(imine)
    result = PoseSet([pose], pd.DataFrame({"pose": [1]}), "VINA", [], [])
    result.add_reference_rmsd(path)
    assert result.features.reference_rmsd.iloc[0] == pytest.approx(0.0, abs=1e-3)


def test_equal_docking_scores_keep_file_order():
    """Ties are broken by pose order, as in the paper's evaluation."""
    import pandas as pd

    from chelate.predict import PoseSet

    frame = pd.DataFrame({"pose": range(1, 301), "docking_score": [-7.0] * 300})
    result = PoseSet([], frame, "VINA", [], [])
    assert result._sorted_by_score(frame)["pose"].tolist() == list(range(1, 301))
