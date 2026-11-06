"""The case-study exporter finds pose files and keeps receptor provenance."""

from figures.case_studies.export_case_studies import (
    docking_filename,
    resolve_docking_file,
)


def test_export_resolves_current_vina_filename(tmp_path):
    name = docking_filename("VINA", "1.A", 1).replace("eRange=20_", "eRange=20.0_")
    path = tmp_path / name
    path.touch()
    assert resolve_docking_file(tmp_path, "VINA", "1.A", 1) == path


def test_export_preserves_validated_connectivity_for_isolated_atoms(tmp_path):
    import hashlib
    import json

    from rdkit import Chem

    from chelate.receptor import load_receptor_pdb
    from figures.case_studies.export_case_studies import copy_receptor_with_provenance

    editable = Chem.RWMol()
    for index in range(3):
        atom = Chem.Atom("C")
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f" C{index} ")
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("A")
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        editable.AddAtom(atom)
    editable.AddBond(0, 1, Chem.BondType.SINGLE)
    molecule = editable.GetMol()
    conformer = Chem.Conformer(3)
    for index in range(3):
        conformer.SetAtomPosition(index, [index * 1.4, 0, 0])
    molecule.AddConformer(conformer)
    source = tmp_path / "prepared.pdb"
    Chem.MolToPDBFile(molecule, str(source))
    assert Chem.MolFromPDBFile(str(source)).GetNumBonds() == 2
    proof = source.with_suffix(".pdb.provenance.json")
    proof.write_text(
        json.dumps(
            {
                "schema": "chelate-receptor-pdb-v1",
                "pdb_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    destination = tmp_path / "exported.pdb"
    copy_receptor_with_provenance(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert (
        destination.with_suffix(".pdb.provenance.json").read_bytes()
        == proof.read_bytes()
    )
    assert load_receptor_pdb(destination).GetNumBonds() == 1


def test_export_refuses_a_stale_conversion_fingerprint(tmp_path):
    import json

    import pytest

    from figures.case_studies.export_case_studies import copy_receptor_with_provenance

    source = tmp_path / "prepared.pdb"
    source.write_text("END\n")
    source.with_suffix(".pdb.provenance.json").write_text(
        json.dumps({"schema": "chelate-receptor-pdb-v1", "pdb_sha256": "wrong"})
    )
    destination = tmp_path / "exported.pdb"
    with pytest.raises(ValueError, match="provenance"):
        copy_receptor_with_provenance(source, destination)
    assert not destination.exists()


def test_export_selects_repeat_without_replacing_run_one(tmp_path, monkeypatch):
    import json
    import re
    from pathlib import Path

    import pandas as pd

    from figures.case_studies import export_case_studies as exporter

    cases = json.loads(exporter.DEFAULT_CASE_MANIFEST.read_text())["cases"]
    case = next(item for item in cases if item["pdb_id"] == "1Z4K")
    source = tmp_path / "systems" / case["system_id"]
    source.mkdir(parents=True)
    receptor = (
        f"{case['system_id']}__{case['apo_pdb_id'].lower()}_{case['apo_chain']}"
        f"_merged_{case['ligand_id']}.pdb"
    )
    (source / receptor).write_text("END\n")
    (source / f"{case['ligand_id']}.sdf").write_text("crystal\n")
    for repeat in (1, 2, 3):
        (source / exporter.docking_filename("GNINA", case["ligand_id"], repeat)).touch()
    metals = tmp_path / "metals.json"
    metals.write_text(json.dumps({case["system_id"]: {"metals": [["Mg", [0, 0, 0]]]}}))
    monkeypatch.setattr(exporter, "DEFAULT_METALS", metals)
    calls = []

    class Result:
        def __init__(self, repeat):
            self.repeat = repeat
            self.features = pd.DataFrame(
                [
                    {
                        "pose": repeat,
                        "reference_mbg_rmsd": repeat / 10,
                        "reference_rmsd": float(repeat),
                        "chelate_prediction": 0.1,
                    }
                ]
            )

        def baseline_top1(self):
            return self.features.iloc[0]

        def top1(self, cutoff):
            return self.features.iloc[0]

        def write_sdf(self, path, row):
            Path(path).write_text(str(self.repeat))

    def summarize(receptor, crystal, poses, method, cutoff, metal_coordinates):
        repeat = int(re.search(r"id=(\d+)_", poses.name).group(1))
        calls.append(repeat)
        result = Result(repeat)
        metric = exporter.metric_record(result.features.iloc[0])
        return result, {
            "number_of_poses": 1,
            "number_passing_cutoff": 1,
            "pred_cutoff": cutoff,
            "top1_original": metric,
            "best_available_pose": metric,
            "chelate_top1": metric,
        }

    monkeypatch.setattr(exporter, "summarize_run", summarize)
    output = tmp_path / "exports"
    first = exporter.export_case(case, source.parent, output, 0.8)
    first_pose = output / "01_1z4k_predCutoff=0p8" / "top1_original.sdf"
    assert first_pose.read_text() == "1"
    calls.clear()
    second = exporter.export_case(case, source.parent, output, 0.8, run_id=2)
    assert first["docking_run_id"] == 1
    assert second["docking_run_id"] == 2
    assert second["files"]["top1_original.sdf"]["mbg_rmsd_A"] == 0.2
    assert [item["run_id"] for item in second["all_runs"]] == [1, 2, 3]
    assert sorted(calls) == [1, 2, 3]
    assert first_pose.read_text() == "1"
    assert (
        output / "01_1z4k_predCutoff=0p8_run=2" / "top1_original.sdf"
    ).read_text() == "2"
