"""The benchmark figure builder evaluates every method on a fixed cohort."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def builder():
    path = (
        Path(__file__).resolve().parents[1]
        / "figures/benchmark/build_benchmark_results.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_cohort_test_builder", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def poses():
    return pd.DataFrame(
        {
            "system_id": ["one", "two"],
            "ligand_id": ["1.X", "1.X"],
            "pose": [1, 1],
            "rmsd": [0.0, 9.0],
            "close_rmsd_3.5_hetero": [0.0, np.nan],
        }
    )


def test_curves_count_missing_outputs_and_labels_as_failures(builder):
    pairs = {("one", "1.X"), ("two", "1.X"), ("missing", "1.X")}
    frame = builder._apply_comparison_cohort(builder._add_uid(poses()), pairs)
    runs = [frame.copy() for _ in range(3)]
    rates, _, total = builder._compute_success_rates_from_runs(runs, builder._mbg_sr)
    assert total == 3
    assert builder._n_mbg(runs) == 3
    np.testing.assert_allclose(rates, np.full(len(builder.TOP_N_POSES), 1 / 3))
    curve, _, total = builder._compute_cumulative_from_runs(
        runs, "rmsd", [0.0, 2.0, 10.0]
    )
    np.testing.assert_allclose(curve, [1 / 3, 1 / 3, 2 / 3])
    assert total == 3


def test_leakage_subset_is_matched_for_the_other_methods(builder):
    pairs = {("one", "1.X"), ("two", "1.X")}
    original = builder._apply_comparison_cohort(builder._add_uid(poses()), pairs)
    target = builder._filter_runs_excluding([original], {"two"})
    baseline = builder._filter_baselines_by_target(target, [original])
    assert target[0].attrs["evaluation_cohort_pairs"] == {("one", "1.X")}
    assert (
        baseline[0].attrs["evaluation_cohort_pairs"]
        == target[0].attrs["evaluation_cohort_pairs"]
    )


def test_empty_declared_leakage_subset_is_reported_as_unavailable(builder):
    frame = builder._apply_comparison_cohort(builder._add_uid(poses()), set())
    assert builder._compute_success_rates_from_runs([frame], builder._mbg_sr) == (
        None,
        None,
        0,
    )


def test_affinity_comparison_uses_every_method_and_repeat(builder):
    def scores(method, missing=False):
        frame = builder._add_uid(poses())
        frame[builder.AFFINITY_SCORE_COL[method]] = [-7.0, -6.0]
        if missing:
            frame = frame.iloc[:1].copy()
        return frame

    methods = {
        "VINA": [scores("VINA") for _ in range(3)],
        "GNINA": [scores("GNINA"), scores("GNINA"), scores("GNINA", True)],
    }
    prepared, pairs = builder._matched_affinity_runs(
        methods, {"one__1.X": 4.0, "two__1.X": 5.0}
    )
    assert pairs == {("one", "1.X")}
    for runs in prepared.values():
        for frame in runs:
            assert set(frame.system_id) == {"one"}
            assert frame.affinity.tolist() == [4.0]
    assert prepared["VINA"][0].docking_score.tolist() == [7.0]
    assert prepared["GNINA"][0].docking_score.tolist() == [-7.0]


def test_nonfinite_affinity_scores_cannot_change_a_single_panel_denominator(builder):
    frame = builder._add_uid(poses())
    frame[builder.AFFINITY_SCORE_COL["VINA"]] = [-7.0, np.inf]
    prepared, pairs = builder._matched_affinity_runs(
        {"VINA": [frame.copy() for _ in range(3)]}, {"one__1.X": 4.0, "two__1.X": 5.0}
    )
    assert pairs == {("one", "1.X")}
    assert all(len(run) == 1 for run in prepared["VINA"])


def test_affinity_is_matched_per_ligand_not_per_system(builder):
    # 3BY0 case: two ligands of one system, only the second has an affinity.
    frame = builder._add_uid(
        pd.DataFrame(
            {
                "system_id": ["one", "one"],
                "ligand_id": ["1.F", "1.G"],
                "pose": [1, 1],
                "rmsd": [0.0, 0.0],
                "close_rmsd_3.5_hetero": [0.0, 0.0],
            }
        )
    )
    frame[builder.AFFINITY_SCORE_COL["VINA"]] = [-7.0, -6.0]
    prepared, pairs = builder._matched_affinity_runs(
        {"VINA": [frame.copy() for _ in range(3)]},
        {"one__1.F": np.nan, "one__1.G": 4.2},
    )
    assert pairs == {("one", "1.G")}
    assert prepared["VINA"][0].affinity.tolist() == [4.2]


def test_benchmark_uses_sample_sd(builder):
    assert builder.RUN_SD_DDOF == 1


def test_free_energies_are_negated_for_the_affinity_panels(builder):
    for method in ("VINA", "SMINA", "AUTODOCK"):
        assert builder.affinity_sign(method) == -1
    assert builder.affinity_sign("GNINA") == builder.affinity_sign("METALLODOCK") == 1


def test_missing_repeat_file_fails(builder, monkeypatch, tmp_path):
    config = builder.make_config(None)
    config["test"]["AUTODOCK"] = ("results", str(tmp_path))
    config["cohort_pairs"] = {("one", "1.X")}
    monkeypatch.setattr(builder, "CONFIG", config)
    with pytest.raises(FileNotFoundError):
        builder.load_runs("AUTODOCK", "redocked")


def test_benchmark_provenance_paths_are_portable(builder, tmp_path, monkeypatch):
    repository = tmp_path / "private_mount" / "repository"
    monkeypatch.setattr(builder, "REPO", str(repository))
    features = repository / "docking/features"
    cohort = repository / "docking/cohorts/comparison_common_test.csv"
    assert builder.portable_artifact_path(str(features)) == (
        "<Zenodo record>/working_package/docking/features"
    )
    assert builder.portable_artifact_path(str(cohort), source_code=True) == (
        "<repository>/docking/cohorts/comparison_common_test.csv"
    )
    external = tmp_path / "private_mount" / "separate_inputs" / "cohort.csv"
    assert builder.portable_artifact_path(str(external)) == "<input>/cohort.csv"
