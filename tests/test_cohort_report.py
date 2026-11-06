"""Fixed-curated rates must reconstruct integer successes before rescaling."""

import pytest

from model.cohort_report import native_record, summarise_native


def example():
    return {
        "total_systems": 10,
        "systems_count": 5,
        "baseline_MBG_sr_top1_RMSD=1": 0.2,
        "MBG_sr_all_systems_top1_RMSD=1": 0.3,
        "MBG_sr_top1_RMSD=1": 0.6,
        "baseline_sr_top1_RMSD=2": 0.4,
        "sr_top1_RMSD=2": 0.8,
        "dock_type": "Re-docked",
        "rep_id": 1,
        "rmsd_cutoff": 0.75,
    }


def test_fixed_denominator_keeps_rejected_and_unavailable_pairs_as_failures():
    result = native_record(example(), "VINA", "curated_156", 156)
    assert result["chelate_mbg_successes"] == 3
    assert result["chelate_mbg_sr"] == 3 / 156
    assert result["chelate_whole_sr"] == 4 / 156
    assert result["coverage"] == 5 / 156
    assert result["chelate_mbg_retained_sr"] == 0.6


def test_disagreeing_conditional_and_overall_counts_are_reported():
    row = example()
    row["MBG_sr_all_systems_top1_RMSD=1"] = 0.4
    with pytest.raises(ValueError, match="numerators disagree"):
        native_record(row, "VINA", "common_evaluable", 10)


def test_rounded_or_mismatched_denominators_are_not_silently_rescaled():
    row = example()
    row["baseline_sr_top1_RMSD=2"] = 0.444
    with pytest.raises(ValueError, match="integer success count"):
        native_record(row, "VINA", "curated_156", 156)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Redocked", "redocked"),
        ("Crossdocked", "crossdocked"),
        ("Re-docked", "redocked"),
        ("Cross-docked", "crossdocked"),
    ],
)
def test_producer_scenario_labels_stay_separate(label, expected):
    row = example()
    row["dock_type"] = label
    assert native_record(row, "VINA", "common_evaluable", 10)["dock_key"] == expected


def test_an_unknown_scenario_is_rejected_instead_of_assumed_crossdocked():
    row = example()
    row["dock_type"] = "unexpected scenario"
    with pytest.raises(ValueError, match="Unknown docking scenario"):
        native_record(row, "VINA", "common_evaluable", 10)


def test_summary_records_how_many_repeats_each_mean_uses():
    import pandas as pd

    rows = [
        {
            "cohort": "c",
            "method": "VINA",
            "dock_key": "redocked",
            "pred_cutoff": cut,
            "rep_id": rep,
            "chelate_mbg_sr": rate,
        }
        for cut, rep, rate in [
            (0.45, 1, 0.04),
            (0.8, 1, 0.3),
            (0.8, 2, 0.3),
            (0.8, 3, 0.36),
        ]
    ]
    summary = summarise_native(pd.DataFrame(rows)).set_index("pred_cutoff")
    assert summary.loc[0.45, "n_repeats"] == 1
    assert summary.loc[0.8, "n_repeats"] == 3
    assert summary.loc[0.8, "chelate_mbg_sr"] == pytest.approx(0.32)
    assert "rep_id" not in summary.columns
