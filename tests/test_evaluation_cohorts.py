"""Method comparisons must use the same identities and the same denominator."""

import numpy as np
import pandas as pd
import pytest

from model.cohorts import (
    METHODS,
    REPEATS,
    SCENARIOS,
    TARGET,
    common_evaluable_pairs,
    filter_to_cohort,
    fixed_cohort_success,
    pair_set,
)


def frame():
    return pd.DataFrame(
        {
            "system_id": ["1abc__1__1.A__1.B_1.C", "2abc__1__1.A__1.B"],
            "ligand_id": ["1.NA", "1.B"],
            "pose": [1, 1],
            "docking_score": [-8.0, -7.0],
            "CNNscore": [0.8, 0.7],
            TARGET: [0.0, 9.0],
            "symm_rmsd": [0.0, 12.0],
        }
    )


def test_crossdocking_suffix_is_normalized_without_losing_na_chain():
    external = frame().rename(columns={"system_id": "alt_system_id"})
    external["alt_system_id"] += "__3def_A"
    assert pair_set(external) == pair_set(frame())
    assert len(filter_to_cohort(external, {("1abc__1__1.A__1.B_1.C", "1.NA")})) == 1


def test_common_cohort_uses_availability_and_never_success_thresholds():
    frames = {(m, s, r): frame() for m in METHODS for s in SCENARIOS for r in REPEATS}
    # The poorly sampled second system is still eligible: accuracy cannot select it out.
    assert common_evaluable_pairs(frames) == pair_set(frame())
    frames[("GNINA", "apo_docked_FH", 3)] = frame().assign(**{TARGET: [0.0, np.nan]})
    assert common_evaluable_pairs(frames) == {("1abc__1__1.A__1.B_1.C", "1.NA")}


def test_an_absent_repeat_cannot_silently_change_the_cohort():
    with pytest.raises(ValueError, match="18"):
        common_evaluable_pairs({("VINA", "redocked_FH", 1): frame()})


def test_missing_outputs_and_negative_labels_count_as_failures_at_fixed_denominator():
    pairs = pair_set(frame()) | {("missing", "1.X")}
    data = frame().assign(**{TARGET: [0.0, -1.0]})
    assert fixed_cohort_success(data, pairs, TARGET, 1.0) == {
        1: 1 / 3,
        3: 1 / 3,
        5: 1 / 3,
    }


def test_external_stats_keep_missing_outputs_in_the_declared_denominator(monkeypatch):
    from model import test_eval

    pairs = pair_set(frame()) | {("missing", "1.X")}
    external = frame().rename(
        columns={"symm_rmsd": "rmsd", "system_id": "alt_system_id"}
    )
    external["alt_system_id"] += "__3def_A"
    external[TARGET] = [0.0, -1.0]
    monkeypatch.setattr(
        test_eval,
        "_load_baseline_runs",
        lambda *_args: [external.copy() for _ in range(3)],
    )
    stats = test_eval.load_external_baseline_stats([1.0], cohort_pairs=pairs)
    for scenario in ["redocked", "crossdocked"]:
        for method in ["AUTODOCK", "GPDOCK", "METALLODOCK"]:
            assert stats[scenario][method]["coverage"] == (3.0, 0.0)
            assert stats[scenario][method]["mbg"][1.0][1] == (1 / 3, 0.0)


def test_native_evaluation_cannot_claim_an_unsupported_denominator():
    from model.cohorts import complete_native_cohort

    pairs = pair_set(frame()) | {("missing", "1.X")}
    with pytest.raises(ValueError, match="do not cover"):
        complete_native_cohort(frame(), pairs, "VINA")


def test_cohort_external_stats_require_every_repeat(monkeypatch):
    import pandas as pd
    import pytest

    from model import test_eval

    run = pd.DataFrame(
        {"system_id": ["s"], "ligand_id": ["l"], "pose": [1], "rmsd": [0.5]}
    )
    monkeypatch.setattr(test_eval, "_load_baseline_runs", lambda method, dock: [run])
    with pytest.raises(FileNotFoundError, match="found 1 of 3"):
        test_eval.load_external_baseline_stats([1.0], cohort_pairs={("s", "l")})


def test_run_sd_is_the_sample_sd_over_repeats():
    import numpy as np
    import pytest

    from model.test_eval import run_sd

    assert run_sd([0.1, 0.2, 0.3]) == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
    assert run_sd([0.4]) == 0.0
