"""Benchmark cohort IDs are metadata, including nested string pairs."""

from figures.benchmark.compare_benchmark_pickles import walk


def compare(a, b):
    out = {"diff": [], "only": [], "same": 0}
    walk(a, b, "", out, 1e-9)
    return out


def test_identical_nested_cohort_ids_compare_successfully():
    pairs = [("system", "1.NA"), ("another", "2.A")]
    assert compare(pairs, pairs) == {"diff": [], "only": [], "same": 1}


def test_changed_nested_cohort_id_is_reported():
    result = compare([("system", "1.NA")], [("system", "1.A")])
    assert result["diff"] == [": nonnumeric values differ"]
    assert result["same"] == 0


def test_nested_cohort_size_change_is_reported():
    result = compare([("system", "1.NA")], [("system", "1.NA"), ("other", "2.B")])
    assert result["diff"] == [": shape (1, 2) vs (2, 2)"]
