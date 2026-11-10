import pytest

from code.position_bias import kendall_tau, position_stats


def test_kendall_tau_extremes():
    a = ["a", "b", "c", "d"]
    assert kendall_tau(a, a) == 1.0
    assert kendall_tau(a, a[::-1]) == -1.0
    assert kendall_tau(a, ["b", "a", "c", "d"]) == pytest.approx(4 / 6)


def test_position_stats():
    cands = [
        {"instance_id": "t1", "base": [["a", 3], ["b", 2], ["c", 1]], "gold": ["a"]},
        {"instance_id": "t2", "base": [["a", 3], ["b", 2], ["c", 1]], "gold": ["a"]},
        {"instance_id": "t3", "base": [["a", 3], ["b", 2]], "gold": ["a"]},  # pas de sortie : ignorée
    ]
    runs = {"t1": {"ranked": ["a", "b", "c"]}, "t2": {"ranked": ["c", "b", "a"]}}
    r = position_stats(cands, runs)
    assert r["n"] == 2
    assert r["top1_is_rrf1"] == 0.5
    assert r["top1_input_rank"]["1"] == {"count": 1, "share": 0.5, "acc1": 1.0}
    assert r["top1_input_rank"]["3"]["acc1"] == 0.0
    assert r["kendall_tau_mean"] == 0.0
