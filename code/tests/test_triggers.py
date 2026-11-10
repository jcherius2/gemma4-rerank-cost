import conftest  # noqa: F401  chemin du paquet « code »

from code import triggers as T


def _run(ranked, sec=1.0, tin=10, tout=2):
    return {"ranked": ranked, "seconds": sec, "tokens_in": tin, "tokens_out": tout}


def test_top1s_handles_missing_graph():
    sep = {"bm25": [["a.py", 3.0], ["b.py", 1.0]], "nodes": [], "graph": [["a.py", 2.0]]}
    assert T.top1s(sep) == ["a.py", None, "a.py"]
    assert T.top1s({}) == [None, None, None]


def test_fire_disagree():
    assert not T.fire_disagree(["a", "a", "a"], 3)
    assert T.fire_disagree(["a", "a", "b"], 3)
    assert not T.fire_disagree(["a", "a", "b"], 2)
    assert T.fire_disagree(["a", "b", "c"], 2)
    assert T.fire_disagree(["a", None, None], 2)
    assert T.fire_disagree([None, None, None], 1)


def test_fire_unbacked():
    assert not T.fire_unbacked("a", ["a", "a", "b"], 2)
    assert T.fire_unbacked("a", ["a", "b", "c"], 2)
    assert T.fire_unbacked(None, [None, None, None], 1)


def test_margin_and_top_score():
    assert abs(T.abs_margin([0.049, 0.047]) - 0.002) < 1e-12
    assert T.abs_margin([0.04]) == float("inf")
    assert T.fire_margin([0.049, 0.0489], 0.0005)
    assert not T.fire_margin([0.049, 0.040], 0.0005)
    assert T.fire_top_score([0.03, 0.02], 0.035)
    assert not T.fire_top_score([0.049], 0.045)
    assert not T.fire_top_score([], 0.045)


def test_cascade_escalates_only_on_contradiction():
    base = ["a", "b", "c"]
    small_ok, small_no, big = _run(["a", "c", "b"]), _run(["b", "a", "c"]), _run(["c", "a", "b"])
    assert T.cascade(base, small_ok, big, 2) == (["a", "c", "b"], False)
    assert T.cascade(base, small_no, big, 2) == (["c", "a", "b"], True)
    assert T.cascade(base, None, big, 2) == (base, False)
    assert T.cascade(base, small_no, None, 2) == (["b", "a", "c"], False)


def test_pareto_front():
    def r(name, s, a):
        return {"rule": name, "seconds_per_task": s, "acc1": {"mean": a}}
    rows = [r("zero", 0.0, 0.3), r("cher", 5.0, 0.5), r("domine", 6.0, 0.4), r("top", 7.0, 0.6),
            r("egal", 0.0, 0.3)]
    assert [x["rule"] for x in T.pareto(rows)] == ["zero", "cher", "top"]


def test_score_all_costs_and_cascade():
    cands = [{"instance_id": "t1", "gold": ["g.py"], "base": [["x.py", 0.049], ["g.py", 0.0489]]},
             {"instance_id": "t2", "gold": ["x.py"], "base": [["x.py", 0.049], ["g.py", 0.030]]}]
    seps = {"t1": {"bm25": [["x.py", 1]], "nodes": [["g.py", 1]], "graph": [["y.py", 1]]},
            "t2": {"bm25": [["x.py", 1]], "nodes": [["x.py", 1]], "graph": [["x.py", 1]]}}
    runs = {"e2b": {"t1": _run(["g.py", "x.py"], 2.0), "t2": _run(["x.py", "g.py"], 2.0)},
            "e4b": {"t1": _run(["g.py", "x.py"], 4.0), "t2": _run(["g.py", "x.py"], 4.0)}}
    rows = {(r["rule"], r["model"]): r for r in T.score_all(cands, seps, runs)}
    assert rows[("jamais", "e2b")]["acc1"]["mean"] == 0.5
    acc = rows[("accord_3sur3", "e4b")]
    assert acc["triggered"] == 1 and acc["acc1"]["mean"] == 1.0 and acc["seconds_per_task"] == 2.0
    assert rows[("toujours", "e4b")]["acc1"]["mean"] == 0.5
    c = rows[("cascade", "e2b->e4b")]
    assert c["triggered"] == 1 and c["seconds_per_task"] == (2 + 4 + 2) / 2 and c["acc1"]["mean"] == 1.0
    cg = rows[("cascade_si_accord_3sur3", "e2b->e4b")]
    assert cg["seconds_per_task"] == 3.0 and cg["tokens_per_task"] == 12.0
