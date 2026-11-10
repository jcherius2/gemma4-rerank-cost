import json

import pytest
from conftest import FILES, TASKS, make_graph, make_tgz

from code import baselines as B
from code import data as D
from code import evaluate as E
from code import rerank as R


# --- data -------------------------------------------------------------------
def test_gold_files_from_patch():
    assert D.gold_files(TASKS[0]["patch"]) == ["pkg/auth.py", "tests/test_auth.py"]
    assert D.gold_files("") == []


@pytest.mark.parametrize("path,expected", [
    ("tests/test_auth.py", True), ("pkg/test_x.py", True), ("pkg/x_test.py", True),
    ("conftest.py", True), ("src/testing/util.py", True),
    ("pkg/auth.py", False), ("pkg/contest.py", False), ("pkg/latest.py", False),
])
def test_is_test_file(path, expected):
    assert D.is_test_file(path) is expected


def test_read_snapshot_strips_root_and_keeps_py(tmp_path):
    files = D.read_snapshot(make_tgz(tmp_path / "s.tgz", FILES))
    paths = [p for p, _ in files]
    assert paths == sorted(FILES)
    assert dict(files)["pkg/claims.py"].startswith("def parse_claims")


def test_load_tasks_and_graph(dataset):
    tasks = D.load_tasks(dataset / "tasks.jsonl")
    assert [t["instance_id"] for t in tasks] == ["pkg_1", "pkg_2", "pkg_3"]
    assert D.find_graph(dataset, "pkg_3") is None
    g = D.load_graph(D.find_graph(dataset, "pkg_1"))
    assert len(g["nodes"]) == 6 and len(g["edges"]) == 1


# --- baselines --------------------------------------------------------------
def test_tokenize_snake_camel():
    toks = B.tokenize("verify_token HttpClient parseJSONBody x self")
    for t in ("verify_token", "verify", "token", "httpclient", "http", "client", "parse", "json", "body"):
        assert t in toks
    assert "x" not in toks and "self" not in toks


def test_bm25_idf_and_ordering():
    bm = B.BM25([["a", "b"], ["a", "c", "c"], ["d"]])
    s = bm.scores(["c"])
    assert s[1] > 0 and s[0] == 0 and s[2] == 0
    assert bm.idf["c"] > bm.idf["a"]


def test_file_bm25_excludes_tests():
    ranked = B.rank_files_bm25(TASKS[0]["problem_statement"], list(FILES.items()))
    paths = [p for p, _ in ranked]
    assert paths[0] == "pkg/auth.py"
    assert "tests/test_auth.py" not in paths


def test_node_bm25_and_graph():
    g = make_graph()
    assert B.node_file({"id": "a/b.py:f"}) == "a/b.py"
    assert B.node_file({"id": "x", "file": "c.py"}) == "c.py"
    nodes = B.rank_files_bm25_nodes("send_request timeout", g)
    assert nodes[0][0] == "pkg/client.py"
    assert all(not D.is_test_file(p) for p, _ in nodes)
    # parse_claims reçoit, par l'arête decode_token -> parse_claims, une part du score de decode_token
    plain = dict(B.rank_files_bm25_nodes("decode_token", g))
    hop = dict(B.rank_files_graph("decode_token", g))
    assert plain.get("pkg/claims.py", 0) == 0 < hop["pkg/claims.py"]


def test_rrf():
    fused = B.rrf([[("a", 9), ("b", 5)], [("b", 3), ("c", 1)]], k=60)
    assert [p for p, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


# --- rerank -----------------------------------------------------------------
def test_signatures_and_prompt():
    sig = R.signatures(FILES["pkg/client.py"])
    assert sig == ["class HttpClient:", "    def send_request(self, url, timeout=None):", "    def close(self):"]
    p = R.build_prompt(TASKS[0]["problem_statement"], [("pkg/auth.py", FILES["pkg/auth.py"])])
    assert "[1] pkg/auth.py" in p and "def verify_token(token):" in p and "Traceback" in p
    p2 = R.build_prompt(TASKS[0]["problem_statement"], [("pkg/auth.py", "")], stack_trace=False)
    assert "Traceback" not in p2 and "KeyError: 'exp'" not in p2 and "Expected a TokenError." in p2


def test_parse_ranking_robust():
    c = ["a/b.py", "x/a/b.py", "c.py"]
    assert R.parse_ranking("1. x/a/b.py\n2. `c.py`\nc.py again", c) == ["x/a/b.py", "c.py", "a/b.py"]
    assert R.parse_ranking("I think [3] then [1]", c) == ["c.py", "a/b.py", "x/a/b.py"]
    assert R.parse_ranking("no idea", c) == c
    assert R.parse_ranking("", []) == []


def test_should_rerank():
    assert R.should_rerank([1.0, 0.95], 0.1)
    assert not R.should_rerank([1.0, 0.5], 0.1)
    assert not R.should_rerank([1.0], 1.0)
    assert not R.should_rerank([1.0, 1.0], 0.0)


def test_rerank_with_fake_backend():
    base = [("a.py", 1.0), ("b.py", 0.99), ("c.py", 0.5)]
    fb = R.FakeBackend()
    ranked, cost = R.rerank("issue", base, {}, fb, threshold=0.1, top_k=2)
    assert ranked == ["b.py", "a.py", "c.py"] and cost["triggered"] and cost["tokens_in"] > 0
    ranked, cost = R.rerank("issue", base, {}, fb, threshold=0.0)
    assert ranked == ["a.py", "b.py", "c.py"] and not cost["triggered"] and fb.calls == 1


# --- evaluate ---------------------------------------------------------------
def test_metrics():
    assert E.acc_at_k(["a", "b"], ["b"], 1) == 0 and E.acc_at_k(["a", "b"], ["b"], 5) == 1
    assert E.reciprocal_rank(["a", "b"], ["b"]) == 0.5 and E.reciprocal_rank(["a"], ["z"]) == 0


def test_bootstrap_deterministic():
    v = [0, 1, 1, 0, 1, 1, 1, 0]
    a, b = E.bootstrap_ci(v, seed=0), E.bootstrap_ci(v, seed=0)
    assert a == b and a["lo"] <= a["mean"] <= a["hi"] and a["mean"] == 0.625
    assert E.bootstrap_ci([])["mean"] is None


@pytest.mark.parametrize("method", ["bm25", "rrf", "rerank"])
def test_evaluate_end_to_end(dataset, method):
    res = E.evaluate(dataset, method, R.FakeBackend(), threshold=1.0)
    s = res["summary"]
    assert s["n_tasks"] == 3 and 0 <= s["acc1"]["mean"] <= s["acc5"]["mean"] == 1.0
    assert res["tasks"][0]["gold"] == ["pkg/auth.py"]
    if method == "rerank":
        assert s["n_triggered"] == 3
    else:
        assert s["acc1"]["mean"] == 1.0 and s["n_triggered"] == 0


def test_cli_writes_json(dataset, tmp_path):
    out = tmp_path / "r.json"
    assert E.main(["--data", str(dataset), "--method", "rrf", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["config"]["method"] == "rrf" and len(res["tasks"]) == 3
