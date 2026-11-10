import conftest  # noqa: F401  chemin du paquet « code »

from code import stats as S


def _rows(accs, secs):
    return [{"acc1": a, "seconds": s} for a, s in zip(accs, secs)]


def test_mcnemar_exact():
    r = S.mcnemar_exact([1, 1, 1, 0, 1], [0, 0, 0, 0, 1])
    assert (r["a_only"], r["b_only"]) == (3, 0)
    assert abs(r["p"] - 0.25) < 1e-12  # 2 * (1/2)^3
    assert S.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0
    assert S.mcnemar_exact([1, 0], [0, 1])["p"] == 1.0


def test_paired_bootstrap():
    same = S.paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], n_boot=200)
    assert same["diff"] == 0 and same["p"] == 1.0
    up = S.paired_bootstrap([1] * 20, [0] * 20, n_boot=200)
    assert up["diff"] == 1 and up["lo"] == 1 and up["p"] == 0.0


def test_select_respects_budget():
    per = {("cheap", "m"): ({}, _rows([0, 1, 0], [1, 1, 1])),
           ("good", "m"): ({}, _rows([1, 1, 1], [9, 9, 9])),
           ("mid", "m"): ({}, _rows([1, 1, 0], [2, 2, 2]))}
    assert S.select(per, [0, 1, 2], lambda n: True, 2.0) == ("mid", "m")
    assert S.select(per, [0, 1, 2], lambda n: True, 10.0) == ("good", "m")
    assert S.select(per, [0, 1, 2], lambda n: n == "good", 1.0) == S.RRF


def test_loro_holds_out_each_repo():
    # sur le dépôt A, « x » est meilleur ; sur B, « y » : le choix appris sur A s'applique à B et inversement
    per = {("ref", "m"): ({}, _rows([0, 0, 0, 0], [1, 1, 1, 1])),
           ("x", "m"): ({}, _rows([1, 1, 0, 0], [1, 1, 1, 1])),
           ("y", "m"): ({}, _rows([0, 0, 1, 1], [1, 1, 1, 1]))}
    res = S.loro(per, ["A", "A", "B", "B"], lambda n: n in ("x", "y"), ref=("ref", "m"))
    assert [(f["repo"], f["rule"]) for f in res["folds"]] == [("A", "y"), ("B", "x")]
    assert res["acc1"]["mean"] == 0.0
    assert res["seconds_per_task"] == 1.0


def test_append_section_is_idempotent(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# T\n\ncorps\n", encoding="utf-8")
    S.append_section(p, S.SECTION + "\n\nA\n")
    S.append_section(p, S.SECTION + "\n\nB\n")
    text = p.read_text(encoding="utf-8")
    assert text.count(S.SECTION) == 1 and text.endswith("B\n") and text.startswith("# T\n\ncorps")
