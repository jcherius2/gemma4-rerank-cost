"""Acceptance test of T2: score.py reproduces Tables 1 and 2 of writeup.md from converted records.

    py ressource/locgate/test_score.py      (from pistes/kaggle-gemma4-papier/)

Base ranking: tasks_rebuilt.jsonl (no RRF scores needed for these two tables); gold: tasks_rebuilt.jsonl (same gold files as results/rrf.json)
(the rebuilt gold differs on rich_3930, which adds two tools/ files: Acc@5 0.667 and MRR 0.496 for RRF);
agreement gate: results/separes.jsonl (ids and paths only).
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in (HERE.parent, HERE.parent.parent) if (p / "results" / "separes.jsonl").exists())
sys.path.insert(0, str(HERE))
import records  # noqa: E402
import score  # noqa: E402

TABLE1 = [
    "| RRF, no model | 0.333 [0.26, 0.42] | 0.659 | 0.493 | 0 | 0 |",
    "| RRF + e2b on all tasks | 0.550 [0.47, 0.64] | 0.752 | 0.650 | 5.02 | 3,106 |",
    "| RRF + e4b on all tasks | 0.620 [0.53, 0.71] | 0.806 | 0.704 | 7.55 | 3,107 |",
]
TABLE2 = [
    "| agreement 3 of 3, e2b | 21 (24 − 3) | 399 | 19.0 |",
    "| always, e2b | 28 (34 − 6) | 648 | 23.1 |",
    "| always, e4b | 37 (38 − 1) | 974 | 26.3 |",
    "| always e4b vs gated e2b (marginal) | 16 (18 − 2) | 575 | 35.9 |",
]


class TestTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = records.with_gold(records.load_tasks(HERE / "tasks_rebuilt.jsonl"),
                                  records.load_gold(HERE / "tasks_rebuilt.jsonl"))
        runs = {}
        with tempfile.TemporaryDirectory() as d:
            for m in ("e4b", "e2b"):
                recs = records.convert_notebook(ROOT / f"results/rerank_gemma-4-{m}-it.jsonl", tasks)
                p = Path(d) / f"{m}.jsonl"
                records.write_records(recs, p)
                runs.update({k: v for k, v in records.load_records(p).items()
                             if next(iter(v.values()))["condition"] == "trace"})
        cls.ref = next(k for k in runs if "e2b" in k)
        cls.res = score.score(tasks, runs, score.load_seps(ROOT / "results/separes.jsonl"),
                              cls.ref, "agreement 3 of 3")

    def test_records_have_no_text(self):
        tasks = records.load_tasks(HERE / "tasks_rebuilt.jsonl")
        recs = records.convert_notebook(ROOT / "results/rerank_gemma-4-e4b-it.jsonl", tasks)
        self.assertEqual(sum(r["condition"] == "trace" for r in recs), 129)
        self.assertTrue(all(set(r) <= set(records.FIELDS) for r in recs))

    def test_table1(self):
        self.assertEqual(score.table1_rows(self.res), TABLE1)

    def test_table2(self):
        self.assertEqual(score.table2_rows(self.res), TABLE2)

    def test_paired_tests(self):
        p = {x["pair"]: x for x in self.res["pairs"]}
        e4_e2 = p["always e4b vs always e2b"]
        self.assertEqual((e4_e2["mcnemar"]["a_only"], e4_e2["mcnemar"]["b_only"]), (12, 3))
        self.assertEqual(f"{e4_e2['mcnemar']['p']:.2g}", "0.035")
        self.assertEqual(f"{p['always e4b vs base']['mcnemar']['p']:.2g}", "1.5e-10")

    def test_position_bias(self):
        pb = self.res["position_bias"]
        self.assertEqual((pb["e4b"]["top1_kept"], pb["e2b"]["top1_kept"]), (55, 59))
        self.assertEqual(pb["e4b"]["top1_from_rank_3_plus"], 45)
        self.assertEqual(f"{pb['e4b']['kendall_tau_mean']:.2f}", "0.53")


if __name__ == "__main__":
    unittest.main()
