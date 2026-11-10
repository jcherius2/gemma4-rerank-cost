"""Test of T5: third-party formats -> run records -> score.py, on the synthetic files of examples/.

    py ressource/locgate/test_adapters.py      (from pistes/kaggle-gemma4-papier/)
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in (HERE.parent, HERE.parent.parent) if (p / "results" / "separes.jsonl").exists())
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
import records  # noqa: E402
import score  # noqa: E402

IDS = ["fastapi_15030", "fastapi_14962", "fastapi_15280", "requests_7505", "requests_7502"]
ACC1 = {"base": [0, 1, 0, 1, 1], "ranked": [1, 0, 1, 0, 1], "swebench": [1, 1, 1, 0, 1]}


class TestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = records.with_gold(records.load_tasks(HERE / "tasks_rebuilt.jsonl"),
                                      records.load_gold(HERE / "tasks_rebuilt.jsonl"))
        cls.runs = {}
        with tempfile.TemporaryDirectory() as d:
            for name, model in (("ranked.json", "demo-retriever"), ("swebench.jsonl", None)):
                recs = adapters.convert(HERE / "examples" / name, run_id=f"demo/{name}", model=model)
                p = Path(d) / (name + ".records.jsonl")
                records.write_records(recs, p)
                cls.runs.update(records.load_records(p))

    def test_patch_paths(self):
        diff = ("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
                "--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a\n+b\n--- a/new.py\n+++ b/new.py\n")
        self.assertEqual(adapters.patch_paths(diff), ["new.py", "src/m.py"])

    def test_records(self):
        sw = self.runs["demo/swebench.jsonl"]
        self.assertEqual(sorted(sw), sorted(IDS))
        self.assertEqual(sw["fastapi_14962"]["ranking"], ["fastapi/routing.py", "fastapi/_compat/v2.py"])
        self.assertEqual(sw["fastapi_15280"]["ranking"][0], "docs_src/vibe/tutorial001_py310.py")
        self.assertEqual(sw["requests_7505"]["seconds"], 1.5)
        self.assertEqual(sw["requests_7502"]["model"], "demo-agentless")
        self.assertEqual(self.runs["demo/ranked.json"]["fastapi_15030"]["model"], "demo-retriever")

    def test_picks(self):
        lines = [
            {"instance_id": "fastapi_15030", "condition": "paths", "seed": 0, "picks": ["a.py", "b.py", "a.py"],
             "seconds": 1.2, "prompt_tokens": 575},
            {"instance_id": "fastapi_15030", "condition": "paths", "seed": 1, "picks": ["b.py"], "seconds": 2.0},
            {"instance_id": "requests_7505", "condition": "outline", "seed": 0, "picks": ["c.py"]},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "picks.jsonl"
            p.write_text("\n".join(__import__("json").dumps(x) for x in lines) + "\n", encoding="utf-8")
            recs = adapters.convert(p, "picks", run_id="demo/picks", model="m",
                                    where={"condition": "paths", "seed": "0"})
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["ranking"], r["seconds"], r["tokens_in"], r["seed"], r["condition"]),
                         (["a.py", "b.py"], 1.2, 575, 0, "paths"))

    def test_score(self):
        tasks = [t for t in self.tasks if t["instance_id"] in IDS]
        tasks.sort(key=lambda t: IDS.index(t["instance_id"]))
        base = score.per_task(tasks, None, {})[score.NEVER]
        self.assertEqual([r["acc1"] for r in base], ACC1["base"])
        for rid, key in (("demo/ranked.json", "ranked"), ("demo/swebench.jsonl", "swebench")):
            rows = score.per_task(tasks, self.runs[rid], {})[score.ALWAYS]
            self.assertEqual([r["acc1"] for r in rows], ACC1[key], rid)
        res = score.score(self.tasks, self.runs, n_boot=200)
        self.assertEqual(res["n_tasks"], 129)
        self.assertEqual(len(res["table1"]), 3)
        self.assertIn("| RRF, no model |", score.markdown(res))


if __name__ == "__main__":
    unittest.main()
