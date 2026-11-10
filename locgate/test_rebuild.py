# Copyright 2026 Cherius
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests de rebuild.py sur données synthétiques (aucun réseau, aucune donnée du concours).

python ressource/locgate/test_rebuild.py   (stdlib seule ; pytest le lit aussi)
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import sys
import tempfile

_SPEC = importlib.util.spec_from_file_location("locgate_rebuild", Path(__file__).with_name("rebuild.py"))
rb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rb)

DIFF = """diff --git a/pkg/core.py b/pkg/core.py
index 1..2 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1 @@
-a
+b
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-a
+b
diff --git a/docs/index.md b/docs/index.md
--- a/docs/index.md
+++ b/docs/index.md
@@ -1 +1 @@
-a
+b
diff --git a/pkg/empty/__init__.py b/pkg/empty/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/pkg/old.py b/pkg/old.py
deleted file mode 100644
--- a/pkg/old.py
+++ /dev/null
@@ -1 +0,0 @@
-a
diff --git a/pkg/util.py b/pkg/util.py
--- a/pkg/util.py
+++ b/pkg/util.py
@@ -1 +1 @@
-a
+b
"""


def test_pr_number():
    assert rb.pr_number("fastapi_15661") == 15661
    assert rb.pr_number("rich_3001") == 3001
    try:
        rb.pr_number("sans-numero")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError attendue")


def test_gold_from_pr_diff_keeps_non_test_python_in_order():
    assert rb.non_test(rb.pr_diff_files(DIFF)) == ["pkg/core.py", "pkg/old.py", "pkg/util.py"]


def test_gold_keeps_only_first_api_page_of_files():
    # rich_3930 : 32 fichiers, le concours ne garde que les 30 premiers.
    diff = "".join(f"diff --git a/m{i:02d}.py b/m{i:02d}.py\n@@ -1 +1 @@\n-a\n+b\n" for i in range(32))
    assert rb.pr_diff_files(diff) == [f"m{i:02d}.py" for i in range(30)]
    assert len(rb.pr_diff_files(diff, max_files=None)) == 32


def test_sha256_is_control_not_text():
    assert rb.sha256_text("x") == hashlib.sha256(b"x").hexdigest()
    assert rb.sha256_text(None) == hashlib.sha256(b"").hexdigest()


def test_rrf_top_without_graph_is_bm25_file_ranking():
    files = [("pkg/auth.py", "def verify_token(token): pass"),
             ("pkg/db.py", "def connect(url): pass"),
             ("tests/test_auth.py", "def test_verify_token(): verify_token('x')")]
    top = rb.rrf_top("verify_token fails on expired token", files, None)
    assert top[0] == "pkg/auth.py"
    assert "tests/test_auth.py" not in top
    assert len(top) <= rb.TOP_N


def test_fetch_pr_diff_uses_cache(tmp_path):
    (tmp_path / "o__r__7.diff").write_text(DIFF, encoding="utf-8")
    assert rb.fetch_pr_diff("o/r", 7, tmp_path) == DIFF


def test_compare_counts(tmp_path):
    ref = {"tasks": [{"instance_id": "r_1", "gold": ["a.py"], "top5": ["a.py", "b.py"]},
                     {"instance_id": "r_2", "gold": ["c.py"], "top5": ["c.py"]}]}
    p = tmp_path / "rrf.json"
    p.write_text(json.dumps(ref), encoding="utf-8")
    recs = [{"instance_id": "r_1", "gold": ["a.py"], "top50": ["a.py", "b.py"]},
            {"instance_id": "r_2", "gold": ["d.py"], "top50": ["x.py"]}]
    res = rb.compare(recs, p, [])
    assert (res["gold_same"], res["top5_same"]) == (1, 1)
    assert res["gold_diff"][0] == {"id": "r_2", "pr_only": ["d.py"], "ref_only": ["c.py"]}


def test_rebuilt_file_has_no_issue_text():
    out = Path(__file__).with_name("tasks_rebuilt.jsonl")
    if not out.exists():
        return  # rien à contrôler avant la première reconstruction
    for line in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert set(r) >= {"instance_id", "repo", "pr", "base_commit", "gold", "top50", "issue_sha256"}
        assert not {"problem_statement", "patch", "test_patch", "hints_text"} & set(r)
        assert len(r["top50"]) <= 50


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            args = [Path(tempfile.mkdtemp())] if "tmp_path" in inspect.signature(fn).parameters else []
            try:
                fn(*args)
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
