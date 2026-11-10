import io
import json
import pdb  # noqa: F401  charge le module standard `code` avant de le masquer
import sys
import tarfile
from pathlib import Path

import pytest

# Le paquet s'appelle « code », comme un module standard (importé par pdb) :
# on place la racine de la piste en tête et on écarte le module standard s'il est chargé.
PISTE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PISTE))
_m = sys.modules.get("code")
if _m is not None and not hasattr(_m, "__path__"):
    del sys.modules["code"]

AUTH = '''import time


class TokenError(Exception):
    pass


def verify_token(token):
    """Check the expiry claim of a token."""
    if token["exp"] < time.time():
        raise TokenError("expired")
    return True


def decode_token(raw):
    return parse_claims(raw)
'''
CLAIMS = '''def parse_claims(raw):
    return dict(part.split("=") for part in raw.split(";"))
'''
CLIENT = '''class HttpClient:
    def send_request(self, url, timeout=None):
        return self._transport.send(url, timeout=timeout)

    def close(self):
        pass
'''
RETRY = '''def retry_after(response):
    return int(response.headers.get("Retry-After", 0))
'''
TEST_AUTH = '''from pkg.auth import verify_token

def test_verify_token_expired():
    verify_token({"exp": 0})
'''
FILES = {
    "pkg/__init__.py": "",
    "pkg/auth.py": AUTH,
    "pkg/claims.py": CLAIMS,
    "pkg/client.py": CLIENT,
    "pkg/retry.py": RETRY,
    "tests/test_auth.py": TEST_AUTH,
}


def _patch(*paths):
    return "".join(f"diff --git a/{p} b/{p}\nindex 0..1 100644\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n" for p in paths)


TASKS = [
    {"instance_id": "pkg_1", "repo": "pkg",
     "problem_statement": "verify_token raises KeyError instead of TokenError when the token has no exp claim.\n"
                          "Traceback (most recent call last):\n  File \"pkg/auth.py\", line 9, in verify_token\n"
                          "KeyError: 'exp'\nExpected a TokenError.",
     "patch": _patch("pkg/auth.py", "tests/test_auth.py")},
    {"instance_id": "pkg_2", "repo": "pkg",
     "problem_statement": "HttpClient.send_request ignores the timeout when the transport is closed.",
     "patch": _patch("pkg/client.py")},
    {"instance_id": "pkg_3", "repo": "pkg",
     "problem_statement": "Retry-After header with a date value crashes retry_after.",
     "patch": _patch("pkg/retry.py")},
]


def make_tgz(path: Path, files: dict, root: str = "repo-abc123") -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        readme = b"not python"
        info = tarfile.TarInfo(f"{root}/README.md")
        info.size = len(readme)
        tar.addfile(info, io.BytesIO(readme))
    return path


def make_graph() -> dict:
    nodes = [
        {"id": "pkg/auth.py::verify_token", "name": "verify_token", "text": AUTH.split("def decode_token")[0]},
        {"id": "pkg/auth.py::decode_token", "name": "decode_token", "text": "def decode_token(raw):\n    return parse_claims(raw)"},
        {"id": "pkg/claims.py::parse_claims", "name": "parse_claims", "text": CLAIMS},
        {"id": "pkg/client.py::HttpClient.send_request", "name": "send_request", "text": CLIENT},
        {"id": "pkg/retry.py::retry_after", "name": "retry_after", "text": RETRY},
        {"id": "tests/test_auth.py::test_verify_token_expired", "name": "test_verify_token_expired", "text": TEST_AUTH},
    ]
    edges = [{"source": "pkg/auth.py::decode_token", "target": "pkg/claims.py::parse_claims", "type": "calls", "key": 0}]
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "graphs").mkdir()
    with open(tmp_path / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for t in TASKS:
            fh.write(json.dumps(t) + "\n")
            make_tgz(tmp_path / "snapshots" / f"{t['instance_id']}.tgz", FILES)
    # pas de graphe pour pkg_3 : repli sur BM25 seul
    for iid in ("pkg_1", "pkg_2"):
        (tmp_path / "graphs" / f"{iid}.json").write_text(json.dumps(make_graph()), encoding="utf-8")
    return tmp_path
