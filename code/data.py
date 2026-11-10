"""Chargement des tâches du concours Gemma 4 Developer Agent.

Formats (relevés le 2026-10-22, à confirmer sur les vrais fichiers) :
- tasks.jsonl : instance_id, repo, base_commit, problem_statement, hints_text,
  patch, test_patch, created_at ;
- snapshots/<instance_id>.tgz : dépôt figé au base_commit ;
- graphs/<instance_id>.json : graphe NetworkX (node-link), nodes[id, name, text].
"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path, PurePosixPath

_DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.MULTILINE)


def load_tasks(path: str | Path) -> list[dict]:
    """Lit tasks.jsonl (une tâche JSON par ligne, lignes vides ignorées)."""
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def gold_files(patch: str) -> list[str]:
    """Fichiers modifiés par le patch, dans l'ordre, sans doublon (côté b/).

    Les vrais patchs (relevé du 27/10) n'ont pas d'en-tête « diff --git » :
    on lit alors les lignes « +++ b/… » (« --- a/… » pour une suppression).
    """
    seen: list[str] = []
    found = [b for _a, b in _DIFF_RE.findall(patch or "")]
    if not found:
        a_side = None
        for line in (patch or "").splitlines():
            if line.startswith("--- "):
                a_side = line[4:].split("\t")[0].strip()
            elif line.startswith("+++ "):
                b_side = line[4:].split("\t")[0].strip()
                path = a_side if b_side == "/dev/null" else b_side
                if path and path != "/dev/null":
                    found.append(path.removeprefix("a/").removeprefix("b/"))
    for b in found:
        if b not in seen:
            seen.append(b)
    return seen


def is_test_file(path: str) -> bool:
    """Vrai pour les fichiers de test : dossiers test(s)/testing, test_*.py, *_test.py, conftest.py."""
    p = PurePosixPath(path.replace("\\", "/"))
    if any(part in {"test", "tests", "testing"} for part in p.parts[:-1]):
        return True
    name = p.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _strip_common_root(paths: list[str]) -> list[str]:
    """Retire un dossier racine unique commun (archive « repo-sha/... »)."""
    if not paths:
        return paths
    firsts = {p.split("/", 1)[0] for p in paths}
    if len(firsts) == 1 and all("/" in p for p in paths):
        return [p.split("/", 1)[1] for p in paths]
    return paths


def read_snapshot(tgz: str | Path, strip_root: bool = True) -> list[tuple[str, str]]:
    """Liste (chemin relatif, texte) des fichiers .py d'un snapshot .tgz.

    Lecture en mémoire, sans extraction sur disque (aucun chemin de l'archive
    n'est écrit). Décodage UTF-8 avec remplacement des octets invalides.
    """
    names: list[str] = []
    texts: list[str] = []
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.endswith(".py"):
                continue
            fh = tar.extractfile(m)
            if fh is None:
                continue
            names.append(m.name.replace("\\", "/").removeprefix("./"))
            texts.append(fh.read().decode("utf-8", errors="replace"))
    if strip_root:
        names = _strip_common_root(names)
    return sorted(zip(names, texts))


def load_graph(path: str | Path) -> dict:
    """Lit un graphe NetworkX node-link ; renvoie le dict brut (nodes, edges/links)."""
    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)
    if "edges" not in g and "links" in g:
        g["edges"] = g["links"]
    g.setdefault("edges", [])
    return g


def read_git_snapshot(repo_dir: str | Path, commit: str) -> list[tuple[str, str]]:
    """Liste (chemin, texte) des fichiers .py d'un clone git au commit donné.

    Remplace les snapshots .tgz (21,5 Go) : les quatre dépôts sont publics et
    tous les base_commit y sont (vérifié le 27/10). Rien n'est extrait sur disque.
    """
    import subprocess

    ls = subprocess.run(["git", "-C", str(repo_dir), "ls-tree", "-r", "-z", commit],
                        capture_output=True, check=True).stdout.decode("utf-8")
    entries = []
    for rec in ls.split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        _mode, kind, sha = meta.split()
        if kind == "blob" and path.endswith(".py"):
            entries.append((path, sha))
    if not entries:
        return []
    proc = subprocess.run(["git", "-C", str(repo_dir), "cat-file", "--batch"],
                          input="".join(f"{sha}\n" for _p, sha in entries).encode(),
                          capture_output=True, check=True)
    buf, pos, out = proc.stdout, 0, []
    for path, _sha in entries:
        nl = buf.index(b"\n", pos)
        size = int(buf[pos:nl].split()[2])
        out.append((path, buf[nl + 1:nl + 1 + size].decode("utf-8", errors="replace")))
        pos = nl + 1 + size + 1
    return sorted(out)


def module_of(path: str) -> str | None:
    """Module pointé d'un fichier : « src/a/b.py » → « a.b », « a/__init__.py » → « a »."""
    if not path.endswith(".py"):
        return None
    parts = path[:-3].split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or None


def attach_files(graph: dict, paths: list[str]) -> int:
    """Pose node["file"] d'après l'id pointé du nœud (plus long préfixe qui est un module).

    Les ids des vrais graphes sont « requests.models.Response.json » (relevé du 27/10).
    Renvoie le nombre de nœuds résolus.
    """
    mods = {m: p for p in paths if (m := module_of(p))}
    done = 0
    for n in graph.get("nodes", []):
        parts = str(n.get("id", "")).split(".")
        for i in range(len(parts), 0, -1):
            p = mods.get(".".join(parts[:i]))
            if p:
                n["file"] = p
                done += 1
                break
    return done


def find_graph(data_dir: str | Path, instance_id: str, repo_commit: str | None = None) -> Path | None:
    """Graphe d'une tâche : graphs/<instance_id>.json, sinon graphs/<repo>_<base_commit>.json."""
    for name in (instance_id, repo_commit):
        if name:
            p = Path(data_dir) / "graphs" / f"{name}.json"
            if p.exists():
                return p
    return None
