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
"""locgate rebuild : reconstruit les 129 tâches sans redistribuer de donnée du concours.

Pour chaque tâche, la sortie ne contient que :
- des pointeurs publics : dépôt, numéro de PR (tiré de l'instance_id), base_commit ;
- l'or : fichiers non-test modifiés par la PR, relus sur github.com (page .diff de la PR,
  hors API, donc hors limite de 60 requêtes/heure) ;
- le top-50 RRF (chemins du dépôt public au base_commit) ;
- l'empreinte sha256 du texte d'issue, jamais le texte.

Le texte d'issue (requête du classement) et les graphes de code restent ceux que
l'utilisateur détient localement (tasks.jsonl, graphs/) : ils sont lus, pas copiés.
Sans graphe, le classement retombe sur le BM25 fichier seul (comme code/evaluate.py).

Usage (depuis pistes/kaggle-gemma4-papier/) :
    python ressource/locgate/rebuild.py --data data --out ressource/locgate/tasks_rebuilt.jsonl
    python ressource/locgate/rebuild.py ... --check results/rrf.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Dossier qui contient le paquet `code` : la piste ici, la racine dans le dépôt public.
PISTE = next(p for p in Path(__file__).resolve().parents[1:3] if (p / "code" / "baselines.py").exists())
if str(PISTE) not in sys.path:
    sys.path.insert(0, str(PISTE))
# Le paquet local « code » masque le module standard du même nom.
sys.modules.pop("code", None)
from code.baselines import rank_files_bm25, rank_files_bm25_nodes, rank_files_graph, rrf  # noqa: E402
from code.data import (attach_files, find_graph, gold_files, is_test_file, load_graph,  # noqa: E402
                       load_tasks, read_git_snapshot)

TOP_N = 50
_NUM_RE = re.compile(r"_(\d+)$")


def pr_number(instance_id: str) -> int:
    """« fastapi_15661 » → 15661."""
    m = _NUM_RE.search(instance_id)
    if not m:
        raise ValueError(f"instance_id sans numéro de PR : {instance_id}")
    return int(m.group(1))


_BLOCK_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.MULTILINE)


API_PAGE = 30  # taille de page par défaut de l'API GitHub « pulls/<N>/files »


def pr_diff_files(diff: str, max_files: int | None = API_PAGE) -> list[str]:
    """Fichiers d'une PR au sens du jeu de tâches : blocs du .diff qui ont un hunk (« @@ »).

    Le patch du concours omet les fichiers nouveaux vides (aucun hunk : vérifié sur
    fastapi_14953/14978/15030/15800, __init__.py vides) mais garde les suppressions
    (fastapi_14609). Il ne couvre que les `max_files` premiers fichiers de la PR, dans
    l'ordre du .diff (première page de l'API « files », 30 par défaut) : rich_3930 a
    32 fichiers, les deux derniers (tools/) sont absents du concours ; aucun état de
    la PR (têtes successives, commit de fusion) ne les omet. Chemin côté b/ (côté a/
    pour une suppression, identique), ordre conservé, sans doublon.
    """
    out: list[str] = []
    heads = list(_BLOCK_RE.finditer(diff or ""))
    if max_files is not None:
        heads = heads[:max_files + 1]  # le bloc suivant ne sert qu'à borner le dernier
    for i, m in enumerate(heads):
        if max_files is not None and i >= max_files:
            break
        end = heads[i + 1].start() if i + 1 < len(heads) else len(diff)
        if "\n@@ " in diff[m.end():end] and m.group(2) not in out:
            out.append(m.group(2))
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def non_test(paths: list[str]) -> list[str]:
    """Convention du jeu de tâches (vérifiée : 278/278 fichiers d'or en .py) :
    l'or ne garde que les .py hors tests (code/data.is_test_file)."""
    return [p for p in paths if p.endswith(".py") and not is_test_file(p)]


def _opener():
    try:  # Avast et autres proxys HTTPS : magasin de certificats du système
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def fetch_pr_diff(repo: str, number: int, cache_dir: Path, retries: int = 3) -> str:
    """Diff complet de la PR (github.com/<repo>/pull/<N>.diff), mis en cache local."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{repo.replace('/', '__')}__{number}.diff"
    if cp.exists():
        return cp.read_text(encoding="utf-8")
    url = f"https://github.com/{repo}/pull/{number}.diff"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "locgate-rebuild"})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            cp.write_text(text, encoding="utf-8")
            return text
        except Exception as e:  # réseau, 429 : nouvel essai espacé
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{url} : {last}")


def fetch_pr_text(repo: str, number: int, cache_dir: Path) -> str | None:
    """Texte de la tâche relu sur GitHub : titre + « \\n\\n » + corps de la PR
    (forme vérifiée par sha256 sur requests_7505). Passe par l'API REST sans jeton
    (60 requêtes/heure) : renvoie None si le quota est épuisé ; le cache rend la
    reconstruction reprenable d'une heure à l'autre."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{repo.replace('/', '__')}__{number}.json"
    if not cp.exists():
        url = f"https://api.github.com/repos/{repo}/pulls/{number}"
        req = urllib.request.Request(url, headers={"User-Agent": "locgate-rebuild",
                                                   "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                return None
            raise
        cp.write_text(json.dumps({"title": d.get("title") or "", "body": d.get("body") or ""},
                                 ensure_ascii=False), encoding="utf-8")
    d = json.loads(cp.read_text(encoding="utf-8"))
    return d["title"] + ("\n\n" + d["body"] if d["body"] else "")  # corps vide : titre seul


def rrf_top(query: str, files: list[tuple[str, str]], graph: dict | None, n: int = TOP_N) -> list[str]:
    """Même fusion que code/evaluate.base_ranking (méthode rrf)."""
    bm = rank_files_bm25(query, files)
    if graph is None:
        ranked = bm
    else:
        ranked = rrf([bm, rank_files_bm25_nodes(query, graph), rank_files_graph(query, graph)])
    return [p for p, _ in ranked[:n]]


def rebuild_task(t: dict, data_dir: Path, cache_dir: Path, offline_gold: bool = False,
                 fetch_text: bool = False) -> dict:
    iid, repo, base = t["instance_id"], t["repo"], t["base_commit"]
    num = pr_number(iid)
    name = repo.split("/")[-1]
    rec = {"instance_id": iid, "repo": repo, "pr": num, "base_commit": base,
           "pr_url": f"https://github.com/{repo}/pull/{num}",
           "issue_sha256": sha256_text(t.get("problem_statement", "")),
           "issue_source": "local competition file (text not redistributed)"}
    if offline_gold:
        rec["gold"], rec["gold_source"] = non_test(gold_files(t.get("patch", ""))), "local patch"
    else:
        rec["gold"], rec["gold_source"] = non_test(pr_diff_files(fetch_pr_diff(repo, num, cache_dir))), "github pr .diff"
    files = read_git_snapshot(data_dir / "repos" / name, base)
    gp = find_graph(data_dir, iid, f"{name}_{base}")
    graph = load_graph(gp) if gp else None
    if graph is not None:
        attach_files(graph, [p for p, _ in files])
    rec["graph"] = graph is not None
    rec["gold_exists"] = any(g in {p for p, _ in files} for g in rec["gold"])
    query = t.get("problem_statement")
    if fetch_text or not query:
        gh = fetch_pr_text(repo, num, cache_dir)
        if gh is not None:
            rec["issue_github_sha256_match"] = sha256_text(gh) == rec["issue_sha256"] if query else None
            if not query:
                rec["issue_sha256"], rec["issue_source"] = sha256_text(gh), "github pr title+body"
                query = gh
    rec["top50"] = rrf_top(query or "", files, graph)
    return rec


def compare(records: list[dict], ref_path: Path, tasks: list[dict]) -> dict:
    """Bilan contre results/rrf.json (qui ne garde que le top-5) et l'or local."""
    ref = {r["instance_id"]: r for r in json.loads(ref_path.read_text(encoding="utf-8"))["tasks"]}
    local_gold = {t["instance_id"]: non_test(gold_files(t.get("patch", ""))) for t in tasks}
    out = {"n": len(records), "gold_same": 0, "gold_same_as_set": 0, "top5_same": 0,
           "gold_diff": [], "top5_diff": []}
    for r in records:
        iid, g = r["instance_id"], ref.get(r["instance_id"], {})
        if r["gold"] == g.get("gold"):
            out["gold_same"] += 1
        else:
            out["gold_diff"].append({"id": iid, "pr_only": sorted(set(r["gold"]) - set(g.get("gold", []))),
                                     "ref_only": sorted(set(g.get("gold", [])) - set(r["gold"]))})
        out["gold_same_as_set"] += set(r["gold"]) == set(g.get("gold", []))
        if r["top50"][:5] == g.get("top5"):
            out["top5_same"] += 1
        else:
            out["top5_diff"].append(iid)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(PISTE / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "tasks_rebuilt.jsonl"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / ".cache"))
    ap.add_argument("--check", default=None, help="results/rrf.json à comparer")
    ap.add_argument("--offline-gold", action="store_true", help="or tiré du patch local (test)")
    ap.add_argument("--fetch-text", action="store_true",
                    help="relit titre+corps de chaque PR (API, 60/h, cache) et contrôle le sha256")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    data_dir = Path(a.data)
    tasks = load_tasks(data_dir / "tasks.jsonl")[:a.limit]
    records = []
    for i, t in enumerate(tasks, 1):
        records.append(rebuild_task(t, data_dir, Path(a.cache), a.offline_gold, a.fetch_text))
        print(f"{i}/{len(tasks)} {t['instance_id']}", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.check:
        res = compare(records, Path(a.check), tasks)
        print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
