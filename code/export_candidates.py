"""Paquet d'entrée du notebook Kaggle de reclassement.

python -m code.export_candidates --data data --out kernel/input/candidates.jsonl
(lancé depuis pistes/kaggle-gemma4-papier/)

Une ligne par tâche : issue, or (hors tests), classement RRF (50 premiers chemins
et scores) et, pour les top_k premiers, les signatures du fichier : tout ce
que le reclasseur lit. Le notebook n'a ainsi besoin ni des dépôts ni des graphes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import attach_files, find_graph, gold_files, is_test_file, load_graph, load_tasks, read_git_snapshot
from .evaluate import base_ranking
from .rerank import signatures


def export(data_dir: str | Path, out: str | Path, top_k: int = 10, keep: int = 50) -> int:
    data_dir = Path(data_dir)
    rows = []
    for t in load_tasks(data_dir / "tasks.jsonl"):
        name = t["repo"].split("/")[-1]
        files = read_git_snapshot(data_dir / "repos" / name, t["base_commit"])
        gp = find_graph(data_dir, t["instance_id"], f"{name}_{t['base_commit']}")
        graph = load_graph(gp) if gp else None
        if graph is not None:
            attach_files(graph, [p for p, _ in files])
        base = base_ranking(t, files, graph, "rrf")[:keep]
        texts = dict(files)
        rows.append({
            "instance_id": t["instance_id"], "repo": t["repo"],
            "issue": t.get("problem_statement", ""),
            "gold": [g for g in gold_files(t.get("patch", "")) if not is_test_file(g)],
            "base": [[p, s] for p, s in base],
            "sigs": {p: signatures(texts.get(p, "")) for p, _ in base[:top_k]},
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="kernel/input/candidates.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    a = ap.parse_args(argv)
    print(export(a.data, a.out, a.top_k), "tâches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
