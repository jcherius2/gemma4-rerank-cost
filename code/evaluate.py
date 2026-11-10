"""Évaluation : Acc@k, MRR, IC bootstrap 95 %, coût par tâche.

python -m code.evaluate --data <dir> --method rrf|rerank|bm25 --out results.json
(lancé depuis pistes/kaggle-gemma4-papier/)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from .baselines import rank_files_bm25, rank_files_bm25_nodes, rank_files_graph, rrf
from .data import (attach_files, find_graph, gold_files, is_test_file, load_graph, load_tasks,
                   read_git_snapshot, read_snapshot)
from .rerank import Backend, FakeBackend, LlamaCppBackend, rerank


def acc_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Moyenne et IC percentile (1 - alpha), rééchantillonnage des tâches, graine fixe."""
    if not values:
        return {"mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"mean": sum(values) / n, "lo": lo, "hi": hi}


def base_ranking(task: dict, files: list[tuple[str, str]], graph: dict | None, method: str) -> list[tuple[str, float]]:
    query = task.get("problem_statement", "")
    bm = rank_files_bm25(query, files)
    if method == "bm25" or graph is None:
        return bm
    return rrf([bm, rank_files_bm25_nodes(query, graph), rank_files_graph(query, graph)])


def evaluate(data_dir: str | Path, method: str = "rrf", backend: Backend | None = None,
             threshold: float = 0.1, top_k: int = 10, stack_trace: bool = True,
             limit: int | None = None, seed: int = 0) -> dict:
    data_dir = Path(data_dir)
    tasks = load_tasks(data_dir / "tasks.jsonl")[:limit]
    rows = []
    for t in tasks:
        iid = t["instance_id"]
        tgz = data_dir / "snapshots" / f"{iid}.tgz"
        name = t.get("repo", "").split("/")[-1]
        files = read_snapshot(tgz) if tgz.exists() else read_git_snapshot(data_dir / "repos" / name, t["base_commit"])
        gp = find_graph(data_dir, iid, f"{name}_{t.get('base_commit')}")
        graph = load_graph(gp) if gp else None
        if graph is not None:
            attach_files(graph, [p for p, _ in files])
        base = base_ranking(t, files, graph, method)
        cost = {"triggered": False, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
        ranked = [p for p, _ in base]
        if method == "rerank":
            ranked, cost = rerank(t.get("problem_statement", ""), base, dict(files), backend or FakeBackend(),
                                  threshold, top_k, stack_trace)
        gold = [g for g in gold_files(t.get("patch", "")) if not is_test_file(g)]
        existing = {p for p, _ in files}
        rows.append({"instance_id": iid, "repo": t.get("repo"), "gold": gold, "top5": ranked[:5],
                     "gold_exists": any(g in existing for g in gold),
                     "graph": gp is not None, "acc1": acc_at_k(ranked, gold, 1),
                     "acc5": acc_at_k(ranked, gold, 5), "rr": reciprocal_rank(ranked, gold), **cost})
    summary = {m: bootstrap_ci([r[m] for r in rows], seed=seed) for m in ("acc1", "acc5", "rr")}
    summary["mrr"] = summary.pop("rr")
    summary.update(
        n_tasks=len(rows),
        n_triggered=sum(r["triggered"] for r in rows),
        n_gold_exists=sum(r["gold_exists"] for r in rows),
        seconds_per_task=statistics.fmean(r["seconds"] for r in rows) if rows else 0.0,
        tokens_in_per_task=statistics.fmean(r["tokens_in"] for r in rows) if rows else 0.0,
        tokens_out_per_task=statistics.fmean(r["tokens_out"] for r in rows) if rows else 0.0,
    )
    config = {"method": method, "threshold": threshold, "top_k": top_k, "stack_trace": stack_trace, "seed": seed}
    return {"config": config, "summary": summary, "tasks": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", choices=["bm25", "rrf", "rerank"], default="rrf")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--backend", choices=["fake", "llamacpp"], default="fake")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-trace", action="store_true", help="ablation : trace de pile retirée de l'issue")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    backend = LlamaCppBackend(a.url) if a.backend == "llamacpp" else FakeBackend()
    res = evaluate(a.data, a.method, backend, a.threshold, a.top_k, not a.no_trace, a.limit, a.seed)
    res["config"]["backend"] = a.backend
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    s = res["summary"]
    print(f"{a.method}: n={s['n_tasks']} Acc@1={s['acc1']['mean']} Acc@5={s['acc5']['mean']} "
          f"MRR={s['mrr']['mean']} triggered={s['n_triggered']} s/task={s['seconds_per_task']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
