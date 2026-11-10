"""Scores du reclassement Gemma 4 tiré du notebook Kaggle, rejoué hors ligne.

python -m code.score_rerank --cands kernel/input/candidates.jsonl --runs results/rerank_gemma-4-e4b-it.jsonl --out results/rerank_e4b_summary.json
(lancé depuis pistes/kaggle-gemma4-papier/)

Le notebook reclasse toujours ; ici la règle de déclenchement (should_rerank)
s'applique aux scores RRF pour chaque seuil : Acc@1, Acc@5, MRR (IC bootstrap)
et coût moyen par tâche (secondes, jetons), seulement sur les tâches déclenchées.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import acc_at_k, bootstrap_ci, reciprocal_rank
from .rerank import should_rerank

THRESHOLDS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def score(cands: list[dict], runs: dict[tuple[str, str], dict], cond: str = "trace",
          top_k: int = 10) -> list[dict]:
    out = []
    for th in THRESHOLDS:
        rows = []
        for t in cands:
            base = [p for p, _ in t["base"]]
            run = runs.get((t["instance_id"], cond)) or runs.get((t["instance_id"], "trace"))
            fire = run is not None and should_rerank([s for _, s in t["base"][:top_k]], th)
            ranked = run["ranked"] + base[top_k:] if fire else base
            rows.append({"acc1": acc_at_k(ranked, t["gold"], 1), "acc5": acc_at_k(ranked, t["gold"], 5),
                         "rr": reciprocal_rank(ranked, t["gold"]), "fire": fire,
                         "seconds": run["seconds"] if fire else 0.0,
                         "tokens": (run["tokens_in"] + run["tokens_out"]) if fire else 0})
        n = len(rows)
        out.append({"threshold": th, "cond": cond, "n": n, "triggered": sum(r["fire"] for r in rows),
                    "acc1": bootstrap_ci([r["acc1"] for r in rows]),
                    "acc5": bootstrap_ci([r["acc5"] for r in rows]),
                    "mrr": bootstrap_ci([r["rr"] for r in rows]),
                    "seconds_per_task": sum(r["seconds"] for r in rows) / n,
                    "tokens_per_task": sum(r["tokens"] for r in rows) / n})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cands = [json.loads(x) for x in open(a.cands, encoding="utf-8")]
    lines = [json.loads(x) for x in open(a.runs, encoding="utf-8")]
    meta = lines[0].get("meta", {})
    runs = {(r["instance_id"], r["cond"]): r for r in lines[1:]}
    res = {"meta": meta, "trace": score(cands, runs, "trace"), "notrace": score(cands, runs, "notrace")}
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in res["trace"]:
        print(f"seuil {r['threshold']:.2f} : déclenchées {r['triggered']:3d}  Acc@1 {r['acc1']['mean']:.3f}  "
              f"Acc@5 {r['acc5']['mean']:.3f}  MRR {r['mrr']['mean']:.3f}  s/tâche {r['seconds_per_task']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
