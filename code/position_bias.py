"""Biais de position du reclassement : le modèle recopie-t-il l'ordre RRF d'entrée ?

Les candidats arrivent au modèle dans l'ordre RRF, sans mélange. Ce script mesure, sur les sorties
brutes (condition « trace »), la part des tâches où le top-1 du modèle est le rang 1 RRF, la
distribution du rang d'entrée du top-1 choisi (avec son Acc@1), et le tau de Kendall moyen entre
ordre d'entrée et ordre de sortie. Sans mélange, ces chiffres ne séparent pas biais de position et
accord réel : ils bornent seulement l'écho possible de l'ordre d'entrée.

python -m code.position_bias  (depuis la racine de la piste ; écrit results/position_bias.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {"e4b": "results/rerank_gemma-4-e4b-it.jsonl", "e2b": "results/rerank_gemma-4-e2b-it.jsonl"}


def kendall_tau(order_in: list[str], order_out: list[str]) -> float:
    """Tau de Kendall entre deux ordres des mêmes éléments (sans ex aequo), sur leur intersection."""
    pos = {p: i for i, p in enumerate(order_in)}
    seq = [pos[p] for p in order_out if p in pos]
    n = len(seq)
    if n < 2:
        return 1.0
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if seq[i] < seq[j] else -1
    return s / (n * (n - 1) / 2)


def position_stats(cands: list[dict], runs: dict[str, dict], top_k: int = 10) -> dict:
    n = 0
    keep1 = 0
    taus = []
    by_rank: dict[int, list[int]] = {}
    for t in cands:
        run = runs.get(t["instance_id"])
        if run is None:
            continue
        base = [p for p, _ in t["base"][:top_k]]
        ranked = [p for p in run["ranked"] if p in base]
        if not ranked:
            continue
        n += 1
        r = base.index(ranked[0]) + 1
        keep1 += r == 1
        by_rank.setdefault(r, []).append(int(ranked[0] in t["gold"]))
        taus.append(kendall_tau(base, ranked))
    dist = {str(r): {"count": len(v), "share": len(v) / n, "acc1": sum(v) / len(v)}
            for r, v in sorted(by_rank.items())}
    return {"n": n, "top1_is_rrf1": keep1 / n if n else 0.0, "top1_is_rrf1_count": keep1,
            "top1_input_rank": dist, "kendall_tau_mean": sum(taus) / n if n else 0.0}


def _load(path: str, cond: str = "trace") -> dict[str, dict]:
    lines = [json.loads(x) for x in open(path, encoding="utf-8")][1:]
    return {r["instance_id"]: r for r in lines if r.get("cond") == cond}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cands", default="kernel/input/candidates.jsonl")
    ap.add_argument("--out", default="results/position_bias.json")
    a = ap.parse_args(argv)
    cands = [json.loads(x) for x in open(a.cands, encoding="utf-8")]
    res = {m: position_stats(cands, _load(p)) for m, p in MODELS.items()}
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for m, r in res.items():
        print(f"{m} : n {r['n']}  top-1 = rang 1 RRF {r['top1_is_rrf1']:.3f}  tau moyen {r['kendall_tau_mean']:.3f}")
        for k, v in r["top1_input_rank"].items():
            print(f"   rang d'entrée {k:>2} : {v['count']:3d} ({v['share']:.2f})  Acc@1 {v['acc1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
