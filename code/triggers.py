"""Déclencheurs du reclassement Gemma 4, évalués hors ligne sur les runs du notebook.

python -m code.triggers export --data data --out results/separes.jsonl
python -m code.triggers score --cands kernel/input/candidates.jsonl --sep results/separes.jsonl \
    --e2b results/rerank_gemma-4-e2b-it.jsonl --e4b results/rerank_gemma-4-e4b-it.jsonl \
    --out results/declencheurs.json --md results/declencheurs.md
(lancé depuis pistes/kaggle-gemma4-papier/)

Les trois classements séparés que fusionne la RRF (evaluate.base_ranking) sont
BM25 fichiers (hors tests), BM25 nœuds du graphe et propagation à 1 saut sur le
graphe ; il n'y a pas de classement par embeddings dans le code. `export` les
reconstruit par tâche (top-10) avec le code existant.

Règles : (a) accord des top-1 séparés, (b) marge absolue et score du 1er RRF,
(c) cascade e2b puis e4b si e2b contredit le top-1 RRF. Coût nul quand rien ne
se déclenche ; moyennes sur toutes les tâches.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .evaluate import acc_at_k, bootstrap_ci, reciprocal_rank
from .rerank import should_rerank

SEPARATE = ("bm25", "nodes", "graph")
MARGINS = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004)
TOP_SCORES = (0.035, 0.040, 0.045, 0.048, 0.049)


# --- règles -------------------------------------------------------------------
def top1s(sep: dict) -> list[str | None]:
    """Top-1 de chaque classement séparé (None si vide, ex. pas de graphe)."""
    return [(sep.get(k) or [[None]])[0][0] for k in SEPARATE]


def fire_disagree(tops: list[str | None], need: int) -> bool:
    """Déclenche si aucun fichier n'est top-1 d'au moins `need` classements (3 : unanimité exigée)."""
    c = Counter(t for t in tops if t is not None)
    return max(c.values(), default=0) < need


def fire_unbacked(rrf_top: str | None, tops: list[str | None], need: int = 2) -> bool:
    """Déclenche si le top-1 RRF n'est top-1 que de moins de `need` classements séparés."""
    return sum(t is not None and t == rrf_top for t in tops) < need


def abs_margin(scores: list[float]) -> float:
    return scores[0] - scores[1] if len(scores) >= 2 else float("inf")


def fire_margin(scores: list[float], m: float) -> bool:
    return abs_margin(scores) < m


def fire_top_score(scores: list[float], t: float) -> bool:
    """Score RRF du 1er sous t : il n'est pas bien classé partout (3/61 = premier des trois)."""
    return bool(scores) and scores[0] < t


# --- application --------------------------------------------------------------
def reranked(base: list[str], run: dict | None, top_k: int = 10) -> list[str]:
    return run["ranked"] + [p for p in base[top_k:] if p not in run["ranked"]] if run else base


def cost(run: dict | None) -> tuple[float, int]:
    return (run["seconds"], run["tokens_in"] + run["tokens_out"]) if run else (0.0, 0)


def cascade(base: list[str], small: dict | None, big: dict | None, top_k: int = 10) -> tuple[list[str], bool]:
    """e2b toujours ; e4b seulement si le top-1 e2b diffère du top-1 RRF. Rend (classement, escalade)."""
    if small is None:
        return base, False
    if small["ranked"] and base and small["ranked"][0] != base[0] and big is not None:
        return reranked(base, big, top_k), True
    return reranked(base, small, top_k), False


def summarize(rule: str, model: str, rows: list[dict], params: dict | None = None) -> dict:
    n = len(rows)
    return {"rule": rule, "model": model, "params": params or {}, "n": n,
            "triggered": sum(r["fire"] for r in rows),
            "acc1": bootstrap_ci([r["acc1"] for r in rows]),
            "acc5": bootstrap_ci([r["acc5"] for r in rows]),
            "mrr": bootstrap_ci([r["rr"] for r in rows]),
            "seconds_per_task": sum(r["seconds"] for r in rows) / n,
            "tokens_per_task": sum(r["tokens"] for r in rows) / n}


def _row(ranked: list[str], gold: list[str], fire: bool, sec: float, tok: int) -> dict:
    return {"acc1": acc_at_k(ranked, gold, 1), "acc5": acc_at_k(ranked, gold, 5),
            "rr": reciprocal_rank(ranked, gold), "fire": fire, "seconds": sec, "tokens": tok}


def rules(t: dict, sep: dict, top_k: int = 10) -> dict[str, tuple[dict, bool]]:
    """Toutes les règles (nom -> (paramètres, déclenche ?)) pour une tâche."""
    scores = [s for _, s in t["base"][:top_k]]
    tops = top1s(sep)
    out = {"jamais": ({}, False), "toujours": ({}, True),
           "ecart_relatif<0.05": ({"seuil": 0.05}, should_rerank(scores, 0.05)),
           "accord_3sur3": ({"need": 3}, fire_disagree(tops, 3)),
           "accord_2sur3": ({"need": 2}, fire_disagree(tops, 2)),
           "rrf_soutenu_2sur3": ({"need": 2}, fire_unbacked(t["base"][0][0] if t["base"] else None, tops, 2))}
    for m in MARGINS:
        out[f"marge_abs<{m}"] = ({"marge": m}, fire_margin(scores, m))
    for s in TOP_SCORES:
        out[f"score1<{s}"] = ({"score": s}, fire_top_score(scores, s))
    return out


def score_all(cands: list[dict], seps: dict[str, dict], runs: dict[str, dict[str, dict]],
              top_k: int = 10) -> list[dict]:
    return [summarize(name, model, rows, params)
            for (name, model), (params, rows) in rows_by_rule(cands, seps, runs, top_k).items()]


def rows_by_rule(cands: list[dict], seps: dict[str, dict], runs: dict[str, dict[str, dict]],
                 top_k: int = 10) -> dict[tuple[str, str], tuple[dict, list[dict]]]:
    """(règle, modèle) -> (paramètres, une ligne par tâche, dans l'ordre de `cands`)."""
    per: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    casc: dict[str, list[dict]] = {"cascade": [], "cascade_si_accord_3sur3": []}
    for t in cands:
        iid, gold = t["instance_id"], t["gold"]
        base = [p for p, _ in t["base"]]
        rs = rules(t, seps.get(iid, {}), top_k)
        for model, rmap in runs.items():
            run = rmap.get(iid)
            for name, (params, fire) in rs.items():
                fire = fire and run is not None
                sec, tok = cost(run if fire else None)
                ranked = reranked(base, run, top_k) if fire else base
                per.setdefault((name, model), (params, []))[1].append(_row(ranked, gold, fire, sec, tok))
        small, big = runs.get("e2b", {}).get(iid), runs.get("e4b", {}).get(iid)
        for name, gate in (("cascade", True), ("cascade_si_accord_3sur3", rs["accord_3sur3"][1])):
            if gate and small is not None:
                ranked, esc = cascade(base, small, big, top_k)
                s1, k1 = cost(small)
                s2, k2 = cost(big if esc else None)
                casc[name].append(_row(ranked, gold, esc, s1 + s2, k1 + k2))
            else:
                casc[name].append(_row(base, gold, False, 0.0, 0))
    for name, rows in casc.items():
        per[(name, "e2b->e4b")] = ({"declenchees": "e4b appelé"}, rows)
    return per


def pareto(rows: list[dict]) -> list[dict]:
    """Points non dominés (Acc@1 plus haute, secondes/tâche plus basses), triés par coût."""
    pts = sorted(rows, key=lambda r: (r["seconds_per_task"], -r["acc1"]["mean"]))
    front, best = [], float("-inf")
    for r in pts:
        if r["acc1"]["mean"] > best + 1e-12:
            front.append(r)
            best = r["acc1"]["mean"]
    return front


# --- export des classements séparés ------------------------------------------
def export(data_dir: str | Path, out: str | Path, keep: int = 10) -> int:
    from .baselines import rank_files_bm25, rank_files_bm25_nodes, rank_files_graph
    from .data import attach_files, find_graph, load_graph, load_tasks, read_git_snapshot
    data_dir = Path(data_dir)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for t in load_tasks(data_dir / "tasks.jsonl"):
            name = t["repo"].split("/")[-1]
            files = read_git_snapshot(data_dir / "repos" / name, t["base_commit"])
            gp = find_graph(data_dir, t["instance_id"], f"{name}_{t['base_commit']}")
            graph = load_graph(gp) if gp else None
            q = t.get("problem_statement", "")
            row = {"instance_id": t["instance_id"], "bm25": rank_files_bm25(q, files)[:keep]}
            if graph is not None:
                attach_files(graph, [p for p, _ in files])
                row["nodes"] = rank_files_bm25_nodes(q, graph)[:keep]
                row["graph"] = rank_files_graph(q, graph)[:keep]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_runs(path: str) -> dict[str, dict]:
    lines = [json.loads(x) for x in open(path, encoding="utf-8")][1:]
    return {r["instance_id"]: r for r in lines if r.get("cond") == "trace"}


def _ci(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.2f} ; {d['hi']:.2f}]"


def markdown(rows: list[dict], n: int) -> str:
    lines = [f"# Déclencheurs du reclassement ({n} tâches)", "",
             "Frontière de Pareto Acc@1 contre secondes par tâche (moyenne sur toutes les tâches, "
             "coût nul sans déclenchement ; T4). IC bootstrap 95 %. Seuils choisis sur ces mêmes "
             "tâches : optimiste. Toutes les lignes : `declencheurs.json`.", "",
             "| Règle | Modèle | Décl. | Acc@1 [IC] | Acc@5 | MRR | s/tâche | jetons/tâche |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in pareto(rows):
        lines.append(f"| {r['rule']} | {r['model']} | {r['triggered']} | {_ci(r['acc1'])} | "
                     f"{r['acc5']['mean']:.3f} | {r['mrr']['mean']:.3f} | {r['seconds_per_task']:.2f} | "
                     f"{r['tokens_per_task']:.0f} |")
    lines += ["", "Cascade : e2b sur toutes les tâches, e4b seulement si le top-1 e2b diffère du top-1 "
              "RRF ; « Décl. » = appels e4b. Absente de la frontière : dominée par e4b seul, ou par "
              "e4b sous marge_abs<0.002, plus rapides à Acc@1 égale ou supérieure.", "",
              "Classements séparés : BM25 fichiers, BM25 nœuds, graphe à 1 saut (le code n'a pas "
              "d'embeddings). accord_3sur3 équivaut à score1<0.049 (3/61 = premier partout).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    for k in ("cands", "sep", "e2b", "e4b", "out", "md"):
        s.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(export(a.data, a.out), "tâches")
        return 0
    cands = [json.loads(x) for x in open(a.cands, encoding="utf-8")]
    seps = {r["instance_id"]: r for r in map(json.loads, open(a.sep, encoding="utf-8"))}
    runs = {"e2b": _load_runs(a.e2b), "e4b": _load_runs(a.e4b)}
    rows = score_all(cands, seps, runs)
    Path(a.out).write_text(json.dumps({"n": len(cands), "rows": rows, "pareto": [
        (r["rule"], r["model"]) for r in pareto(rows)]}, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(a.md).write_text(markdown(rows, len(cands)), encoding="utf-8")
    for r in sorted(rows, key=lambda r: r["seconds_per_task"]):
        print(f"{r['rule']:26s} {r['model']:9s} décl {r['triggered']:3d}  Acc@1 {_ci(r['acc1'])}  "
              f"Acc@5 {r['acc5']['mean']:.3f}  s {r['seconds_per_task']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
