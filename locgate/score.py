"""locgate score: accuracy, cost and gates for any file localizer, from run records.

    python ressource/locgate/score.py records.jsonl [more.jsonl ...] [--base tasks.jsonl] [--gold gold.json]
        [--sep separes.jsonl]
        [--ref ref.jsonl | --ref RUN_ID] [--ref-gate "agreement 3 of 3"] [--condition trace] [--json]

Reports, per run: Acc@1 [95% bootstrap CI], Acc@5, MRR, seconds and tokens per task (Table 1);
the cost-accuracy frontier of the gates (needs `--sep` for agreement gates and RRF scores in `--gold`
for margin gates); net tasks corrected and seconds per corrected task (Table 2); paired tests
(bootstrap + exact McNemar) against the base and the reference; position bias (needs `input_order`).

Ranking of a run = its ranking, completed by the base files it does not contain. A gate that does not
fire keeps the base ranking at zero cost. Means are over all tasks of the gold file.

Functions come from the paper code (code/evaluate.py, code/triggers.py, code/stats.py,
code/position_bias.py), copied so that locgate needs only the standard library. Apache-2.0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import load_gold, load_records, load_tasks, with_gold  # noqa: E402

SEPARATE = ("bm25", "nodes", "graph")
MARGINS = (0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004)
TOP_SCORES = (0.035, 0.040, 0.045, 0.048, 0.049)
NEVER, ALWAYS = "never", "always"


# --- metrics (code/evaluate.py, code/stats.py) ---------------------------------
def acc_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    if not values:
        return {"mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(values) / n, "lo": means[int(alpha / 2 * n_boot)],
            "hi": means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]}


def mcnemar_exact(a: list[float], b: list[float]) -> dict:
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n01, n10)
    return {"a_only": n01, "b_only": n10, "p": min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    le = sum(m <= 0 for m in means) / n_boot
    ge = sum(m >= 0 for m in means) / n_boot
    return {"diff": sum(d) / n, "lo": means[int(0.025 * n_boot)], "hi": means[int(0.975 * n_boot) - 1],
            "p": min(1.0, 2 * min(le, ge))}


def kendall_tau(order_in: list[str], order_out: list[str]) -> float:
    pos = {p: i for i, p in enumerate(order_in)}
    seq = [pos[p] for p in order_out if p in pos]
    n = len(seq)
    if n < 2:
        return 1.0
    s = sum(1 if seq[i] < seq[j] else -1 for i in range(n) for j in range(i + 1, n))
    return s / (n * (n - 1) / 2)


# --- gates (code/triggers.py, code/rerank.py) ------------------------------------
def gates(t: dict, sep: dict | None, top_k: int = 10) -> dict[str, bool]:
    """Gate name -> fires? Agreement gates need the separate rankings, margin gates the RRF scores."""
    out = {NEVER: False, ALWAYS: True}
    if sep:
        tops = [(sep.get(k) or [[None]])[0][0] for k in SEPARATE]
        c = Counter(x for x in tops if x is not None)
        rrf_top = t["base"][0] if t["base"] else None
        out["agreement 3 of 3"] = max(c.values(), default=0) < 3
        out["agreement 2 of 3"] = max(c.values(), default=0) < 2
        out["RRF-supported 2 of 3"] = sum(x is not None and x == rrf_top for x in tops) < 2
    scores = t.get("scores")
    if scores is not None:
        s = scores[:top_k]
        gap = s[0] - s[1] if len(s) >= 2 else float("inf")
        rel = (s[0] - s[1]) / abs(s[0]) if len(s) >= 2 and s[0] else 0.0
        out["rel. gap < 0.05"] = len(s) >= 2 and rel < 0.05
        for m in MARGINS:
            out[f"abs. margin < {m:g}"] = gap < m
        for v in TOP_SCORES:
            out[f"top score < {v:g}"] = bool(s) and s[0] < v
    return out


def completed(base: list[str], ranking: list[str]) -> list[str]:
    return ranking + [p for p in base if p not in ranking]


def per_task(tasks: list[dict], run: dict[str, dict] | None, seps: dict, top_k: int = 10) -> dict[str, list[dict]]:
    """Gate -> one row per task (acc1, acc5, rr, fire, seconds, tokens), in task order."""
    out: dict[str, list[dict]] = {}
    for t in tasks:
        rec = (run or {}).get(t["instance_id"])
        for name, fire in gates(t, seps.get(t["instance_id"]) if seps else None, top_k).items():
            fire = fire and rec is not None
            ranked = completed(t["base"], rec["ranking"]) if fire else t["base"]
            sec = float(rec["seconds"]) if fire else 0.0
            tok = ((rec.get("tokens_in") or 0) + (rec.get("tokens_out") or 0)) if fire else 0
            out.setdefault(name, []).append({
                "acc1": acc_at_k(ranked, t["gold"], 1), "acc5": acc_at_k(ranked, t["gold"], 5),
                "acc10": acc_at_k(ranked, t["gold"], 10),
                "rr": reciprocal_rank(ranked, t["gold"]), "fire": fire, "seconds": sec, "tokens": tok})
    return out


def summarize(gate: str, model: str, rows: list[dict], n_boot: int = 2000) -> dict:
    n = len(rows)
    return {"gate": gate, "model": model, "n": n, "calls": sum(r["fire"] for r in rows),
            "acc1": bootstrap_ci([r["acc1"] for r in rows], n_boot),
            "acc5": sum(r["acc5"] for r in rows) / n, "acc10": sum(r["acc10"] for r in rows) / n, "mrr": sum(r["rr"] for r in rows) / n,
            "seconds_per_task": sum(r["seconds"] for r in rows) / n,
            "seconds_total": sum(r["seconds"] for r in rows),
            "tokens_per_task": sum(r["tokens"] for r in rows) / n}


def pareto(rows: list[dict]) -> list[dict]:
    pts = sorted(rows, key=lambda r: (r["seconds_per_task"], -r["acc1"]["mean"]))
    front, best = [], float("-inf")
    for r in pts:
        if r["acc1"]["mean"] > best + 1e-12:
            front.append(r)
            best = r["acc1"]["mean"]
    return front


def position_bias(tasks: list[dict], run: dict[str, dict]) -> dict | None:
    n = keep1 = 0
    taus, from_rank = [], Counter()
    for t in tasks:
        rec = run.get(t["instance_id"])
        if not rec or not rec.get("input_order"):
            continue
        inp = rec["input_order"]
        ranked = [p for p in rec["ranking"] if p in inp]
        if not ranked:
            continue
        n += 1
        r = inp.index(ranked[0]) + 1
        keep1 += r == 1
        from_rank[r] += 1
        taus.append(kendall_tau(inp, ranked))
    if not n:
        return None
    return {"n": n, "top1_kept": keep1, "top1_kept_share": keep1 / n, "kendall_tau_mean": sum(taus) / n,
            "top1_from_rank_3_plus": sum(v for r, v in from_rank.items() if r >= 3),
            "top1_input_rank": {str(r): from_rank[r] for r in sorted(from_rank)}}


def short(model: str) -> str:
    return model.replace("gemma-4-", "").replace("-it", "")


def run_label(run_id: str, model: str) -> str:
    """Nom court du modèle, suivi de ce que le run_id ajoute après modèle/matériel/condition (ex. « shuffle-s1 »)."""
    extra = run_id.split("/")[3:]
    return " ".join([short(model)] + extra)


# --- whole report --------------------------------------------------------------
def score(tasks: list[dict], runs: dict[str, dict[str, dict]], seps: dict | None = None,
          ref: str | None = None, ref_gate: str | None = None, n_boot: int = 2000) -> dict:
    labels = {rid: run_label(rid, next(iter(r.values()))["model"]) for rid, r in runs.items()}
    if len(set(labels.values())) < len(labels):  # deux runs du même nom : le run_id entier les sépare
        labels = {rid: rid for rid in runs}
    rows = {rid: per_task(tasks, run, seps or {}) for rid, run in runs.items()}
    base_rows = per_task(tasks, None, {})[NEVER]
    base = summarize(NEVER, "none", base_rows, n_boot)
    summ = {rid: {g: summarize(g, labels[rid], rs, n_boot) for g, rs in r.items() if g != NEVER}
            for rid, r in rows.items()}
    table1 = [base] + sorted((s[ALWAYS] for s in summ.values()), key=lambda r: r["acc1"]["mean"])
    frontier = pareto([base] + [r for s in summ.values() for r in s.values()])

    def a1(rs):
        return [r["acc1"] for r in rs]

    def corrected(label, rs, against, against_total=0.0):
        m = mcnemar_exact(a1(rs), a1(against))
        net = m["a_only"] - m["b_only"]
        total = sum(r["seconds"] for r in rs) - against_total
        return {"config": label, "gained": m["a_only"], "lost": m["b_only"], "net": net,
                "seconds_total": total, "seconds_per_corrected": total / net if net > 0 else None}

    def pair(label, a, b):
        return {"pair": label, **paired_bootstrap(a1(a), a1(b), 10000), "mcnemar": mcnemar_exact(a1(a), a1(b))}

    table2, pairs = [], []
    for rid in runs:
        table2.append(corrected(f"always, {labels[rid]}", rows[rid][ALWAYS], base_rows))
        pairs.append(pair(f"always {labels[rid]} vs base", rows[rid][ALWAYS], base_rows))
    if ref is not None:
        ref_rows = rows[ref][ref_gate or ALWAYS]
        if ref_gate and ref_gate != ALWAYS:
            table2.append(corrected(f"{ref_gate}, {labels[ref]}", ref_rows, base_rows))
            pairs.append(pair(f"{ref_gate} {labels[ref]} vs base", ref_rows, base_rows))
            pairs.append(pair(f"{ref_gate} {labels[ref]} vs always {labels[ref]}", ref_rows, rows[ref][ALWAYS]))
        for rid in runs:
            if rid == ref:
                continue
            pairs.append(pair(f"always {labels[rid]} vs always {labels[ref]}", rows[rid][ALWAYS], rows[ref][ALWAYS]))
            if ref_gate and ref_gate != ALWAYS:
                pairs.append(pair(f"{ref_gate} {labels[ref]} vs always {labels[rid]}", ref_rows, rows[rid][ALWAYS]))
            mg = corrected(f"always {labels[rid]} vs {'gated ' if ref_gate and ref_gate != ALWAYS else ''}"
                           f"{labels[ref]} (marginal)", rows[rid][ALWAYS], ref_rows,
                           sum(r["seconds"] for r in ref_rows))
            mg["marginal"] = True
            table2.append(mg)
    table2.sort(key=lambda r: (r.get("marginal", False), r["seconds_per_corrected"] or float("inf")))
    return {"n_tasks": len(tasks), "table1": table1, "frontier": frontier, "table2": table2, "pairs": pairs,
            "gates": {rid: list(s.values()) for rid, s in summ.items()},
            "position_bias": {labels[rid]: position_bias(tasks, run) for rid, run in runs.items()}}


# --- markdown --------------------------------------------------------------------
def _ci(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def table1_rows(res: dict) -> list[str]:
    out = []
    for r in res["table1"]:
        name = "RRF, no model" if r["gate"] == NEVER else f"RRF + {r['model']} on all tasks"
        sec = "0" if r["gate"] == NEVER else f"{r['seconds_per_task']:.2f}"
        out.append(f"| {name} | {_ci(r['acc1'])} | {r['acc5']:.3f} | {r['mrr']:.3f} | {sec} | "
                   f"{r['tokens_per_task']:,.0f} |")
    return out


def frontier_rows(res: dict) -> list[str]:
    return [f"| {r['gate']} | {r['model']} | {r['calls']} | {_ci(r['acc1'])} | {r['acc5']:.3f} | {r['mrr']:.3f} | "
            f"{r['seconds_per_task']:.2f} | {r['tokens_per_task']:,.0f} |" for r in res["frontier"]]


def table2_rows(res: dict) -> list[str]:
    return [f"| {r['config']} | {r['net']} ({r['gained']} − {r['lost']}) | {r['seconds_total']:.0f} | "
            + (f"{r['seconds_per_corrected']:.1f} |" if r["seconds_per_corrected"] else "n/a |")
            for r in res["table2"]]


def markdown(res: dict) -> str:
    L = [f"# locgate score ({res['n_tasks']} tasks)", "", "## Table 1: always reranking", "",
         "| Method | Acc@1 [95% CI] | Acc@5 | MRR | s/task | tokens/task |", "|---|---|---:|---:|---:|---:|",
         *table1_rows(res), "", "## Cost-accuracy frontier of the gates", "",
         "| Gate | Model | Calls | Acc@1 [95% CI] | Acc@5 | MRR | s/task | tokens/task |",
         "|---|---|---:|---|---:|---:|---:|---:|", *frontier_rows(res), "",
         "## Table 2: seconds per net corrected task (vs base)", "",
         "| Configuration | Net tasks corrected | Total s | s per corrected task |", "|---|---:|---:|---:|",
         *table2_rows(res), "", "## Paired tests (Acc@1; bootstrap 10,000 draws, exact McNemar)", ""]
    for p in res["pairs"]:
        m = p["mcnemar"]
        L.append(f"- {p['pair']}: {p['diff']:+.3f} [{p['lo']:+.2f}, {p['hi']:+.2f}], {m['a_only']}/{m['b_only']}, "
                 f"McNemar p = {m['p']:.2g}, bootstrap p = {p['p']:.3g}")
    L += ["", "## Position bias", ""]
    for name, pb in res["position_bias"].items():
        L.append(f"- {name}: " + ("no input_order" if pb is None else
                 f"top-1 kept {pb['top1_kept']}/{pb['n']} ({pb['top1_kept_share']:.0%}), "
                 f"mean Kendall tau {pb['kendall_tau_mean']:.2f}, top-1 from input rank >= 3: "
                 f"{pb['top1_from_rank_3_plus']}"))
    return "\n".join(L) + "\n"


def load_seps(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {s["instance_id"]: {k: s.get(k) for k in SEPARATE} for s in map(json.loads, f) if s}


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--base", default=str(here / "tasks_rebuilt.jsonl"),
                    help="tasks with base ranking and gold (tasks_rebuilt.jsonl, or a candidates file with RRF scores)")
    ap.add_argument("--gold", default=None,
                    help="override the gold: tasks jsonl or results json with tasks[].gold (e.g. results/rrf.json)")
    ap.add_argument("--sep", default=None, help="separate rankings (bm25/nodes/graph) for agreement gates")
    ap.add_argument("--ref", default=None, help="reference: a records file or a run_id")
    ap.add_argument("--ref-gate", default=None, help='gate applied to the reference, e.g. "agreement 3 of 3"')
    ap.add_argument("--condition", default=None, help="keep only records with this condition")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    runs: dict[str, dict[str, dict]] = {}
    for p in a.records:
        runs.update(load_records(p))
    ref = a.ref
    if ref and os.path.exists(ref):
        extra = load_records(ref)
        if a.condition:
            extra = {k: v for k, v in extra.items() if next(iter(v.values())).get("condition") == a.condition}
        if len(extra) != 1:
            ap.error(f"--ref file must hold one run (after --condition), found {sorted(extra)}")
        ref = next(iter(extra))
        runs.update(extra)
    if a.condition:
        runs = {k: v for k, v in runs.items() if next(iter(v.values())).get("condition") == a.condition}
    if ref and ref not in runs:
        ap.error(f"reference run {ref!r} not found")
    tasks = load_tasks(a.base)
    if a.gold:
        tasks = with_gold(tasks, load_gold(a.gold))
    res = score(tasks, runs, load_seps(a.sep), ref, a.ref_gate, a.n_boot)
    out = json.dumps(res, indent=1) + "\n" if a.json else markdown(res)
    sys.stdout.buffer.write(out.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
