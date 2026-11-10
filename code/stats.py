"""Validation croisée par dépôt et tests appariés des déclencheurs (hors GPU).

python -m code.stats            # chemins par défaut (DEFAULTS)
python -m code.stats --regen    # régénère aussi Tableau 1 (score_rerank) et Figure 1 (triggers)
(lancé depuis pistes/kaggle-gemma4-papier/, après `python -m code.triggers score`, dont il
complète le Markdown par une section « Validation par dépôt et tests appariés »)

Validation croisée (leave-one-repo-out) : pour chaque dépôt tenu à l'écart, on choisit sur
les autres dépôts la configuration (seuil, modèle) d'Acc@1 maximale dont le coût moyen
(s/tâche) ne dépasse pas celui de la règle de référence (accord 3/3 + e2b) sur ces mêmes
dépôts ; égalité départagée par le coût le plus bas. On l'applique au dépôt tenu à l'écart,
puis on concatène les tâches tenues à l'écart (129) pour Acc@1 et s/tâche. Familles : marge
absolue (MARGINS), score du 1er (TOP_SCORES), et « choix libre » parmi toutes les règles.

Tests appariés sur les mêmes tâches : bootstrap des différences d'Acc@1 (IC 95 %, p bilatéral)
et McNemar exact (binomial sur les paires discordantes).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from math import comb
from pathlib import Path

from .evaluate import bootstrap_ci
from .triggers import _load_runs, rows_by_rule

REF = ("accord_3sur3", "e2b")
RRF = ("jamais", "e2b")
PAIRS = ((("toujours", "e4b"), RRF), (("toujours", "e2b"), RRF), (("toujours", "e4b"), ("toujours", "e2b")),
         (REF, RRF), (REF, ("toujours", "e2b")), (REF, ("toujours", "e4b")))
FAMILIES = {"marge_abs": lambda name: name.startswith("marge_abs<"),
            "score1": lambda name: name.startswith("score1<"),
            "choix_libre": lambda name: True}
SECTION = "## Validation par dépôt et tests appariés"


# --- tests appariés -----------------------------------------------------------
def mcnemar_exact(a: list[float], b: list[float]) -> dict:
    """b01 = a juste et b faux, b10 = l'inverse ; p bilatéral exact (binomiale 1/2)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p": p}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    """Différence moyenne a - b, IC percentile 95 % et p bilatéral (rééchantillonnage des tâches)."""
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    le = sum(m <= 0 for m in means) / n_boot
    ge = sum(m >= 0 for m in means) / n_boot
    return {"diff": sum(d) / n, "lo": means[int(0.025 * n_boot)], "hi": means[int(0.975 * n_boot) - 1],
            "p": min(1.0, 2 * min(le, ge))}


# --- validation croisée par dépôt ---------------------------------------------
def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def select(per: dict, idx: list[int], keep, budget: float) -> tuple[str, str]:
    """Configuration d'Acc@1 maximale sur `idx` à coût <= budget (égalité : coût le plus bas)."""
    best, key = None, None
    for (name, model), (_, rows) in per.items():
        if not keep(name):
            continue
        sub = [rows[i] for i in idx]
        acc, sec = _mean(sub, "acc1"), _mean(sub, "seconds")
        if sec > budget + 1e-9:
            continue
        cand = (acc, -sec)
        if best is None or cand > best:
            best, key = cand, (name, model)
    return key or RRF


def loro(per: dict, repos: list[str], keep, ref: tuple[str, str] = REF) -> dict:
    """Leave-one-repo-out : seuil choisi hors dépôt, évalué sur le dépôt tenu à l'écart."""
    held: list[dict | None] = [None] * len(repos)
    folds = []
    for repo in sorted(set(repos)):
        test = [i for i, r in enumerate(repos) if r == repo]
        train = [i for i, r in enumerate(repos) if r != repo]
        budget = _mean([per[ref][1][i] for i in train], "seconds")
        name, model = select(per, train, keep, budget)
        for i in test:
            held[i] = per[(name, model)][1][i]
        sub = [held[i] for i in test]
        folds.append({"repo": repo, "n": len(test), "rule": name, "model": model,
                      "budget_s": budget, "acc1": _mean(sub, "acc1"), "seconds": _mean(sub, "seconds")})
    rows = [r for r in held if r is not None]
    return {"acc1": bootstrap_ci([r["acc1"] for r in rows]), "seconds_per_task": _mean(rows, "seconds"),
            "acc1_values": [r["acc1"] for r in rows], "folds": folds}


def run(cands: list[dict], seps: dict, runs: dict) -> dict:
    per = rows_by_rule(cands, seps, runs)
    repos = [t["repo"] for t in cands]
    acc = {k: [r["acc1"] for r in rows] for k, (_, rows) in per.items()}
    cv = {fam: loro(per, repos, keep) for fam, keep in FAMILIES.items()}
    pairs = []
    for a, b in PAIRS:
        pairs.append({"a": "+".join(a), "b": "+".join(b), "acc1_a": sum(acc[a]) / len(acc[a]),
                      "acc1_b": sum(acc[b]) / len(acc[b]), "bootstrap": paired_bootstrap(acc[a], acc[b]),
                      "mcnemar": mcnemar_exact(acc[a], acc[b])})
    for fam, res in cv.items():
        pairs.append({"a": f"cv_{fam}", "b": "+".join(RRF), "acc1_a": res["acc1"]["mean"],
                      "acc1_b": sum(acc[RRF]) / len(acc[RRF]),
                      "bootstrap": paired_bootstrap(res["acc1_values"], acc[RRF]),
                      "mcnemar": mcnemar_exact(res["acc1_values"], acc[RRF])})
    ref_rows = per[REF][1]
    return {"n": len(cands), "reference": "+".join(REF), "reference_acc1": _mean(ref_rows, "acc1"),
            "reference_seconds": _mean(ref_rows, "seconds"),
            "cv": {fam: {k: v for k, v in res.items() if k != "acc1_values"} for fam, res in cv.items()},
            "paired": pairs}


def _p(p: float, n_boot: int | None = None) -> str:
    if n_boot and p == 0:
        return f"< {1 / n_boot:g}"
    return f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"


def markdown(res: dict) -> str:
    lines = [SECTION, "",
             f"Leave-one-repo-out ({res['n']} tâches, 4 dépôts) : seuil et modèle choisis sur les trois autres "
             f"dépôts (Acc@1 maximale à coût ≤ celui de {res['reference']} sur ces dépôts), appliqués au "
             f"dépôt tenu à l'écart. Référence sans validation : Acc@1 {res['reference_acc1']:.3f}, "
             f"{res['reference_seconds']:.2f} s/tâche.", "",
             "| Famille | Acc@1 [IC] | s/tâche | Choix par pli (dépôt : règle, modèle) |", "|---|---|---:|---|"]
    for fam, r in res["cv"].items():
        ch = " ; ".join(f"{f['repo'].split('/')[-1]} : {f['rule']}, {f['model']}" for f in r["folds"])
        a = r["acc1"]
        lines.append(f"| {fam} | {a['mean']:.3f} [{a['lo']:.2f} ; {a['hi']:.2f}] | "
                     f"{r['seconds_per_task']:.2f} | {ch} |")
    lines += ["", "Tests appariés (mêmes tâches) : bootstrap des différences d'Acc@1 (10 000 tirages) et "
              "McNemar exact (a seul juste / b seul juste).", "",
              "| a | b | Acc@1 a − b [IC] | p bootstrap | a seul / b seul | p McNemar |",
              "|---|---|---|---:|---:|---:|"]
    for p in res["paired"]:
        bs, mc = p["bootstrap"], p["mcnemar"]
        lines.append(f"| {p['a']} | {p['b']} | {bs['diff']:+.3f} [{bs['lo']:+.2f} ; {bs['hi']:+.2f}] | "
                     f"{_p(bs['p'], 10000)} | {mc['a_only']} / {mc['b_only']} | {_p(mc['p'])} |")
    return "\n".join(lines) + "\n"


def append_section(md_path: Path, section: str) -> None:
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    text = re.split(r"\n" + re.escape(SECTION), text)[0].rstrip("\n") + "\n\n"
    md_path.write_text(text + section, encoding="utf-8")


DEFAULTS = {"cands": "kernel/input/candidates.jsonl", "sep": "results/separes.jsonl",
            "e2b": "results/rerank_gemma-4-e2b-it.jsonl", "e4b": "results/rerank_gemma-4-e4b-it.jsonl",
            "out": "results/stats.json", "md": "results/declencheurs.md"}


def regen(a: argparse.Namespace) -> None:
    """Tableau 1 (score_rerank), Figure 1 (triggers score), biais de position, puis cette section."""
    from . import position_bias, score_rerank, triggers
    for m in ("e4b", "e2b"):
        score_rerank.main(["--cands", a.cands, "--runs", getattr(a, m), "--out", f"results/rerank_{m}_summary.json"])
    triggers.main(["score", "--cands", a.cands, "--sep", a.sep, "--e2b", a.e2b, "--e4b", a.e4b,
                   "--out", "results/declencheurs.json", "--md", a.md])
    position_bias.main(["--cands", a.cands])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    ap.add_argument("--regen", action="store_true", help="relance d'abord score_rerank et triggers score")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if a.regen:
        regen(a)
    cands = [json.loads(x) for x in open(a.cands, encoding="utf-8")]
    seps = {r["instance_id"]: r for r in map(json.loads, open(a.sep, encoding="utf-8"))}
    runs = {"e2b": _load_runs(a.e2b), "e4b": _load_runs(a.e4b)}
    res = run(cands, seps, runs)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(res)
    append_section(Path(a.md), md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
