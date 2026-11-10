"""Bases sans LLM : BM25 fichier (hors tests), BM25 nœud agrégé par fichier,
score graphe à 1 saut, fusion de rangs RRF."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .data import is_test_file

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
    "the a an and or of to in is it be for on with as by at this that self "
    "def return import from if else not none true false".split()
)


def tokenize(text: str) -> list[str]:
    """Identifiants découpés en snake_case et camelCase, en minuscules.

    Un identifiant composé garde aussi sa forme entière (« verify_token » donne
    verify_token, verify, token). Jetons d'un caractère et mots vides retirés.
    """
    out: list[str] = []
    for ident in _IDENT_RE.findall(text or ""):
        parts = [p for chunk in ident.split("_") for p in _CAMEL_RE.findall(chunk)]
        parts = [p.lower() for p in parts]
        if len(parts) > 1:
            out.append(ident.lower())
        out.extend(parts)
    return [t for t in out if len(t) > 1 and t not in STOPWORDS]


class BM25:
    """Okapi BM25, idf = ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(self.len) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        res = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            for t, qn in q.items():
                f = tf.get(t)
                if f:
                    s += qn * self.idf[t] * f * (self.k1 + 1) / (f + norm)
            res.append(s)
        return res


def _sorted(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rank_files_bm25(query: str, files: list[tuple[str, str]], exclude_tests: bool = True) -> list[tuple[str, float]]:
    """BM25 au niveau fichier ; le document est chemin + contenu."""
    files = [(p, t) for p, t in files if not (exclude_tests and is_test_file(p))]
    if not files:
        return []
    bm = BM25([tokenize(p + "\n" + t) for p, t in files])
    return _sorted(dict(zip((p for p, _ in files), bm.scores(tokenize(query)))))


def node_file(node: dict) -> str | None:
    """Fichier d'un nœud : attribut file/path, sinon partie « .py » de l'id
    (formats « a/b.py::f », « a/b.py:f », « a/b.py »). À vérifier sur les vraies données."""
    for key in ("file", "path", "filepath"):
        if node.get(key):
            return str(node[key]).replace("\\", "/")
    nid = str(node.get("id", ""))
    m = re.match(r"(.+?\.py)(?:$|[:#])", nid)
    return m.group(1).replace("\\", "/") if m else None


def node_scores_bm25(query: str, graph: dict, exclude_tests: bool = True) -> dict[str, float]:
    """Score BM25 de chaque nœud (nom + texte), nœuds de test exclus."""
    nodes = [n for n in graph.get("nodes", []) if node_file(n) and not (exclude_tests and is_test_file(node_file(n)))]
    if not nodes:
        return {}
    bm = BM25([tokenize(f"{n.get('name', '')}\n{n.get('text', '')}") for n in nodes])
    return {str(n["id"]): s for n, s in zip(nodes, bm.scores(tokenize(query)))}


def _by_file_max(graph: dict, scores: dict[str, float]) -> list[tuple[str, float]]:
    agg: dict[str, float] = {}
    for n in graph.get("nodes", []):
        nid = str(n.get("id"))
        if nid in scores:
            f = node_file(n)
            agg[f] = max(agg.get(f, float("-inf")), scores[nid])
    return _sorted(agg)


def rank_files_bm25_nodes(query: str, graph: dict, exclude_tests: bool = True) -> list[tuple[str, float]]:
    """BM25 au niveau nœud, agrégé par fichier (max)."""
    return _by_file_max(graph, node_scores_bm25(query, graph, exclude_tests))


def rank_files_graph(query: str, graph: dict, alpha: float = 0.5, exclude_tests: bool = True) -> list[tuple[str, float]]:
    """Score graphe à 1 saut : s'(v) = s(v) + alpha * max_{u voisin de v} s(u),
    voisinage non orienté, s = BM25 nœud ; agrégé par fichier (max)."""
    s = node_scores_bm25(query, graph, exclude_tests)
    nbrs: dict[str, set[str]] = defaultdict(set)
    for e in graph.get("edges", []):
        a, b = str(e.get("source")), str(e.get("target"))
        nbrs[a].add(b)
        nbrs[b].add(a)
    prop = {v: sv + alpha * max((s.get(u, 0.0) for u in nbrs.get(v, ())), default=0.0) for v, sv in s.items()}
    return _by_file_max(graph, prop)


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion : somme de 1 / (k + rang), rang à partir de 1."""
    acc: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, (path, _s) in enumerate(ranking, start=1):
            acc[path] += 1.0 / (k + r)
    return _sorted(dict(acc))
