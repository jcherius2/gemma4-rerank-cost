"""Reclassement du top-k par un LLM (Gemma 4 e4b quantifié), déclenché
seulement quand l'écart entre les deux premiers scores est faible."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from abc import ABC, abstractmethod

_SIG_RE = re.compile(r"^\s*(?:@\w|(?:async\s+)?def\s|class\s)")
_TRACE_START = "Traceback (most recent call last)"


def signatures(text: str, max_lines: int = 40) -> list[str]:
    """Les max_lines premières lignes de signatures (def, async def, class, décorateurs)."""
    out = [ln.rstrip() for ln in text.splitlines() if _SIG_RE.match(ln)]
    return out[:max_lines]


def strip_traceback(issue: str) -> str:
    """Retire les blocs « Traceback ... » : l'en-tête, les lignes indentées qui
    suivent et la ligne d'exception finale. Sert à l'ablation sans trace de pile."""
    out, skipping = [], False
    for ln in issue.splitlines():
        if _TRACE_START in ln:
            skipping = True
            continue
        if skipping:
            if ln.startswith((" ", "\t")) or not ln.strip():
                continue
            skipping = False  # ligne d'exception finale, ex. « KeyError: 'exp' »
            continue
        out.append(ln)
    return "\n".join(out)


def build_prompt(issue: str, candidates: list[tuple[str, str]], max_sig_lines: int = 40,
                 stack_trace: bool = True) -> str:
    """candidates : (chemin, texte du fichier), dans l'ordre de la base."""
    if not stack_trace:
        issue = strip_traceback(issue)
    parts = [
        "You are given a GitHub issue and candidate Python files from the repository.",
        "Rank the candidate files from most to least likely to need a change to fix the issue.",
        "Answer with the file paths only, one per line, most likely first.",
        "", "### Issue", issue.strip(), "", "### Candidates",
    ]
    for i, (path, text) in enumerate(candidates, start=1):
        parts.append(f"[{i}] {path}")
        parts.extend(signatures(text, max_sig_lines) or ["(no signatures)"])
        parts.append("")
    parts.append("### Ranking")
    return "\n".join(parts)


def parse_ranking(output: str, candidates: list[str]) -> list[str]:
    """Liste ordonnée de chemins tirée d'une sortie libre.

    Ordre de première apparition des chemins candidats (les plus longs essayés
    d'abord) ; à défaut, indices « [3] » ou « 3. » ; candidats absents ajoutés
    ensuite dans l'ordre d'origine. Le résultat est une permutation des candidats.
    """
    found: list[str] = []
    if candidates:
        alt = "|".join(re.escape(c) for c in sorted(candidates, key=len, reverse=True))
        for m in re.finditer(alt, output or ""):
            if m.group(0) not in found:
                found.append(m.group(0))
        if not found:
            for m in re.finditer(r"\[(\d+)\]|^\s*(\d+)[.)]", output or "", re.MULTILINE):
                i = int(m.group(1) or m.group(2)) - 1
                if 0 <= i < len(candidates) and candidates[i] not in found:
                    found.append(candidates[i])
    return found + [c for c in candidates if c not in found]


def should_rerank(scores: list[float], threshold: float) -> bool:
    """Déclenche si (s1 - s2) / |s1| < threshold. threshold <= 0 : jamais ;
    threshold >= 1 : toujours (si au moins deux candidats)."""
    if len(scores) < 2:
        return False
    if threshold >= 1:
        return True
    s1, s2 = sorted(scores, reverse=True)[:2]
    gap = (s1 - s2) / abs(s1) if s1 else 0.0
    return gap < threshold


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, int, int, float]:
        """Renvoie (texte, jetons d'entrée, jetons de sortie, secondes)."""


class FakeBackend(Backend):
    """Backend déterministe pour les tests : renvoie les candidats du prompt
    dans l'ordre inverse (ou l'ordre donné par `order`). Jetons = mots."""

    def __init__(self, order: list[str] | None = None):
        self.order = order
        self.calls = 0

    def generate(self, prompt: str) -> tuple[str, int, int, float]:
        self.calls += 1
        cands = re.findall(r"^\[\d+\] (\S+)$", prompt, re.MULTILINE)
        ranked = [c for c in self.order if c in cands] if self.order else list(reversed(cands))
        text = "\n".join(ranked)
        return text, len(prompt.split()), len(text.split()), 0.0


class LlamaCppBackend(Backend):
    """Serveur llama.cpp (`llama-server`), point d'entrée POST /completion."""

    def __init__(self, url: str = "http://127.0.0.1:8080", n_predict: int = 256, timeout: float = 600.0):
        self.url, self.n_predict, self.timeout = url.rstrip("/"), n_predict, timeout

    def generate(self, prompt: str) -> tuple[str, int, int, float]:
        body = json.dumps({"prompt": prompt, "n_predict": self.n_predict, "temperature": 0.0,
                           "cache_prompt": False}).encode("utf-8")
        req = urllib.request.Request(self.url + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        return (data.get("content", ""), int(data.get("tokens_evaluated", 0)),
                int(data.get("tokens_predicted", 0)), dt)


def rerank(issue: str, base: list[tuple[str, float]], texts: dict[str, str], backend: Backend,
           threshold: float, top_k: int = 10, stack_trace: bool = True) -> tuple[list[str], dict]:
    """Reclasse le top_k de `base` si la règle d'écart le demande.

    Renvoie (classement complet, coût) ; coût = triggered, tokens_in, tokens_out, seconds.
    """
    paths = [p for p, _ in base]
    cost = {"triggered": False, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
    if not should_rerank([s for _, s in base[:top_k]], threshold):
        return paths, cost
    head = paths[:top_k]
    prompt = build_prompt(issue, [(p, texts.get(p, "")) for p in head], stack_trace=stack_trace)
    out, tin, tout, sec = backend.generate(prompt)
    cost.update(triggered=True, tokens_in=tin, tokens_out=tout, seconds=sec)
    return parse_ranking(out, head) + paths[top_k:], cost
