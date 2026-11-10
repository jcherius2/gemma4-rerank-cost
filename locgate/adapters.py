"""locgate adapters: third-party localizer outputs -> run records (schema.md).

Three formats:
  ranked   JSON object {instance_id: [path, ...]} (best first), e.g. a retriever dump.
  picks    JSONL with `instance_id` and `picks` (best first), optional seconds/prompt_tokens/seed/
           condition, filtered with --where (e.g. --where condition=paths --where seed=0).
  swebench JSONL, one object per line with `instance_id` and either `found_files` (Agentless-style
           localization) or `model_patch` (SWE-bench prediction): paths are read from the
           `--- a/` / `+++ b/` headers in order of first appearance (`/dev/null` skipped).

    python ressource/locgate/adapters.py ressource/locgate/examples/ranked.json \
        --run-id demo/ranked --model my-retriever --out ressource/locgate/records/demo-ranked.jsonl

Only ids, paths and numbers are kept (never issue text or patch bodies). Apache-2.0, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import validate, write_records  # noqa: E402

_HEADER = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?(\S+)")


def patch_paths(patch: str) -> list[str]:
    """Repo-relative paths touched by a unified diff, in order of first appearance."""
    out: list[str] = []
    for line in (patch or "").splitlines():
        m = _HEADER.match(line)
        if m and m.group(1) != "/dev/null" and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _record(iid: str, ranking: list[str], run_id: str, model: str, extra: dict | None = None, **kw) -> dict:
    extra = extra or {}
    rank = list(dict.fromkeys(p for p in ranking if isinstance(p, str) and p))
    return validate({
        "run_id": run_id, "instance_id": iid, "ranking": rank, "input_order": None,
        "seconds": float(extra.get("seconds", kw.get("seconds", 0.0))),
        "tokens_in": extra.get("tokens_in"), "tokens_out": extra.get("tokens_out"),
        "model": model, "hardware": kw.get("hardware"), "seed": kw.get("seed"),
        "condition": kw.get("condition")})


def from_ranked_json(path: str | Path, run_id: str, model: str, **kw) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object {{instance_id: [paths]}}")
    return [_record(iid, paths, run_id, model, **kw) for iid, paths in data.items()]


def from_swebench_jsonl(path: str | Path, run_id: str, model: str | None = None, **kw) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "found_files" in r:
                ranking = list(r["found_files"] or [])
            elif "model_patch" in r:
                ranking = patch_paths(r["model_patch"])
            else:
                raise ValueError(f"{path}:{n}: neither found_files nor model_patch")
            m = model or r.get("model_name_or_path") or "unknown"
            out.append(_record(r["instance_id"], ranking, run_id, m, r, **kw))
    return out


def from_picks_jsonl(path: str | Path, run_id: str, model: str, where: dict | None = None, **kw) -> list[dict]:
    """JSONL with `instance_id` and `picks` (best first), optional `seconds`, `prompt_tokens`, `seed`,
    `condition`, as in a reranker log with several conditions and seeds per task. `where` keeps only the
    lines whose fields equal the given values, e.g. {"condition": "paths", "seed": 0}."""
    where = where or {}
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if any(str(r.get(k)) != str(v) for k, v in where.items()):
                continue
            if not isinstance(r.get("picks"), list):
                raise ValueError(f"{path}:{n}: no picks list")
            extra = {"seconds": r.get("seconds", kw.get("seconds", 0.0)), "tokens_in": r.get("prompt_tokens")}
            kw2 = dict(kw, seed=r.get("seed", kw.get("seed")), condition=r.get("condition", kw.get("condition")))
            out.append(_record(r["instance_id"], r["picks"], run_id, model, extra, **kw2))
    return out


def convert(path: str | Path, fmt: str = "auto", **kw) -> list[dict]:
    if fmt == "auto":
        fmt = "ranked" if Path(path).suffix == ".json" else "swebench"
    fn = {"ranked": from_ranked_json, "swebench": from_swebench_jsonl, "picks": from_picks_jsonl}[fmt]
    return fn(path, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--format", choices=("auto", "ranked", "swebench", "picks"), default="auto",
                    help="auto: .json -> ranked, otherwise swebench")
    ap.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                    help="picks only: keep lines with this field value (repeatable)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=None, help="default: model_name_or_path (swebench) or 'unknown'")
    ap.add_argument("--seconds", type=float, default=0.0, help="per-task seconds when the input has none")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fmt = a.format if a.format != "auto" else ("ranked" if Path(a.input).suffix == ".json" else "swebench")
    model = a.model or (None if fmt == "swebench" else "unknown")
    kw = {"where": dict(w.split("=", 1) for w in a.where)} if fmt == "picks" else {}
    recs = convert(a.input, fmt, run_id=a.run_id, model=model, seconds=a.seconds,
                   hardware=a.hardware, condition=a.condition, **kw)
    write_records(recs, a.out)
    print(f"{a.out}: {len(recs)} records, run {a.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
