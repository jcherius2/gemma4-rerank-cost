"""locgate run records: schema (see schema.md), loading, validation and conversion.

Converts the notebook outputs `results/rerank_gemma-4-*.jsonl` (first line `meta`, then one line per
task and prompt condition) to run records. The `raw` model output is dropped: records keep only ids,
paths and numbers.

    python ressource/locgate/records.py results/rerank_gemma-4-e4b-it.jsonl ... \
        --base ressource/locgate/tasks_rebuilt.jsonl --out-dir ressource/locgate/records

Apache-2.0, standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ("run_id", "instance_id", "ranking", "seconds", "model")
FIELDS = REQUIRED + ("input_order", "tokens_in", "tokens_out", "hardware", "seed", "condition")


def validate(rec: dict) -> dict:
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise ValueError(f"record {rec.get('instance_id')!r}: missing {missing}")
    extra = set(rec) - set(FIELDS)
    if extra:
        raise ValueError(f"record {rec.get('instance_id')!r}: unknown fields {sorted(extra)}")
    if not isinstance(rec["ranking"], list) or not all(isinstance(p, str) for p in rec["ranking"]):
        raise ValueError(f"record {rec['instance_id']!r}: ranking must be a list of paths")
    return rec


def load_records(path: str | Path) -> dict[str, dict[str, dict]]:
    """run_id -> instance_id -> record."""
    runs: dict[str, dict[str, dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = validate(json.loads(line))
                runs.setdefault(r["run_id"], {})[r["instance_id"]] = r
    return runs


def load_tasks(path: str | Path) -> list[dict]:
    """Tasks with `instance_id`, `repo`, `gold`, `base` (list of paths, RRF order).

    Accepts `tasks_rebuilt.jsonl` (`top50`) or a candidates file whose `base` is [[path, score], ...];
    only ids, gold, paths and scores are kept (`scores` when present)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            row = {"instance_id": t["instance_id"], "repo": t.get("repo"), "gold": t["gold"]}
            if "top50" in t:
                row["base"] = list(t["top50"])
            else:
                row["base"] = [p for p, _ in t["base"]]
                row["scores"] = [s for _, s in t["base"]]
            out.append(row)
    return out


def load_gold(path: str | Path) -> dict[str, list[str]]:
    """instance_id -> gold files, from a tasks jsonl or a results json (`{"tasks": [...]}`, e.g. rrf.json)."""
    p = Path(path)
    if p.suffix == ".json":
        rows = json.load(open(p, encoding="utf-8"))["tasks"]
    else:
        rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return {r["instance_id"]: list(r["gold"]) for r in rows}


def with_gold(tasks: list[dict], gold: dict[str, list[str]]) -> list[dict]:
    missing = [t["instance_id"] for t in tasks if t["instance_id"] not in gold]
    if missing:
        raise ValueError(f"no gold for {len(missing)} tasks, e.g. {missing[:3]}")
    return [{**t, "gold": gold[t["instance_id"]]} for t in tasks]


def convert_notebook(path: str | Path, tasks: list[dict] | None = None, top_k: int | None = None) -> list[dict]:
    """Notebook rerank jsonl -> run records (one run per prompt condition)."""
    lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    meta = lines[0].get("meta", {}) if lines and "meta" in lines[0] else {}
    body = lines[1:] if meta else lines
    model = Path(str(meta.get("model_dir", "unknown")).rstrip("/")).parent.name or "unknown"
    k = top_k or meta.get("top_k", 10)
    hw = " / ".join(str(meta[x]) for x in ("device", "dtype") if meta.get(x)) or None
    base = {t["instance_id"]: t["base"][:k] for t in tasks or []}
    out = []
    for r in body:
        cond = r.get("cond")
        out.append(validate({
            "run_id": f"{model}/{(meta.get('device') or 'hw').replace(' ', '-').lower()}/{cond}",
            "instance_id": r["instance_id"], "ranking": list(r["ranked"]),
            "input_order": base.get(r["instance_id"]), "seconds": float(r["seconds"]),
            "tokens_in": r.get("tokens_in"), "tokens_out": r.get("tokens_out"),
            "model": model, "hardware": hw, "seed": None, "condition": cond}))
    return out


def write_records(recs: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="notebook rerank_*.jsonl files")
    ap.add_argument("--base", default=str(Path(__file__).with_name("tasks_rebuilt.jsonl")),
                    help="tasks file giving the input order (RRF top-k)")
    ap.add_argument("--out-dir", default=str(Path(__file__).with_name("records")))
    a = ap.parse_args(argv)
    tasks = load_tasks(a.base)
    for p in a.inputs:
        recs = convert_notebook(p, tasks)
        out = Path(a.out_dir) / (Path(p).stem.replace("rerank_", "") + ".jsonl")
        write_records(recs, out)
        print(f"{out}: {len(recs)} records, runs {sorted({r['run_id'] for r in recs})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
