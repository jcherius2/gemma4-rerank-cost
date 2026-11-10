"""Build kernel/input/candidates.jsonl, the input of `python -m code.stats --regen`.

Joins the public candidates (data/candidates_public.jsonl, a copy of the Kaggle dataset
jcherius/gemma4-loc-candidates-public) with the gold files that `locgate rebuild` derived
from the public pull request diffs (locgate/tasks_rebuilt.jsonl). No competition text is
needed: the scoring code reads only the candidates and the gold files.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    gold = {}
    with open(os.path.join(ROOT, "locgate", "tasks_rebuilt.jsonl"), encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            gold[t["instance_id"]] = t["gold"]
    out_dir = os.path.join(ROOT, "kernel", "input")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(os.path.join(ROOT, "data", "candidates_public.jsonl"), encoding="utf-8") as f, \
            open(os.path.join(out_dir, "candidates.jsonl"), "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            c = json.loads(line)
            c["gold"] = gold[c["instance_id"]]
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} tasks written to kernel/input/candidates.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
