# locgate run record (JSONL)

One JSON object per line, one line per (task, run). A file may hold several runs, told apart by `run_id`.
Records hold only ids, file paths and numbers: never issue text, prompts or raw model output.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `run_id` | str | yes | Identifier of the run, e.g. `gemma-4-e4b-it/t4-fp16/trace`. |
| `instance_id` | str | yes | Task id (`<repo>_<pr>`), as in `tasks_rebuilt.jsonl`. |
| `ranking` | list[str] | yes | Files returned by the localizer, best first (repo-relative paths). |
| `input_order` | list[str] or null | no | Candidates in the order shown to the model (needed for position bias). |
| `seconds` | float | yes | Wall-clock seconds spent on this task (0 if the localizer is free). |
| `tokens_in` | int or null | no | Prompt tokens, if known. |
| `tokens_out` | int or null | no | Generated tokens, if known. |
| `model` | str | yes | Model or method name. |
| `hardware` | str or null | no | E.g. `Tesla T4 / torch.float16`, `CPU / llama.cpp Q4_K_M`. |
| `seed` | int or null | no | Sampling or shuffle seed (null for greedy, unshuffled). |
| `condition` | str or null | no | Prompt variant (`trace`, `notrace`, ...). |

Scoring (`score.py`): a ranking of top-k files is completed by the base (RRF top-50) files not in it,
so Acc@5 and MRR use the same tail for every localizer. Gold and base come from `tasks_rebuilt.jsonl`.

Convert the competition notebook outputs:

    python ressource/locgate/records.py results/rerank_gemma-4-e4b-it.jsonl results/rerank_gemma-4-e2b-it.jsonl \
        --base ressource/locgate/tasks_rebuilt.jsonl --out-dir ressource/locgate/records

Convert third-party localizer outputs (`adapters.py`): a JSON `{instance_id: [paths best first]}` (`ranked`),
or a SWE-bench/Agentless JSONL with `instance_id` and `found_files`, or `model_patch` whose `--- a/`/`+++ b/`
paths are kept in order of first appearance (`swebench`; `seconds`, `tokens_in/out` carried when present):

    python ressource/locgate/adapters.py ressource/locgate/examples/swebench.jsonl --run-id demo/agentless \
        --out ressource/locgate/records/demo-agentless.jsonl   # --format auto|ranked|swebench, --model, --seconds
    python ressource/locgate/score.py ressource/locgate/records/demo-agentless.jsonl --gold results/rrf.json

`examples/` holds two synthetic demo files (5 real task ids, plausible paths, no issue text).
