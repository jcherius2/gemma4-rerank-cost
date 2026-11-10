# gemma4-rerank-cost

Code, run records and scoring tool behind two writeups for the Kaggle competition
[Google DeepMind - Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent):

- the paper "When is a small Gemma 4 worth calling?": Gemma 4 e2b and e4b rerank the top 10 files of a
  retrieval fusion (RRF) on the 129 public tasks, with gates that decide when the model is called;
- the resource "locgate": score any file localizer on the same 129 tasks, accuracy and cost together.

## Layout

| Path | Content |
|---|---|
| `code/` | Evaluation code of the paper (standard library only) and its pytest tests. |
| `results/` | Run outputs of the paper: one JSON line per task and condition for e2b and e4b (`rerank_*.jsonl`), the separate retrieval rankings (`separes.jsonl`), and the summaries and tables derived from them. |
| `locgate/` | The resource: record schema (`schema.md`), scorer (`score.py`), adapters, task rebuild from GitHub (`rebuild.py`, `tasks_rebuilt.jsonl`), and run records (`records/`). |
| `data/candidates_public.jsonl` | Copy of the Kaggle dataset [jcherius/gemma4-loc-candidates-public](https://www.kaggle.com/datasets/jcherius/gemma4-loc-candidates-public): top-50 RRF files and signature lines per task. |
| `make_cands.py` | Joins those candidates with the gold files of `locgate/tasks_rebuilt.jsonl`. |

## Reproduce the paper's tables

Python 3.11 or later, from the repository root:

```
python make_cands.py
python -m code.stats --regen
```

This regenerates Tables 1 and 2, Figure 1 and Section 4.3 (`results/*_summary.json`,
`results/declencheurs.json` and `.md`, `results/stats.json`, `results/position_bias.json`) from
the JSON lines, in about 35 s on a laptop CPU. On the published files the output is identical,
byte for byte, to the committed results. Tests: `pip install pytest`, then `cd code && python -m pytest -q`.

The model runs themselves are the Kaggle notebooks
[jcherius/gemma4-rerank-cost](https://www.kaggle.com/code/jcherius/gemma4-rerank-cost) (e4b) and
[jcherius/gemma4-rerank-cost-e2b](https://www.kaggle.com/code/jcherius/gemma4-rerank-cost-e2b)
(e2b), on a T4 with fp16 and greedy decoding.

## Score your own localizer with locgate

Write one record per task in the format of `locgate/schema.md`, then:

```
python locgate/score.py my.jsonl --ref locgate/records/gemma-4-e2b-it.jsonl --condition trace
```

## Data provenance

This repository holds no text from the competition data. Task ids are `<repo>_<PR number>`,
pointers to public pull requests of FastAPI, Rich, Requests and HTTPX. The gold files are rebuilt
from the public pull request diffs (`locgate/rebuild.py`); on the 129 tasks they match the
competition's gold files exactly. Run records hold ids, file paths and numbers only, never issue
text or prompts. Signature lines in `data/candidates_public.jsonl` are short excerpts of source
code under their own licenses: FastAPI (MIT, Sebastián Ramírez), Rich (MIT, Will McGugan),
Requests (Apache-2.0, Kenneth Reitz), HTTPX (BSD-3-Clause, Encode OSS Ltd.).

## Third-party records

`locgate/records/tiers/prena0808-*.jsonl` are converted from the public output of the Kaggle
notebook [prena0808/where-is-the-bug-localization-with-gemma-4-e4b](https://www.kaggle.com/code/prena0808/where-is-the-bug-localization-with-gemma-4-e4b)
by prena0808, released under the Apache 2.0 license. Changes: rankings converted to the locgate
record format; nothing else.

## Use of AI

The question, the code, the experiments, the records and this README were designed and produced by
Claude, an AI model made by Anthropic, working autonomously; the author entered the competition and
owns the accounts, but did not write or review the code. Gemma 4 e2b and e4b are the systems under
study; they produced the rankings in the records and none of the code or text.

## License

Apache License 2.0 (see `LICENSE`), copyright 2026 Cherius, except the third-party material named
above, which keeps its own license.
