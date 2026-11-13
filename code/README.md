# Code

Full-factorial benchmark of local Naive-RAG configurations, with search, selection and test scoring kept apart. All experimental settings are in `protocol.yaml`.

## Setup

```bash
./setup_env.sh --all
RAGBENCH_FAKE=1 .venv/bin/python -m pytest -q tests
```

`setup_env.sh` installs Ollama and all models inside `runtime/` at the repository root, without sudo or a system service. The tests run without GPU or Ollama (`RAGBENCH_FAKE=1` replaces Ollama, Chroma and RAGAS).

## Run

```bash
./run_all.sh prep   # answer sample for the judge decision
./run_all.sh main   # everything after the judge decision, in protocol order
```

Finished work is skipped, so both commands can be restarted after an interruption. `python -m ragbench status` shows the progress. Retrieval needs the Chroma collections built from the manuals, which are not redistributed.

## Rerun the analysis from the published results

```bash
gunzip -k ../results/ragbench.sqlite.gz
.venv/bin/python -m ragbench analyze
```

`analyze` reads only the database and the CSV files in `../data` and `../results`; it needs no GPU and no model.

## Manual inputs

| File | Who | Note |
|---|---|---|
| `data/reference_repairs.csv` | LLM | Six dev references rewritten from the manual text before the run |
| `data/test_b_human.csv` | LLM agent without access to results | Independently written test questions. Despite the file name, LLM-written |
| `results/human_calibration_DONE_rater{A,B}.csv` | Two LLM agents, blind | Decides the judge. LLM ratings, not human review |
| `results/human_test_DONE_rater{A,B}.csv` | Two LLM agents, blind | Ratings of test answers. LLM ratings |
| `results/human_person_DONE_raterA.csv` | One author | The only human rating, 40 answers |

## Layout

| Path | Content |
|---|---|
| `protocol.yaml` | All experimental settings, fixed before the run |
| `ragbench/prepare.py` | Question sets from the raw question files |
| `ragbench/backend.py` | Ollama, Chroma, RAGAS |
| `ragbench/run.py` | Retrieval cache, sequential generation, judging |
| `ragbench/search.py` | Optimizers on the lookup table, leave-one-manual-out, freeze |
| `ragbench/latency.py` | Repeated timing, CPU-only, concurrency |
| `ragbench/analysis.py` | Statistics and `report.json` |
| `ragbench/paper.py` | Numbers and tables for the manuscript, results table for `../plots` |
| `ragbench/dashboard.py` | Progress page for the running experiment |

`results/ragbench.sqlite` holds every answer, score and timing. `results/frozen_finalists.json` was written before any test question ran and records SHA-256 hashes of the protocol, the code and the data; `load_frozen` refuses to continue if protocol or data differ.
