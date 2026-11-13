<p align="center">
  <img src="docs/assets/header.svg" alt="Resource-Aware RAG Benchmark: 960 configurations measured, 61,736 answers judged, 43 finalists frozen, selected setup ranked 2 of 43 on unused questions" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Adam-yes/resource-aware-rag-benchmark/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Adam-yes/resource-aware-rag-benchmark/ci.yml?branch=main&label=CI&style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-2F6BFF?style=flat-square">
  <img alt="License" src="https://img.shields.io/github/license/Adam-yes/resource-aware-rag-benchmark?style=flat-square">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local%20LLMs-1F2933?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/paper-under%20review-E0A13D?style=flat-square">
</p>

<p align="center">
  <a href="#motivation">Motivation</a> ·
  <a href="#protocol">Protocol</a> ·
  <a href="#results">Results</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="#data-and-models">Data and models</a>
</p>

---

## Motivation

Maintenance, troubleshooting and operator support in manufacturing rely on technical manuals. A retrieval-augmented generation (RAG) pipeline can answer questions about these manuals in natural language, but its answer quality depends on several settings that interact: chunk size, chunk overlap, retrieval depth, the language model and the decoding temperature. On local hardware, latency and memory matter as much as answer quality.

This repository measures the whole configuration space instead of sampling it, and then asks three questions on the complete table: how quality and latency trade off, whether configuration search does better than random search, and whether a configuration selected on development questions holds up on questions it has never seen.

The repository accompanies the manuscript "Resource-Aware Retrieval-Augmented Technical Knowledge Access for Smart Manufacturing: Quality-Latency Trade-offs and Configuration Search on a Full-Factorial Benchmark", which is currently under review.

## Protocol

<p align="center">
  <img src="docs/assets/protocol.svg" alt="Protocol: measure, choose judge, replay search, hold out manuals, freeze, test once" width="100%">
</p>

- All 960 combinations of eight local models, four chunk sizes, two chunk overlaps, five Top-k values and three temperatures were measured on 50 development questions, one request at a time.
- A judge from a different vendor than the benchmark generator (gpt-oss:20b) scored the answers with RAGAS Factual Correctness. It was chosen by a fixed rule against blind ratings; qwen3-coder:30b is the second judge.
- On the finished lookup table, a genetic algorithm, a coevolutionary variant, TPE, coordinate ascent and random search were replayed with 50 seeds at budgets of 25 to 150, and NSGA-II, MO-TPE and random search for quality and latency together.
- Search results were validated by leaving one manual out. 43 finalists were then frozen, with SHA-256 hashes of protocol, code and data, before they were scored once on 162 unused and 40 independently written questions.

All settings are in [code/protocol.yaml](code/protocol.yaml).

## Results

<p align="center">
  <img src="docs/assets/quality-vs-latency.svg" alt="Development F1 against latency for all 960 configurations with the Pareto front" width="100%">
</p>

For deployment the trade-off matters more than the single best score. A gemma3n e2b setup reaches 92% of the best F1 at half the latency, and a 2B model reaches 80% at less than a quarter.

<p align="center">
  <img src="docs/assets/search-vs-random.svg" alt="Search methods against random search, inside the lookup table and on held-out manuals" width="100%">
</p>

Within the table, the genetic algorithm, TPE and coordinate ascent found better configurations than random search from a budget of 50 evaluations on. On held-out manuals, none of them beat random search.

<p align="center">
  <img src="docs/assets/held-out-test.svg" alt="Development against test F1 for the 43 frozen finalists" width="100%">
</p>

The configuration selected on the development questions ranked 2nd of 43 on 162 unused questions and 4th of 43 on 40 independently written questions.

## Repository layout

| Path | Content |
|---|---|
| `code/` | Pipeline, analysis and tests ([code/README.md](code/README.md)) |
| `code/protocol.yaml` | All experimental settings, fixed before the run |
| `data/questions.csv` | Dev, unused and independently written questions with references |
| `data/reference_repairs.csv` | Six dev references rewritten from the manual text before the run |
| `data/test_b_human.csv`, `data/test_b_comments.csv` | Independently written questions (LLM-written, see below) and safety flags |
| `results/ragbench.sqlite.gz` | Every answer, score and timing (gzip, SQLite) |
| `results/answers.csv.gz` | Answer-level table: answer, F1 of both judges, latency, tokens |
| `results/benchmark_table.csv` | One row per configuration: mean dev F1, latency |
| `results/optimizer_runs.csv`, `results/lomo*.csv` | Search replay and leave-one-manual-out runs |
| `results/frozen_finalists.json` | Finalists and hashes, written before any test question ran |
| `results/report.json`, `results/tables/` | Statistics reported in the manuscript |
| `results/human_*` | Rating sheets, keys and completed ratings |
| `plots/results_table.xlsx`, `plots/figures.ipynb` | Results table and the notebook that draws every figure of the manuscript from it; the figures are written to `plots/figures/` and are not stored here |
| `docs/assets/` | README figures and the script that draws them from `results/` |

The README figures are drawn by `python docs/assets/make_figures.py` (standard library only).

## Reproduce

```bash
cd code
./setup_env.sh --all
RAGBENCH_FAKE=1 .venv/bin/python -m pytest -q tests      # no GPU or Ollama needed
gunzip -k ../results/ragbench.sqlite.gz
.venv/bin/python -m ragbench analyze                     # statistics from the stored results
.venv/bin/python -c "from ragbench.paper import figures; figures()"   # results table and figures in ../plots
```

A full rerun (`./run_all.sh prep`, then `./run_all.sh main`) needs the manuals for the Chroma collections, which are not redistributed.

## Hardware and run time

The experiment ran on a workstation with an NVIDIA RTX A6000 (48 GB GPU memory), an Intel Core i9-14900K and 64 GB RAM, using Ollama 0.40.14, and took about four days of run time. Generating all 61,736 answers took 5.0 hours; most of the remaining time went into scoring them with the judges. Latencies on this GPU are lower than on the smaller devices usually installed at a machine; `results/report.json` and the manuscript also give CPU-only latencies.

## Work done by LLMs

LLM agents wrote and checked the 40 independently written questions, rewrote six dev references, and rated answers blind for the judge decision and the test answers. These are LLM ratings. One author rated 40 answers; this is the only human rating.

## Changes after the freeze

`results/frozen_finalists.json` records hashes of the protocol, the code and the data at the freeze. Afterwards the following changed:

- Reporting code: `analysis.py`, `paper.py`, `dashboard.py` and a path in `common.py`.
- For publication, the Python package was renamed to `ragbench` and the header comment of `code/protocol.yaml` was reworded. No setting changed. `protocol_sha256` was recomputed for the published file; at the freeze it was `201d849066421981cd9617782f61c61c1967aebd95cb3e5c7c8a4d488781997e`. `code_sha256` still refers to the code as it was at the freeze.

Generation, judging and search are unchanged. `load_frozen` checks the protocol and data hashes.

## Data and models

The repository does not contain the technical manuals, model weights or local vector databases. The manuals are third-party documents and are not redistributed; the models are pulled through Ollama and are subject to their own licenses (names and tags in [code/protocol.yaml](code/protocol.yaml)).

## Citation

Citation details will be added after publication. Until then, please cite the repository itself (see [CITATION.cff](CITATION.cff)).

## License

Code and documentation are licensed under the Apache License 2.0. Data and model weights are subject to their own licenses.
