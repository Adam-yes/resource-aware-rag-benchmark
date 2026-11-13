# Contributing

Contributions are welcome, especially if they make the study easier to reproduce or improve the code, the documentation or the analysis.

Before opening a pull request:

1. Set up the environment as described in `code/README.md`.
2. Run the tests: `cd code && RAGBENCH_FAKE=1 python -m pytest -q tests`. They need no GPU and no Ollama.
3. Do not add third-party documents, model weights, local vector stores or virtual environments.
4. Do not change files in `data/` or `code/protocol.yaml`: their hashes are recorded in `results/frozen_finalists.json`.
5. Keep public documentation in English.

For larger changes, please open an issue first and describe what should change and how it affects the reproduction.
