"""Everything that touches Ollama, Chroma or RAGAS. RAGBENCH_FAKE=1 swaps it for a deterministic
stand-in so the whole pipeline can be tested without a GPU."""
import os
import time
from functools import lru_cache

import requests

from .common import DATA, OLLAMA_URL, sha

FAKE = os.environ.get("RAGBENCH_FAKE") == "1"

# Unchanged from the archived pipeline.
PROMPT = """### Instruction:
You are a concise assistant. Answer the Question strictly using only the provided Context.
Rules:
1. If the answer is not explicitly in the Context, output exactly: "I don't know."
2. Do not explain why you don't know.
3. Keep the answer short.
4. Do not use outside knowledge.
### Context:
{context}
### Question:
{question}
### Answer:
"""
# Baseline without retrieval. The prompt above could only ever answer "I don't know." at k=0,
# which is what inflated the Top-k correlation in the archived grid.
CLOSED_BOOK_PROMPT = """### Instruction:
You are a concise assistant. Answer the Question from your own knowledge. Keep the answer short.
### Question:
{question}
### Answer:
"""


def _fake_vec(text):
    return [int(sha(text + str(i), 2), 16) / 255 for i in range(4)]


def embed(text, model):
    if FAKE:
        return _fake_vec(text)
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": text}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"][0]


@lru_cache(maxsize=1)
def _chroma():
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    import chromadb
    return chromadb.PersistentClient(path=str(DATA / "chroma_db"))


def retrieve(question, chunk_size, overlap, k, model):
    """Top-k chunks from the archived index. Chroma returns them by similarity, so a stored
    top-10 list also answers every smaller k."""
    if FAKE:
        return [f"chunk {i} of {chunk_size}/{overlap} for {question}" for i in range(k)]
    col = _chroma().get_collection(f"chroma_{chunk_size}_{overlap}")
    return col.query(query_embeddings=[embed(question, model)], n_results=k)["documents"][0]


def format_context(docs):
    return "\n\n".join(f"Context {i + 1}: {d}" for i, d in enumerate(docs))


def chat(prompt, model, temperature, opts, extra=None):
    """One blocking request. Returns answer text and timings in seconds."""
    t0 = time.perf_counter()
    if FAKE:
        # Stable, smooth toy landscape that rewards context and bigger models.
        n = max(prompt.count("Context "), 0)
        size = 1 + sum(map(ord, model)) % 7
        return {"answer": f"answer n={n} size={size} t={temperature} h={sha(prompt, 4)}",
                "wall_s": 0.1 * size + 0.01 * n, "total_s": 0.1 * size, "load_s": 0.0, "prompt_s": 0.01, "eval_s": 0.09,
                "in_tok": 100 * n, "out_tok": 50}
    options = {"temperature": temperature, **opts, **(extra or {})}
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=3600, json={
        "model": model, "stream": False, "options": options,
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    j = r.json()
    return {"answer": j["message"]["content"], "wall_s": time.perf_counter() - t0,
            "total_s": j.get("total_duration", 0) / 1e9, "load_s": j.get("load_duration", 0) / 1e9,
            "prompt_s": j.get("prompt_eval_duration", 0) / 1e9, "eval_s": j.get("eval_duration", 0) / 1e9,
            "in_tok": j.get("prompt_eval_count", 0), "out_tok": j.get("eval_count", 0)}


def model_info(model):
    if FAKE:
        return {"digest": "fake", "size_bytes": 1, "size_vram": 1}
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30).json()["models"]
    t = next(m for m in tags if m["name"] == model)
    ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=30).json().get("models", [])
    vram = next((m.get("size_vram", 0) for m in ps if m["name"] == model), 0)
    return {"digest": t["digest"], "size_bytes": t["size"], "size_vram": vram}


def make_scorer(judge_label, cfg):
    """Async callable (answer, reference) -> F1, using the same RAGAS metric as the archived run.
    A label such as 'model@43' reruns that judge under another seed."""
    judge_model, _, seed = judge_label.partition("@")
    cfg = {**cfg, "seed": int(seed)} if seed else cfg
    if FAKE:
        async def fake(answer, reference):
            import re
            m = re.search(r"n=(\d+) size=(\d+)", answer)
            n, size = map(int, m.groups()) if m else (0, 0)
            jitter = int(sha(answer + reference + judge_label, 3), 16) / 16 ** 3
            return min(1.0, 0.05 * size + 0.04 * min(n, 8) + 0.25 * jitter)
        return fake
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # Nothing about these unpublished runs leaves the machine.
    from langchain_ollama import ChatOllama
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import FactualCorrectness
    from ragas.run_config import RunConfig
    llm = ChatOllama(model=judge_model, temperature=0, num_ctx=cfg["num_ctx"], seed=cfg["seed"], keep_alive="30m")
    metric = FactualCorrectness(mode="f1", atomicity=cfg["atomicity"], coverage=cfg["coverage"],
                                llm=LangchainLLMWrapper(llm, bypass_temperature=True))
    metric.init(RunConfig(timeout=720, max_retries=2, max_wait=30, seed=cfg["seed"]))  # As archived.

    async def score(answer, reference):
        return float(await metric.single_turn_ascore(SingleTurnSample(response=answer, reference=reference), timeout=720))
    return score

