"""Resumable LLM stages: retrieval cache, sequential generation, judging. Reruns skip finished work."""
import asyncio
import json
import random
import time

from . import backend as B
from .common import cfg_id, db, grid_configs, parse_cfg, protocol, questions, sha

MAX_K = 10


def ensure_retrieval(qs, configs, p):
    pairs = sorted({(c["chunk_size"], c["chunk_overlap"]) for c in configs})
    with db() as con:
        done = set(con.execute("SELECT chunk_size, chunk_overlap, qid FROM retrieval"))
        for cs, ov in pairs:
            for q in qs.itertuples():
                if (cs, ov, q.qid) in done:
                    continue
                t0 = time.perf_counter()
                docs = B.retrieve(q.question, cs, ov, MAX_K, p["embedding_model"])
                con.execute("INSERT INTO retrieval VALUES(?,?,?,?,?)",
                            (cs, ov, q.qid, json.dumps(docs), 1000 * (time.perf_counter() - t0)))
            con.commit()


def generate(configs, split, seed=42, limit_q=None, prompt_kind="rag"):
    """One request at a time, model by model, so timings hold no queueing and no judge load.
    The first request after a model switch only warms it up and is repeated."""
    p = protocol()
    qs = questions(split)
    qs = qs if limit_q is None else qs.head(limit_q)
    ensure_retrieval(qs, configs, p)
    opts = {**p["generation"], "seed": seed}
    with db() as con:
        ctx = {(r[0], r[1], r[2]): json.loads(r[3]) for r in con.execute("SELECT chunk_size,chunk_overlap,qid,docs FROM retrieval")}
        done = set(con.execute("SELECT cfg, qid FROM answers WHERE seed=?", (seed,)))
        warm = None
        # Shuffled within a model, so slow drift of the machine cannot pose as a parameter effect.
        order = random.Random(seed).sample(configs, len(configs))
        for c in sorted(order, key=lambda c: c["model"]):
            cid = cfg_id(c) if prompt_kind == "rag" else f"{prompt_kind}|{c['model']}"
            for q in qs.itertuples():
                if (cid, q.qid) in done:
                    continue
                docs = ctx[(c["chunk_size"], c["chunk_overlap"], q.qid)][:c["top_k"]]
                if prompt_kind == "closed_book":
                    prompt = B.CLOSED_BOOK_PROMPT.format(question=q.question)
                elif prompt_kind == "oracle":
                    prompt = B.PROMPT.format(context=q.reference_contexts, question=q.question)
                else:
                    prompt = B.PROMPT.format(context=B.format_context(docs), question=q.question)
                if warm != c["model"]:
                    cold = B.chat(prompt, c["model"], c["temperature"], opts)
                    info = B.model_info(c["model"])
                    con.execute("INSERT OR REPLACE INTO models VALUES(?,?,?,?,?)",
                                (c["model"], info["digest"], info["size_bytes"], info["size_vram"], cold["load_s"]))
                    warm = c["model"]
                r = B.chat(prompt, c["model"], c["temperature"], opts)
                con.execute("INSERT OR REPLACE INTO answers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, q.qid, seed, r["answer"], len(prompt), r["wall_s"], r["total_s"], r["load_s"],
                             r["prompt_s"], r["eval_s"], r["in_tok"], r["out_tok"]))
                done.add((cid, q.qid))
            con.commit()


def judge(judge_model, split, cfgs=None, keys=None):
    """Scores every answer once per (judge, reference, answer): identical answers share one judgment.
    cfgs: only these configurations. keys: only these (qid, answer) pairs."""
    p = protocol()
    qs = questions(split).set_index("qid")
    with db() as con:
        rows = con.execute("SELECT cfg, qid, answer FROM answers").fetchall()
        done = {(a, b) for a, b in con.execute("SELECT ref_sha, ans_sha FROM scores WHERE judge=? AND f1 IS NOT NULL", (judge_model,))}
    todo = {}
    for cid, qid, ans in rows:
        if qid in qs.index and (cfgs is None or cid in cfgs) and (keys is None or (qid, ans) in keys):
            ref = qs.loc[qid, "reference"]
            key = (sha(ref), sha(ans))
            if key not in done:
                todo[key] = (ans, ref)
    print(f"{judge_model}: {len(todo)} unique judgments to do on {split}")
    if not todo:
        return
    scorer = B.make_scorer(judge_model, p["judge"])

    async def go():
        sem = asyncio.Semaphore(p["judge"]["concurrency"])
        con = db()

        async def one(key, ans, ref):
            async with sem:
                t0 = time.perf_counter()
                f1, err = None, None
                for attempt in range(3):
                    try:
                        f1 = await scorer(ans, ref)
                        err = None
                        break
                    except Exception as e:  # Recorded per item; the archived run lost the reason.
                        err = f"{type(e).__name__}: {e}"[:300]
                con.execute("INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?)",
                            (judge_model, key[0], key[1], f1, err, time.perf_counter() - t0))
                con.commit()
        await asyncio.gather(*(one(k, a, r) for k, (a, r) in todo.items()))
        con.close()
    asyncio.run(go())
    with db() as con:
        failed = con.execute("SELECT count(*) FROM scores WHERE judge=? AND f1 IS NULL", (judge_model,)).fetchone()[0]
    # An outage must not pass as data: stop, let the service restart Ollama, and retry only these items.
    if failed > max(5, 0.01 * len(todo)):
        raise SystemExit(f"{judge_model}: {failed} judgments failed, stopping for a retry")


def noise_configs(p, table):
    n = p["noise"]
    pool = [c for c in grid_configs(p) if c["temperature"] > 0]
    top = list(table[table.temperature > 0].sort_values("f1", ascending=False).head(n["n_top"]).cfg)
    return [parse_cfg(c) for c in top] + random.Random(n["sample_seed"]).sample(
        [c for c in pool if cfg_id(c) not in top], n["n_random"])


def insert_controls(split="dev"):
    """Anchors with a known expected score: the reference itself, another question's reference, a refusal."""
    qs = questions(split)
    refs = qs.reference.tolist()
    with db() as con:
        for i, q in enumerate(qs.itertuples()):
            for kind, ans in (("reference", q.reference), ("wrong_question", refs[(i + 1) % len(refs)]), ("refusal", "I don't know.")):
                con.execute("INSERT OR REPLACE INTO answers(cfg, qid, seed, answer) VALUES(?,?,42,?)", (f"control|{kind}", q.qid, ans))


def archived_arms(p):
    return [parse_cfg(c) for c in p["archived_arms"]]


def frozen_configs():
    from .search import load_frozen
    return [parse_cfg(c) for c in load_frozen()["configs"]]
