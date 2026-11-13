"""Resource measurements of frozen operating points: repeated sequential timing, CPU-only, concurrency."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import backend as B
from .common import cfg_id, db, parse_cfg, protocol, questions


def _prompts(c, qs):
    with db() as con:
        ctx = {r[0]: json.loads(r[1]) for r in con.execute(
            "SELECT qid, docs FROM retrieval WHERE chunk_size=? AND chunk_overlap=?", (c["chunk_size"], c["chunk_overlap"]))}
    return [(q.qid, B.PROMPT.format(context=B.format_context(ctx[q.qid][:c["top_k"]]), question=q.question)) for q in qs.itertuples()]


def _done(kind):
    """Rows of one configuration are saved together, so a restart can skip what is complete."""
    with db() as con:
        return set(con.execute("SELECT DISTINCT cfg, level FROM measurements WHERE kind=?", (kind,)))


def _save(rows):
    with db() as con:
        con.executemany("INSERT INTO measurements VALUES(?,?,?,?,?,?,?)", rows)


def sequential(cfgs):
    """Server must run with 1 slot. Warm model, one request at a time."""
    p = protocol()
    l = p["latency"]
    qs = questions("dev")
    opts = p["generation"]
    done = _done("gpu_seq_s")
    for c in cfgs:
        if (cfg_id(c), 0) in done:
            continue
        prompts = _prompts(c, qs)
        B.chat(prompts[0][1], c["model"], c["temperature"], opts)
        _save([("gpu_seq_s", cfg_id(c), qid, rep, 0, B.chat(prompt, c["model"], c["temperature"], opts)["wall_s"], "")
               for rep in range(l["repetitions"]) for qid, prompt in prompts])


def cpu_only(cfgs, n_questions=20):
    """Proxy for a machine without a GPU: num_gpu=0 on a fixed thread count."""
    p = protocol()
    qs = questions("dev").head(n_questions)
    extra = {"num_gpu": 0, "num_thread": p["latency"]["cpu_threads"]}
    done = _done("cpu_seq_s")
    for c in cfgs:
        if (cfg_id(c), 0) in done:
            continue
        prompts = _prompts(c, qs)
        B.chat(prompts[0][1], c["model"], c["temperature"], p["generation"], extra)
        _save([("cpu_seq_s", cfg_id(c), qid, 0, 0, B.chat(pr, c["model"], c["temperature"], p["generation"], extra)["wall_s"], "")
               for qid, pr in prompts])


def concurrency(cfgs):
    """N simulated users send requests back to back. Server must run with several slots."""
    p = protocol()
    l = p["latency"]
    qs = questions("dev")
    done = _done("concurrent_s")
    for c in cfgs:
        prompts = [pr for _, pr in _prompts(c, qs)]
        if all((cfg_id(c), level) in done for level in l["concurrency_levels"]):
            continue
        B.chat(prompts[0], c["model"], c["temperature"], p["generation"])  # Warm up: the server was just restarted.
        for level in l["concurrency_levels"]:
            if (cfg_id(c), level) in done:
                continue
            jobs = [prompts[i % len(prompts)] for i in range(l["concurrency_requests"])]
            with ThreadPoolExecutor(level) as ex:
                walls = list(ex.map(lambda pr: B.chat(pr, c["model"], c["temperature"], p["generation"])["wall_s"], jobs))
            _save([("concurrent_s", cfg_id(c), "", i, level, w, "") for i, w in enumerate(walls)])


def operating_points(frozen, n):
    """Evenly spaced along the frozen Pareto front, from fastest to best."""
    front = frozen["pareto"]
    return [parse_cfg(front[i]) for i in sorted({round(x) for x in np.linspace(0, len(front) - 1, n)})]
