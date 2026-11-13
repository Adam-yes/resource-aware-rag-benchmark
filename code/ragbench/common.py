"""Paths, protocol, question sets, configuration grid and the SQLite result store."""
import hashlib
import itertools
import os
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
DATA = Path(os.environ.get("RAGBENCH_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("RAGBENCH_RESULTS", ROOT / "results"))
# Archive with the raw question files; only `prepare` reads it, the derived question sets ship in data/.
ARCHIVE = Path(os.environ.get("RAGBENCH_ARCHIVE", ROOT / "_extracted/archive"))
PROTOCOL_PATH = CODE / "protocol.yaml"
OLLAMA_URL = os.environ.get("RAGBENCH_OLLAMA", "http://127.0.0.1:11434")
FIELDS = ["model", "chunk_size", "chunk_overlap", "top_k", "temperature"]


def protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def sha(text, n=16):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def cfg_id(c):
    return "|".join(str(c[f]) for f in FIELDS)


def parse_cfg(cid):
    m, cs, ov, k, t = cid.rsplit("|", 4)
    return {"model": m, "chunk_size": int(cs), "chunk_overlap": int(ov), "top_k": int(k), "temperature": float(t)}


def grid_configs(p=None):
    g = (p or protocol())["grid"]
    return [dict(zip(FIELDS, v)) for v in itertools.product(*(g[f] for f in FIELDS))]


def questions(split=None):
    """split: 'dev', 'test_a', 'test_b' or None for all. Applies human reference repairs if present."""
    df = pd.read_csv(DATA / "questions.csv", dtype={"qid": str})
    repairs = DATA / "reference_repairs.csv"
    if repairs.exists():
        fix = pd.read_csv(repairs, dtype={"qid": str}).dropna(subset=["repaired_reference"])
        fix = fix.set_index("qid").repaired_reference
        df["reference"] = df.qid.map(fix).fillna(df.reference)
    return df if split is None else df[df.split == split].reset_index(drop=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval(chunk_size INT, chunk_overlap INT, qid TEXT, docs TEXT, ms REAL,
  PRIMARY KEY(chunk_size, chunk_overlap, qid));
CREATE TABLE IF NOT EXISTS answers(cfg TEXT, qid TEXT, seed INT, answer TEXT, ctx_chars INT, wall_s REAL, total_s REAL,
  load_s REAL, prompt_s REAL, eval_s REAL, in_tok INT, out_tok INT, PRIMARY KEY(cfg, qid, seed));
CREATE TABLE IF NOT EXISTS scores(judge TEXT, ref_sha TEXT, ans_sha TEXT, f1 REAL, error TEXT, seconds REAL,
  PRIMARY KEY(judge, ref_sha, ans_sha));
CREATE TABLE IF NOT EXISTS measurements(kind TEXT, cfg TEXT, qid TEXT, rep INT, level INT, value REAL, extra TEXT);
CREATE TABLE IF NOT EXISTS models(model TEXT PRIMARY KEY, digest TEXT, size_bytes INT, size_vram INT, cold_load_s REAL);
"""


def db():
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(os.environ.get("RAGBENCH_DB", RESULTS / "ragbench.sqlite"), timeout=60)
    con.executescript(SCHEMA)
    return con


def scored_answers(judge, split="dev", seed=42, cfgs=None):
    """One row per answer with its judge score; only questions of `split`."""
    q = questions(split)[["qid", "reference", "source_file", "process", "query_style", "query_length"]]
    q["ref_sha"] = q.reference.map(sha)
    with db() as con:
        a = pd.read_sql("SELECT * FROM answers WHERE seed=?", con, params=(seed,))
        s = pd.read_sql("SELECT ref_sha, ans_sha, f1 FROM scores WHERE judge=?", con, params=(judge,))
    a = a.merge(q, on="qid")
    if cfgs is not None:
        a = a[a.cfg.isin(cfgs)]
    a["ans_sha"] = a.answer.map(sha)
    return a.merge(s, on=["ref_sha", "ans_sha"], how="left")


def config_table(judge, split="dev", seed=42):
    """Mean F1 and sequential latency per configuration."""
    a = scored_answers(judge, split, seed)
    a = a[a.cfg.str.count(r"\|") == 4]  # Configurations only, not baselines and judge controls.
    t = a.groupby("cfg").agg(f1=("f1", "mean"), n_answers=("f1", "size"), n_scored=("f1", "count"), gen_s=("wall_s", "mean"),
                             latency_p95_s=("wall_s", lambda x: x.quantile(0.95)), out_tok=("out_tok", "mean")).reset_index()
    t["f1_failures_zero"] = a.assign(f1=a.f1.fillna(0)).groupby("cfg").f1.mean().reindex(t.cfg).values
    t = pd.concat([t, pd.DataFrame([parse_cfg(c) for c in t.cfg])], axis=1)
    with db() as con:
        r = pd.read_sql("SELECT chunk_size, chunk_overlap, qid, ms FROM retrieval", con)
    # Median: the first query on a collection also loads its index from disk.
    r = r[r.qid.isin(questions("dev").qid)].groupby(["chunk_size", "chunk_overlap"]).ms.median().rename("retrieval_ms").reset_index()
    t = t.merge(r, on=["chunk_size", "chunk_overlap"], how="left")
    t["latency_s"] = t.gen_s + t.retrieval_ms.fillna(0) / 1000  # What an operator waits for.
    return t


def grid_table(judge, p=None):
    t = config_table(judge)
    return t[t.cfg.isin({cfg_id(c) for c in grid_configs(p)})].reset_index(drop=True)
