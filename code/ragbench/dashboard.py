"""Local read-only progress dashboard: python -m ragbench.dashboard, then http://127.0.0.1:8770.
It can start nothing and binds to this machine only, because the results are unpublished."""
import json
import os
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import check_cfgs
from .common import DATA, OLLAMA_URL, RESULTS, cfg_id, config_table, db, grid_configs, protocol, questions, sha
from .search import pareto_mask

PORT = int(os.environ.get("RAGBENCH_PORT", 8770))
HTML = Path(__file__).with_name("dashboard.html")
GPU_HISTORY = deque(maxlen=240)  # 20 minutes at one sample per 5 s.


def gpu():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, util, used, total, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "util": float(util), "mem_used_gb": float(used) / 1024, "mem_total_gb": float(total) / 1024, "temp": float(temp), "power": float(power)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def sample_gpu():
    while True:
        g = gpu()
        if g:
            GPU_HISTORY.append({"t": time.time(), "util": g["util"]})
        time.sleep(5)


def ollama(path):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}{path}", timeout=2) as r:
            return json.load(r)
    except OSError:
        return None


def running_stages():
    out = subprocess.run(["pgrep", "-af", "--", "-m ragbench |ollama pull"], capture_output=True, text=True).stdout
    return ["setup" if "ollama pull" in line else line.split("-m ragbench ", 1)[1].split()[0]
            for line in out.splitlines() if ("-m ragbench " in line or "ollama pull" in line) and "pgrep" not in line and "bash" not in line.split()[1]]


def _load(p):
    with db() as con:
        answers = pd.read_sql("SELECT cfg, qid, seed, answer, wall_s FROM answers", con)
        scores = pd.read_sql("SELECT judge, ref_sha, ans_sha, f1, seconds FROM scores", con)
        kinds = dict(con.execute("SELECT kind, count(*) FROM measurements GROUP BY kind").fetchall())
    return answers, scores, kinds


def state():
    p = protocol()
    decision = RESULTS / "judge_decision.json"
    primary = json.loads(decision.read_text())["primary"] if decision.exists() else p["judge_primary"]
    other = p["judge_secondary"] if primary == p["judge_primary"] else p["judge_primary"]
    q = questions()
    answers, scores, kinds = _load(p)
    q["ref_sha"] = q.reference.map(sha)
    answers = answers.merge(q[["qid", "split", "ref_sha"]], on="qid", how="inner")
    answers["ans_sha"] = answers.answer.map(sha)

    def judged(judge, mask):
        pairs = answers.loc[mask, ["ref_sha", "ans_sha"]].drop_duplicates()
        s = scores[scores.judge == judge]
        done = pairs.merge(s, on=["ref_sha", "ans_sha"]) if len(pairs) else pairs
        return len(done), len(pairs)

    grid = {cfg_id(c) for c in grid_configs(p)}
    known = grid | set(p["archived_arms"])
    frozen_file = RESULTS / "frozen_finalists.json"
    frozen = json.loads(frozen_file.read_text()) if frozen_file.exists() else None
    n_test = int(q.split.isin(["test_a", "test_b"]).sum())
    is_dev = answers.split == "dev"
    is_test = answers.split != "dev"
    gen_dev = int((is_dev & (answers.seed == 42) & answers.cfg.isin(known)).sum())
    noise_n = p["noise"]["n_top"] + p["noise"]["n_random"]
    tags = ollama("/api/tags")
    installed = {m["name"] for m in tags["models"]} if tags else set()
    needed = {p["embedding_model"], p["judge_primary"], p["judge_secondary"], *p["grid"]["model"]}
    rated = lambda tag: sum((RESULTS / f"human_{tag}_DONE_rater{r}.csv").exists() for r in "AB")
    l = p["latency"]
    judge_dev = judged(primary, is_dev & answers.cfg.isin(known))
    judge_test = judged(primary, is_test)
    second = judged(other, (is_dev & answers.cfg.isin(check_cfgs(p))) | is_test)
    n_final = len(frozen["configs"]) if frozen else 0
    n_pareto = len(frozen["pareto"]) if frozen else 0
    timed = answers[answers.wall_s.notna() & answers.cfg.isin(known)]
    # Per model, because models finish one after another: the mean so far says little about the ones still to come.
    per_model = timed.groupby(timed.cfg.str.rsplit("|", n=4).str[0]).wall_s.mean()
    gen_s = float(per_model.mean()) if len(per_model) else None
    speed = lambda j: scores[scores.judge == j].seconds.dropna().mean() / p["judge"]["concurrency"]
    any_judge = scores.seconds.dropna().mean() / p["judge"]["concurrency"] if len(scores) else float("nan")
    judge_s, other_s = (None if pd.isna(x) else float(x) for x in (speed(primary) if pd.notna(speed(primary)) else any_judge,
                                                                   speed(other) if pd.notna(speed(other)) else any_judge))
    gen_left = 0.0
    if gen_s is not None:
        have = answers[is_dev & (answers.seed == 42) & answers.cfg.isin(known)].cfg.value_counts()
        for cid in known:
            gen_left += max(50 - have.get(cid, 0), 0) * per_model.get(cid.rsplit("|", 4)[0], gen_s)
    dev_answers = answers[is_dev]
    share = dev_answers[["qid", "answer"]].drop_duplicates().shape[0] / len(dev_answers) if len(dev_answers) else 0.56

    def stage(key, label, done, total, who="gpu"):
        return {"key": key, "label": label, "done": done, "total": max(total, 0), "who": who, "complete": total > 0 and done >= total, "eta_s": None}

    stages = [
        # The models are deleted at the end to save disk space; that does not undo this step.
        stage("setup", "Modelle installieren", len(needed) if (RESULTS / "report.json").exists() else len(installed & needed), len(needed), who="setup"),
        stage("generate-dev", "Antworten für 960 Konfigurationen", gen_dev, (len(grid) + len(p["archived_arms"])) * 50),
        stage("rating", "Stichprobe blind bewerten", rated("calibration"), 2, who="llm"),
        stage("judge-gate", "Judge anhand der Bewertungen wählen", int(decision.exists()), 1, who="auto"),
        stage("judge-dev", "Antworten bewerten", judge_dev[0], max(judge_dev[1], 1) if gen_dev else 0),
        stage("generate-noise", "Messrauschen mit weiteren Seeds", int((answers.seed != 42).sum()), noise_n * 50 * len(p["noise"]["seeds"])),
        stage("freeze", "Optimierer vergleichen und einfrieren", int(frozen is not None), 1, who="auto"),
        stage("generate-test", "Unberührte Testfragen beantworten", int(is_test.sum()), n_final * n_test),
        stage("judge-test", "Testantworten bewerten", judge_test[0], judge_test[1]),
        stage("second-judge", "Zweiter Judge als Gegenprobe", second[0], second[1]),
        stage("latency", "Latenz, CPU und Parallelität messen", sum(kinds.values()),
              n_final * 50 * l["repetitions"] + n_pareto * 20 + l["n_load_points"] * len(l["concurrency_levels"]) * l["concurrency_requests"] if frozen else 0, who="gpu"),
        stage("analyze", "Auswertung und Abbildungen", int((RESULTS / "report.json").exists()), 1, who="auto"),
    ]

    by_key = {x["key"]: x for x in stages}
    finals = n_final or sum(p["finalists"][k] for k in ("n_pareto", "n_selected", "n_coevolution", "n_random")) + 1 + len(p["archived_arms"])
    paretos = n_pareto or p["finalists"]["n_pareto"]
    if gen_s is not None and judge_s is not None:
        noise_answers = noise_n * 50 * len(p["noise"]["seeds"])
        n_grid_answers = (len(grid) + len(p["archived_arms"])) * 50
        test_answers = finals * n_test
        checks = p["judge_check"]["n_grid_configs"] * 50 + test_answers
        baselines = 2 * len(p["grid"]["model"]) * 50
        controls = 3 * 50 + p["judge_controls"]["n_retest_configs"] * 50
        n_done_noise = int((answers.seed != 42).sum())
        forecast = {
            "generate-dev": gen_left,
            "rating": 0 if by_key["rating"]["complete"] else 1800,
            "judge-dev": max(n_grid_answers * share - judge_dev[0], 0) * judge_s,
            "generate-noise": (noise_answers - n_done_noise) * gen_s + (noise_answers * share + controls) * judge_s * (n_done_noise < noise_answers),
            "freeze": 0 if frozen else 1800,
            "generate-test": (test_answers - int(is_test.sum()) + baselines) * gen_s,
            "judge-test": max((test_answers + baselines) * share - judge_test[0], 0) * judge_s,
            "second-judge": max(checks * share - second[0], 0) * other_s,
            "latency": 0 if by_key["latency"]["complete"] else
                       (finals * 50 * l["repetitions"] + l["n_load_points"] * len(l["concurrency_levels"]) * l["concurrency_requests"]) * gen_s
                       + paretos * 20 * gen_s * 25,  # shortcut: CPU assumed 25 times slower than GPU until measured.
            "analyze": 0 if by_key["analyze"]["complete"] else 600,
        }
        for key, seconds in forecast.items():
            by_key[key]["eta_s"] = None if by_key[key]["complete"] or seconds < 1 else float(seconds)
    remaining = sum(x["eta_s"] or 0 for x in stages) if gen_s is not None and judge_s is not None else None

    table = []
    if len(scores):
        t = config_table(primary).dropna(subset=["f1"])
        t = t[t.cfg.isin(grid)]
        if len(t):
            t["front"] = pareto_mask(t.f1, t.latency_s)
            table = t[["cfg", "model", "chunk_size", "chunk_overlap", "top_k", "temperature", "f1", "latency_s", "n_scored", "front"]].to_dict("records")
    log = RESULTS / "run.log"
    run = running_stages()
    report = RESULTS / "report.json"
    fails = int(scores[scores.judge == primary].f1.isna().sum())
    ps = ollama("/api/ps")
    return {
        "time": time.time(), "demo": os.environ.get("RAGBENCH_DEMO") == "1", "running": run,
        "remaining_s": remaining, "finish_ts": time.time() + remaining if remaining is not None else None,
        "go": bool(len(answers) or set(run) - {"setup"}), "stages": stages, "primary": primary, "judge_decided": decision.exists(),
        "gpu": gpu(), "gpu_history": list(GPU_HISTORY), "ollama_up": tags is not None,
        "loaded": [m["name"] for m in ps["models"]] if ps else [],
        "answers": len(answers), "judgments": int((scores.judge == primary).sum()), "judge_failures": fails,
        "avg_gen_s": gen_s, "avg_judge_s": judge_s,
        "models_missing": sorted(needed - installed), "table": table,
        "report": json.loads(report.read_text()) if report.exists() else None,
        "log": log.read_text(errors="replace").splitlines()[-14:] if log.exists() else [],
        "paused": next((json.loads(f.read_text()) for f in (RESULTS / "PAUSED.json", RESULTS / "GUARD_PAUSED.json") if f.exists()), None),
        "human": [
            {"label": "Referenzen prüfen", "detail": "LLM, 8 markierte Referenzen, gegen die Handbücher", "done": (DATA / "reference_repairs.csv").exists()},
            {"label": "Unabhängige Testfragen schreiben", "detail": "LLM-Agent ohne Kenntnis der Ergebnisse, rund 40 Fragen", "done": (DATA / "test_b_human.csv").exists()},
            {"label": "Stichprobe 1 blind bewerten", "detail": "Zwei LLM-Agenten, 100 Antworten", "done": rated("calibration") == 2},
            {"label": "Stichprobe 2 blind bewerten", "detail": "Zwei LLM-Agenten, 50 Testantworten", "done": rated("test") == 2},
            {"label": "Empfohlen: 40 Antworten selbst prüfen", "detail": "Du, ganz am Ende, rund 30 min, mit Empfehlungen. Nur das darf im Paper menschliche Prüfung heißen", "done": rated("person") >= 1},
        ],
    }


def finite(x):
    """JSON for a browser: NaN and infinity become null, numpy numbers become plain ones."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body, kind = HTML.read_bytes(), "text/html; charset=utf-8"
        elif self.path == "/api/state":
            try:
                body, kind = json.dumps(finite(cached_state()), allow_nan=False).encode(), "application/json"
            except Exception as e:  # Keep serving while a stage rewrites its files.
                body, kind = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


_cache = {"t": 0, "v": None}
_lock = threading.Lock()


def cached_state():
    with _lock:
        if time.time() - _cache["t"] > 4:
            _cache.update(t=time.time(), v=state())
        return _cache["v"]


if __name__ == "__main__":
    threading.Thread(target=sample_gpu, daemon=True).start()
    print(f"Dashboard: http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
