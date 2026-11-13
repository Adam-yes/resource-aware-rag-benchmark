"""python -m ragbench <stage>. Every LLM stage is resumable. See run_all.sh for the order."""
import argparse
import json
import os
import random
import subprocess

import pandas as pd

from . import analysis as A
from . import latency as L
from . import prepare, run
from .common import RESULTS, ROOT, cfg_id, db, grid_configs, grid_table, parse_cfg, protocol, questions, scored_answers
from .search import freeze, load_frozen, replay, replay_lomo


def primary_judge(p):
    f = RESULTS / "judge_decision.json"
    if not f.exists():
        raise SystemExit("No judge decision yet: rate the calibration sheet, then run judge-gate.")
    return json.loads(f.read_text())["primary"]


def other_judge(p):
    prim = primary_judge(p)
    return p["judge_secondary"] if prim == p["judge_primary"] else p["judge_primary"]


def check_cfgs(p):
    return {cfg_id(c) for c in random.Random(p["judge_check"]["sample_seed"]).sample(grid_configs(p), p["judge_check"]["n_grid_configs"])}


def retest_cfgs(p):
    c = p["judge_controls"]
    return {cfg_id(x) for x in random.Random(c["sample_seed"]).sample(grid_configs(p), c["n_retest_configs"])}


def calibration_cfgs():
    return set(pd.read_csv(RESULTS / "human_calibration_KEY.csv").cfg)


def status():
    with db() as con:
        for sql, label in (("SELECT seed, count(DISTINCT cfg), count(*), avg(wall_s) FROM answers GROUP BY seed", "answers(seed, cfgs, rows, s/answer)"),
                           ("SELECT judge, count(*), sum(f1 IS NULL), avg(seconds) FROM scores GROUP BY judge", "scores(judge, rows, failed, s/item)"),
                           ("SELECT kind, count(*) FROM measurements GROUP BY kind", "measurements"),
                           ("SELECT model, size_vram, cold_load_s FROM models", "models")):
            print(label, con.execute(sql).fetchall())
    print("grid size", len(grid_configs()))


def analyze(p):
    primary, secondary = primary_judge(p), other_judge(p)
    df = grid_table(primary, p)
    df.to_csv(RESULTS / "benchmark_table.csv", index=False)
    runs = pd.read_csv(RESULTS / "optimizer_runs.csv")
    A.optimizer_stats(runs).to_csv(A._out("optimizers_conditional_on_table.csv"), index=False)
    A.factorial_anova(df).to_csv(A._out("anova_config_level.csv"))
    A.answer_level_effects(primary).to_csv(A._out("effects_answer_level.csv"))
    A.failures(primary).to_csv(A._out("judge_failures.csv"), index=False)
    A.lexical_coverage().to_csv(A._out("lexical_coverage.csv"), index=False)
    A.fig_regret(runs); A.fig_pareto(df); A.fig_heatmaps(df)
    report = {"failures_as_zero_spearman": float(df.f1.corr(df.f1_failures_zero, method="spearman")),
              "noise": A.noise_estimate(primary, [42] + p["noise"]["seeds"]),
              "judge_check": A.judge_check(primary, secondary, check_cfgs(p)),
              "judge_controls": A.judge_controls(primary, f"{primary}@{p['judge_controls']['retest_seed']}", retest_cfgs(p)),
              "llm_rater_calibration": A.rating_results("calibration", [p["judge_primary"], p["judge_secondary"]])}
    for name, f in (("all", "lomo.csv"), ("fdm", "lomo_fdm.csv")):
        s = A.lomo_stats(pd.read_csv(RESULTS / f), p)
        s.to_csv(A._out(f), index=False)
        report[f"lomo_{name}"] = s.round(4).to_dict("records")
    t, report["generalization"] = A.generalization(primary, load_frozen())
    if not t.empty:
        t.to_csv(A._out("generalization.csv"), index=False)
        A.fig_generalization(t)
    for tag in ("test", "person"):
        if (RESULTS / f"human_{tag}_DONE_raterA.csv").exists():
            report[f"rating_{tag}"] = A.rating_results(tag, [primary, secondary])
    (RESULTS / "report.json").write_text(json.dumps(report, indent=1, default=float))
    best = df.loc[df.f1.idxmax()]
    A.write_macros({"GridSize": len(df), "BestDevFone": best.f1, "BestDevLatency": best.latency_s})
    print(json.dumps(report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "status", "generate-dev", "calibration-export", "judge-calibration", "judge-gate", "judge-dev",
                                      "generate-noise", "judge-noise", "optimize", "freeze", "generate-test", "judge-test", "second-judge",
                                      "baselines", "controls", "latency-seq", "latency-cpu", "latency-load", "rating-export-test", "cleanup-models", "analyze"])
    ap.add_argument("--limit-configs", type=int, help="Pilot: only this many random grid configurations.")
    ap.add_argument("--limit-questions", type=int, help="Pilot: only the first questions.")
    a = ap.parse_args()
    p = protocol()
    grid = grid_configs(p)
    if a.limit_configs:
        grid = random.Random(0).sample(grid, a.limit_configs)
    splits = [s for s in ("test_a", "test_b") if len(questions(s))] if a.stage not in ("prepare",) else []

    if a.stage == "prepare":
        prepare.main()
    elif a.stage == "status":
        status()
    elif a.stage == "generate-dev":
        run.generate(grid, "dev", limit_q=a.limit_questions)
        if not a.limit_configs:
            run.generate(run.archived_arms(p), "dev")
    elif a.stage == "calibration-export":
        with db() as con:
            ans = pd.read_sql("SELECT cfg, qid, answer FROM answers WHERE seed=42", con)
        print(A.export_rating_sheet(ans[ans.cfg.isin({cfg_id(c) for c in grid})], "calibration", p["judge_gate"]["n_answers"], p["judge_gate"]["sample_seed"]), "answers exported")
    elif a.stage == "judge-calibration":
        key = pd.read_csv(RESULTS / "human_calibration_KEY.csv", dtype={"qid": str}, keep_default_na=False)
        for j in (p["judge_primary"], p["judge_secondary"]):
            run.judge(j, "dev", keys=set(zip(key.qid, key.answer)))
    elif a.stage == "judge-gate":
        if (RESULTS / "judge_decision.json").exists():
            return
        res = A.rating_results("calibration", [p["judge_primary"], p["judge_secondary"]])
        choice = A.gate_decision(res, p["judge_primary"], p["judge_secondary"], p["judge_gate"])
        print(json.dumps(res, indent=1))
        if choice is None:
            raise SystemExit("Neither judge reaches min_spearman. Stop and decide with a human.")
        (RESULTS / "judge_decision.json").write_text(json.dumps({"primary": choice, "evidence": res}, indent=1))
        print("primary judge:", choice)
    elif a.stage == "judge-dev":
        run.judge(primary_judge(p), "dev")
    elif a.stage == "generate-noise":
        table = grid_table(primary_judge(p), p)
        for seed in p["noise"]["seeds"]:
            run.generate(run.noise_configs(p, table), "dev", seed=seed)
    elif a.stage == "judge-noise":
        run.judge(primary_judge(p), "dev")
    elif a.stage == "optimize":
        if (RESULTS / "frozen_finalists.json").exists():  # A restarted service must not rewrite what the freeze used.
            return
        primary = primary_judge(p)
        df = grid_table(primary, p)
        assert len(df) == len(grid) and df.n_scored.ge(45).all(), "Table incomplete: finish generate-dev and judge-dev first."
        ans = scored_answers(primary)
        ans = ans[ans.cfg.isin(set(df.cfg))].dropna(subset=["f1"])
        replay(df, p, ans).to_csv(RESULTS / "optimizer_runs.csv", index=False)
        replay_lomo(ans, p).to_csv(RESULTS / "lomo.csv", index=False)
        replay_lomo(ans, p, "FDM").to_csv(RESULTS / "lomo_fdm.csv", index=False)
    elif a.stage == "freeze":
        # A restarted service passes here again; the frozen set must stay as it is.
        if (RESULTS / "frozen_finalists.json").exists():
            print("already frozen")
        else:
            print(len(freeze(grid_table(primary_judge(p), p), pd.read_csv(RESULTS / "optimizer_runs.csv"), p)["configs"]), "configurations frozen")
    elif a.stage == "generate-test":
        for split in splits:
            run.generate(run.frozen_configs(), split)
    elif a.stage == "judge-test":
        for split in splits:
            run.judge(primary_judge(p), split)
    elif a.stage == "second-judge":
        run.judge(other_judge(p), "dev", cfgs=check_cfgs(p))
        for split in splits:
            run.judge(other_judge(p), split)
    elif a.stage == "baselines":
        # Retrieval removed, and perfect retrieval: brackets what retrieval can contribute per model.
        any_cfg = {"chunk_size": p["grid"]["chunk_size"][0], "chunk_overlap": p["grid"]["chunk_overlap"][0], "top_k": 1, "temperature": 0.0}
        for kind in ("closed_book", "oracle"):
            run.generate([{**any_cfg, "model": m} for m in p["grid"]["model"]], "dev", prompt_kind=kind)
        run.judge(primary_judge(p), "dev")
    elif a.stage == "controls":
        run.insert_controls()
        run.judge(primary_judge(p), "dev", cfgs={"control|reference", "control|wrong_question", "control|refusal"})
        run.judge(f"{primary_judge(p)}@{p['judge_controls']['retest_seed']}", "dev", cfgs=retest_cfgs(p))
    elif a.stage == "latency-seq":
        L.sequential(run.frozen_configs())
    elif a.stage == "latency-cpu":
        L.cpu_only([parse_cfg(c) for c in load_frozen()["pareto"]])
    elif a.stage == "latency-load":
        L.concurrency(L.operating_points(load_frozen(), p["latency"]["n_load_points"]))
    elif a.stage == "rating-export-test":
        with db() as con:
            ans = pd.read_sql("SELECT cfg, qid, answer FROM answers WHERE seed=42", con)
        q = set(questions("test_b").qid)
        h = p["human_rating_test"]
        print(A.export_rating_sheet(ans[ans.qid.isin(q) & ans.cfg.isin(load_frozen()["pareto"])], "test", h["n_answers"], h["sample_seed"]), "answers exported")
        # Optional sheet for a real person: the only evidence the manuscript may call human review.
        print(A.export_rating_sheet(ans[ans.cfg.isin(load_frozen()["pareto"])], "person", h["n_person"], h["sample_seed"] + 1), "answers exported for a person")
    elif a.stage == "cleanup-models":
        # Every LLM stage is finished and the digests are in the models table. Only this project's server is addressed.
        assert (RESULTS / "report.json").exists(), "Analysis has not run yet."
        for m in {p["embedding_model"], p["judge_primary"], p["judge_secondary"], *p["grid"]["model"]}:
            subprocess.run([str(ROOT / "runtime/bin/ollama"), "rm", m], env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"}, check=False)
    elif a.stage == "analyze":
        analyze(p)


if __name__ == "__main__":
    main()
