"""Runs with: RAGBENCH_FAKE=1 .venv/bin/python -m pytest -q tests  (no GPU, no Ollama)."""
import os
import random
import sqlite3

os.environ["RAGBENCH_FAKE"] = "1"

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ragbench import search as S
from ragbench.common import FIELDS, cfg_id, grid_configs, parse_cfg, protocol


@pytest.fixture
def table():
    """Known landscape: one unique best configuration, fast models score lower."""
    p = protocol()
    cfgs = grid_configs(p)
    rng = np.random.default_rng(0)
    rows = []
    for c in cfgs:
        m = p["grid"]["model"].index(c["model"])
        f1 = 0.02 * (7 - m) + 0.01 * c["top_k"] + 0.0001 * c["chunk_size"] + rng.normal(0, 0.005)
        rows.append({"cfg": cfg_id(c), "f1": f1, "latency_s": 1 + (7 - m) + 0.1 * c["top_k"] + rng.normal(0, 0.01)})
    return pd.DataFrame(rows), p


def test_grid_size_and_id_roundtrip():
    cfgs = grid_configs()
    assert len(cfgs) == 960
    assert all(parse_cfg(cfg_id(c)) == c for c in cfgs[::97])


def test_mutation_stays_in_range_and_model_can_jump():
    levels = [list(protocol()["grid"][f]) for f in FIELDS]
    rng = random.Random(1)
    jumps = set()
    for _ in range(3000):
        ind = [0, 1, 0, 1, 1]
        out = S.mutate(list(ind), rng, levels, 1.0)
        assert all(0 <= g < len(v) for g, v in zip(out, levels))
        assert all(abs(out[i] - ind[i]) == 1 for i in range(1, 5))  # Ordered parameters only step.
        jumps.add(out[0])
    assert len(jumps) == len(levels[0]) - 1  # The model reaches every other value.


@pytest.mark.parametrize("name", list(S.SINGLE) + ["nsga2", "mo_tpe"])
def test_optimizer_respects_budget_and_is_reproducible(table, name):
    df, p = table
    t = S.Table(df, p["grid"])
    fn = {**S.SINGLE, **S.MULTI}[name]
    results = []
    for _ in range(2):
        t.reset(30)
        with pytest.raises(S.Budget):
            fn(t, random.Random(f"{name}-0"), p["search"]["ga"])
        assert len(t.seen) == 30
        results.append(list(t.seen))
    assert results[0] == results[1]


def test_informed_search_beats_random_on_structured_landscape(table):
    df, p = table
    t = S.Table(df, p["grid"])
    def best(fn, seed):
        t.reset(50)
        try:
            fn(t, random.Random(seed), p["search"]["ga"])
        except S.Budget:
            pass
        return max(v[0] for v in t.seen.values())
    for fn in (S.ga, S.tpe, S.coordinate_ascent):
        assert np.mean([best(fn, s) for s in range(15)]) > np.mean([best(S.random_search, s) for s in range(15)])


def _two_family_table(p):
    """Family A shines on the easy majority of questions and fails the hard rest; B is decent on all.
    A wins on the plain mean, so only a search that attends to hard questions favors B."""
    rows = []
    for i, c in enumerate(grid_configs(p)):
        a = p["grid"]["model"].index(c["model"]) % 2 == 0
        noise = np.random.default_rng(i)
        for q in range(30):
            hard = q < 10
            f1 = (0.0 if hard else 0.8) if a else (0.45 if hard else 0.55)
            rows.append({"cfg": cfg_id(c), "qid": f"q{q}", "f1": float(np.clip(f1 + noise.normal(0, 0.03), 0, 1)), "wall_s": 1.0})
    ans = pd.DataFrame(rows)
    return ans, ans.groupby("cfg").agg(f1=("f1", "mean"), latency_s=("wall_s", "mean")).reset_index()


def test_coevolution_respects_budget_and_is_reproducible():
    p = protocol()
    ans, df = _two_family_table(p)
    t = S.Table(df, p["grid"], ans)
    seen = []
    for _ in range(2):
        t.reset(30)
        with pytest.raises(S.Budget):
            S.coevolution(t, random.Random("x"), p["search"]["ga"])
        assert len(t.seen) == 30
        seen.append((list(t.seen), t.pick()))
    assert seen[0] == seen[1] and seen[0][1] in t.seen


def test_coevolution_favors_configurations_that_survive_hard_questions():
    p = protocol()
    ans, df = _two_family_table(p)
    hard = ans[ans.qid.isin([f"q{i}" for i in range(10)])].groupby("cfg").f1.mean()
    t = S.Table(df, p["grid"], ans)
    def winner_score(fn, seed):
        t.reset(50)
        try:
            fn(t, random.Random(seed), p["search"]["ga"])
        except S.Budget:
            pass
        return hard[t.pick()]
    assert np.mean([winner_score(S.coevolution, s) for s in range(12)]) > np.mean([winner_score(S.ga, s) for s in range(12)]) + 0.1


def test_pareto_and_hypervolume():
    f1, lat = [0.1, 0.5, 0.4, 0.5], [1.0, 3.0, 4.0, 3.0]
    assert S.pareto_mask(f1, lat).tolist() == [True, True, False, True]
    # On a log10 axis: best 0.1 from 1 to 10, best 0.5 from 10 to 100.
    assert S.hypervolume([(0.1, 1.0), (0.5, 10.0), (0.4, 50.0)], 100.0) == pytest.approx(0.1 + 0.5)
    assert S.hypervolume([(0.9, 9.0)], 5.0) == 0


def test_replay_and_freeze_guard(table, tmp_path, monkeypatch):
    df, p = table
    p = {**p, "search": {**p["search"], "n_seeds": 2, "budgets": [20]}}
    monkeypatch.setattr(S, "RESULTS", tmp_path)
    runs = S.replay(df, p)
    single = runs[runs["mode"] == "single"]
    assert (single.regret >= 0).all() and single.n_eval.eq(20).all()
    assert runs[runs["mode"] == "multi"].hv_ratio.between(0, 1).all()
    p["search"]["freeze_budget"] = 20
    frozen = S.freeze(df, runs, p)
    assert frozen["best_dev"] == df.loc[df.f1.idxmax(), "cfg"] and frozen["best_dev"] in frozen["configs"]
    assert set(frozen["configs"]) - set(frozen["archived_arms"]) <= set(df.cfg)
    with pytest.raises(SystemExit):  # A frozen set is never overwritten.
        S.freeze(df, runs, p)


def test_nsga2_finds_better_front_than_random(table):
    df, p = table
    t = S.Table(df, p["grid"])
    ref = df.latency_s.max()
    def hv(fn, seed):
        t.reset(60)
        try:
            fn(t, random.Random(seed), p["search"]["ga"])
        except S.Budget:
            pass
        return S.hypervolume(list(t.seen.values()), ref)
    assert np.mean([hv(S.nsga2, s) for s in range(10)]) > np.mean([hv(S.random_search, s) for s in range(10)])


def test_judge_gate_rule():
    from ragbench.analysis import gate_decision
    g = {"min_spearman": 0.5, "max_gap": 0.1}
    r = lambda d, o: {"D": {"spearman_human": d}, "O": {"spearman_human": o}}
    assert gate_decision(r(0.7, 0.75), "D", "O", g) == "D"
    assert gate_decision(r(0.6, 0.75), "D", "O", g) == "O"
    assert gate_decision(r(0.4, 0.55), "D", "O", g) == "O"
    assert gate_decision(r(0.4, 0.45), "D", "O", g) is None


def test_full_pipeline_on_fake_backend(tmp_path, monkeypatch):
    """Prepare to analysis on a tiny grid: fails if any stage stops fitting together, and proves that
    only frozen configurations ever see a test question."""
    from ragbench import analysis as A, common as C, latency as L, prepare, run

    monkeypatch.setattr(C, "DATA", tmp_path); monkeypatch.setattr(prepare, "DATA", tmp_path); monkeypatch.setattr(S, "DATA", tmp_path)
    monkeypatch.setattr(C, "RESULTS", tmp_path); monkeypatch.setattr(S, "RESULTS", tmp_path); monkeypatch.setattr(A, "RESULTS", tmp_path)
    monkeypatch.setenv("RAGBENCH_DB", str(tmp_path / "t.sqlite"))
    p = protocol()
    p["grid"] = {"model": ["qwen-a", "m-b", "m-c"], "chunk_size": [256, 512], "chunk_overlap": [0, 25], "top_k": [1, 5], "temperature": [0.0, 0.4]}
    p["archived_arms"] = ["m-b|1024|25|10|0.3"]
    p["noise"].update(n_random=3, n_top=3)
    p["search"].update(budgets=[10], n_seeds=3, freeze_budget=10, lomo_seeds=1)
    p["search"]["ga"]["pop_size"] = 4
    p["finalists"].update(n_pareto=3, n_selected=2, n_random=4, n_coevolution=2)
    p["latency"].update(repetitions=1, concurrency_requests=4, concurrency_levels=[1, 2], n_load_points=2)
    p["bootstrap"]["n"] = 50
    for mod in (C, run, A, L):
        monkeypatch.setattr(mod, "protocol", lambda p=p: p)
    monkeypatch.setattr(S, "protocol", lambda p=p: p)

    if (C.ARCHIVE / "data").exists():
        df = prepare.build_questions()
    else:  # Public repository: the raw archive is not part of it, the derived question sets are.
        df = pd.read_csv(Path(__file__).resolve().parents[2] / "data/questions.csv", dtype={"qid": str}, keep_default_na=False)
        df = df[df.split.isin(["dev", "test_a"])]
    assert (df.split == "dev").sum() == 50 and (df.split == "test_a").sum() == 162
    assert prepare.write_sheets(df) >= 8
    pd.concat([df[df.split == "dev"], df[df.split == "test_a"].head(12)]).to_csv(tmp_path / "questions.csv", index=False)

    grid = C.grid_configs(p)
    run.generate(grid, "dev")
    run.generate(run.archived_arms(p), "dev")
    run.generate(grid, "dev")  # Resume: nothing new.
    with C.db() as con:
        assert con.execute("SELECT count(*) FROM answers").fetchone()[0] == (len(grid) + 1) * 50
        ans = pd.read_sql("SELECT cfg, qid, answer FROM answers", con)

    # Calibration happens before the full judging run, and the raters stay blind.
    assert A.export_rating_sheet(ans, "calibration", 20, 1) == 20
    sheet = pd.read_csv(tmp_path / "human_calibration_SHEET_raterA.csv")
    assert not {"cfg", "f1"} & set(sheet.columns)
    key = pd.read_csv(tmp_path / "human_calibration_KEY.csv", dtype={"qid": str})
    run.judge("J1", "dev", keys=set(zip(key.qid, key.answer)))
    with C.db() as con:
        assert con.execute("SELECT count(*) FROM scores").fetchone()[0] <= 20
    col = "rating_0_wrong_1_partial_2_correct"
    run.judge("J2", "dev", keys=set(zip(key.qid, key.answer)))
    for r in "AB":
        sheet.assign(**{col: np.arange(20) % 3}).to_csv(tmp_path / f"human_calibration_DONE_rater{r}.csv", index=False)
    res = A.rating_results("calibration", ["J1", "J2"])
    assert res["kappa_raters"] == pytest.approx(1) and res["J1"]["n"] == 20
    (tmp_path / "human_calibration_DONE_raterB.csv").rename(tmp_path / "b.csv")  # One person alone is enough.
    assert A.rating_results("calibration", ["J1"])["kappa_raters"] is None
    (tmp_path / "b.csv").rename(tmp_path / "human_calibration_DONE_raterB.csv")

    run.judge("J1", "dev")
    t = C.grid_table("J1", p)
    assert len(t) == len(grid) and t.n_scored.eq(50).all() and t.f1.between(0, 1).all()
    assert (t.latency_s >= t.gen_s).all()
    noisy = run.noise_configs(p, t)
    assert all(c["temperature"] > 0 for c in noisy)
    for s in p["noise"]["seeds"]:
        run.generate(noisy, "dev", seed=s)
    run.judge("J1", "dev")
    assert A.noise_estimate("J1", [42] + p["noise"]["seeds"])["n_configs"] == len(noisy)

    run.insert_controls()
    run.judge("J1", "dev", cfgs={"control|reference", "control|wrong_question", "control|refusal"})
    retest = set(t.cfg[:2])
    run.judge("J1@43", "dev", cfgs=retest)
    ctl = A.judge_controls("J1", "J1@43", retest)
    assert len(ctl["controls"]) == 3 and ctl["retest_n"] == 100
    assert len(C.grid_table("J1", p)) == len(grid)  # Controls stay out of the table.

    answers_grid = C.scored_answers("J1")
    runs = S.replay(t, p, answers_grid[answers_grid.cfg.isin(set(t.cfg))].dropna(subset=["f1"]))
    runs.to_csv(tmp_path / "optimizer_runs.csv", index=False)
    assert "coevolution" in set(runs.method)
    assert not A.optimizer_stats(runs).empty
    assert "C(model)" in A.factorial_anova(t).index
    effects = A.answer_level_effects("J1")
    assert "C(model)" in effects.index
    assert effects.loc["C(chunk_size)", "df_constraint"] == 1  # The archived arm's chunk size is not a level.
    a = C.scored_answers("J1")
    a = a[a.cfg.isin(set(t.cfg))].dropna(subset=["f1"])
    lomo = S.replay_lomo(a, p)
    assert lomo.manual.nunique() == 20 and set(lomo.method) == set(S.SINGLE) | {"coevolution"}
    stats = A.lomo_stats(lomo, p)
    assert stats.set_index("method").diff_vs_random["random"] == 0 and not stats.beats_random[stats.method == "random"].any()

    frozen = S.freeze(t, runs, p)
    assert set(frozen["coevolution"]) <= set(frozen["configs"]) and frozen["coevolution"]
    assert frozen["data_sha256"] and frozen["code_sha256"] and "m-b|1024|25|10|0.3" in frozen["configs"]
    run.generate(run.frozen_configs(), "test_a")
    run.judge("J1", "test_a")
    g, summary = A.generalization("J1", frozen)
    assert set(g[g.view == "all"].cfg) == set(frozen["configs"])
    assert "selection_optimism" in summary["test_a/all"] and "test_a/short" in summary
    with C.db() as con:  # Only frozen configurations ever saw a test question.
        test_q = set(C.questions("test_a").qid)
        seen = {c for c, q in con.execute("SELECT cfg, qid FROM answers") if q in test_q}
    assert seen == set(frozen["configs"])

    check = list(t.cfg[:5])
    run.judge("J2", "dev", cfgs=set(check))
    assert A.judge_check("J1", "J2", check)["n_configs"] == 5
    assert A.failures("J1").failed.sum() == 0

    run.generate([{**grid[0], "model": m} for m in p["grid"]["model"]], "dev", prompt_kind="oracle")
    assert C.grid_table("J1", p).cfg.str.count(r"\|").eq(4).all()

    L.sequential(run.frozen_configs()[:1]); L.cpu_only(run.frozen_configs()[:1], 2)
    L.concurrency(L.operating_points(frozen, 2))
    with C.db() as con:
        before = con.execute("SELECT count(*) FROM measurements").fetchone()[0]
    L.sequential(run.frozen_configs()[:1]); L.cpu_only(run.frozen_configs()[:1], 2)  # A restart repeats nothing.
    L.concurrency(L.operating_points(frozen, 2))
    with C.db() as con:
        assert con.execute("SELECT count(*) FROM measurements").fetchone()[0] == before
    with C.db() as con:
        assert {r[0] for r in con.execute("SELECT DISTINCT kind FROM measurements")} == {"gpu_seq_s", "cpu_seq_s", "concurrent_s"}

    from ragbench import dashboard as D
    monkeypatch.setattr(D, "RESULTS", tmp_path); monkeypatch.setattr(D, "DATA", tmp_path); monkeypatch.setattr(D, "protocol", lambda p=p: p)
    monkeypatch.setattr(D, "check_cfgs", lambda p: set(check)); monkeypatch.setattr(D, "ollama", lambda path: None)
    monkeypatch.setattr(D, "running_stages", lambda: [])
    p["judge_primary"], p["judge_secondary"] = "J1", "J2"
    state = D.state()
    by = {x["key"]: x for x in state["stages"]}
    assert by["generate-dev"]["complete"] and by["freeze"]["complete"] and by["rating"]["complete"] and not by["analyze"]["complete"]
    assert by["judge-test"]["complete"] and sum(r["front"] for r in state["table"]) >= 1
    assert len(state["table"]) == len(grid) and state["paused"] is None and len(state["human"]) == 5
    assert state["remaining_s"] >= 0 and state["finish_ts"] >= state["time"]  # Forecast exists once timings exist.

    A.fig_regret(runs); A.fig_pareto(t); A.fig_heatmaps(t); A.fig_generalization(g)
    A.write_macros({"BestF": 0.4567, "Name": "x"})
    assert "\\newcommand{\\rrBestF}{0.4567}" in (tmp_path / "results.tex").read_text()


def _tiny_db(tmp_path, monkeypatch):
    from ragbench import common as C, run
    monkeypatch.setattr(C, "DATA", tmp_path); monkeypatch.setattr(C, "RESULTS", tmp_path)
    monkeypatch.setenv("RAGBENCH_DB", str(tmp_path / "t.sqlite"))
    pd.DataFrame({"qid": ["q1", "q2"], "split": "dev", "question": ["a", "b"], "reference": ["r1", "r2"], "reference_contexts": "c",
                  "source_file": ["m1", "m2"], "process": "FDM", "query_style": "x", "query_length": "SHORT", "persona_name": "p",
                  "pointer_ref": False, "toc_ref": False}).to_csv(tmp_path / "questions.csv", index=False)
    with C.db() as con:
        con.execute("INSERT INTO answers(cfg, qid, seed, answer) VALUES('c','q1',42,'x'),('c','q2',42,'y')")
    return C, run


def test_failed_judgment_stops_the_stage_and_is_retried(tmp_path, monkeypatch):
    from ragbench import backend as B
    C, run = _tiny_db(tmp_path, monkeypatch)
    p = protocol(); p["judge"]["concurrency"] = 2
    monkeypatch.setattr(run, "protocol", lambda: p)
    async def boom(answer, reference):
        raise OSError("ollama down")
    async def ok(answer, reference):
        return 0.5
    monkeypatch.setattr(B, "make_scorer", lambda j, c: boom)
    monkeypatch.setattr(run.B, "make_scorer", lambda j, c: boom)
    run.judge("J", "dev")  # 2 failures stay under the floor of 5, so the stage ends...
    with C.db() as con:
        assert con.execute("SELECT count(*) FROM scores WHERE f1 IS NULL").fetchone()[0] == 2
    monkeypatch.setattr(run.B, "make_scorer", lambda j, c: ok)
    run.judge("J", "dev")  # ...and the next pass retries them.
    with C.db() as con:
        assert con.execute("SELECT count(*) FROM scores WHERE f1=0.5").fetchone()[0] == 2


def test_judge_gate_rejects_nan():
    from ragbench.analysis import gate_decision
    g = {"min_spearman": 0.5, "max_gap": 0.1}
    nan = float("nan")
    assert gate_decision({"D": {"spearman_human": nan}, "O": {"spearman_human": nan}}, "D", "O", g) is None
    assert gate_decision({"D": {"spearman_human": nan}, "O": {"spearman_human": .9}}, "D", "O", g) == "O"


def test_lomo_survives_a_missing_held_out_score(table):
    from ragbench.analysis import lomo_stats
    df, p = table
    p = {**p, "search": {**p["search"], "lomo_seeds": 1, "freeze_budget": 10}}
    a = pd.concat([df.assign(qid=f"q{i}", source_file=f"m{i}", process="FDM", wall_s=df.latency_s) for i in range(3)])
    a = a[~((a.source_file == "m0") & (a.f1 > df.f1.quantile(.5)))]  # Winners have no score on this manual.
    lomo = S.replay_lomo(a, p)
    assert lomo.held_f1.isna().any()
    assert lomo_stats(lomo, p).held_f1.notna().all()


def test_change_after_freeze_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RESULTS", tmp_path); monkeypatch.setattr(S, "DATA", tmp_path)
    (tmp_path / "questions.csv").write_text("a")
    (tmp_path / "frozen_finalists.json").write_text(__import__("json").dumps(
        {"configs": [], "data_sha256": S._data_sha(), "protocol_sha256": __import__("hashlib").sha256(S.PROTOCOL_PATH.read_bytes()).hexdigest()}))
    assert S.load_frozen()["configs"] == []
    (tmp_path / "test_b_human.csv").write_text("changed")
    with pytest.raises(SystemExit):
        S.load_frozen()


def test_dashboard_sees_a_running_stage():
    import subprocess
    import time
    from ragbench import dashboard as D
    job = subprocess.Popen(["bash", "-c", "exec -a 'python -m ragbench judge-dev' sleep 30"])
    try:
        time.sleep(0.3)
        assert "judge-dev" in D.running_stages()
    finally:
        job.kill()


def test_dashboard_sends_strict_json_even_with_nan():
    import json
    from ragbench import dashboard as D
    state = {"a": float("nan"), "b": [np.float64("inf"), np.int64(3), {"c": np.float64(0.5), "d": np.bool_(True)}]}
    out = json.loads(json.dumps(D.finite(state), allow_nan=False))  # Browsers reject NaN in JSON.
    assert out == {"a": None, "b": [None, 3, {"c": 0.5, "d": True}]}


def test_main_effect_is_tested_over_all_models_not_the_reference():
    from ragbench.analysis import effects_from
    rng = np.random.default_rng(0)
    rows = []
    for q in range(30):
        for m in ("a", "b"):
            for k in (1, 10):
                for cs in (256, 512):
                    gain = 0.4 if (m == "b" and k == 10) else 0.0  # Top-k helps only model b.
                    rows.append({"qid": f"q{q}", "model": m, "top_k": k, "chunk_size": cs, "chunk_overlap": 0,
                                 "temperature": 0.0, "f1": 0.3 + gain + 0.02 * (q % 5) + rng.normal(0, 0.05)})
    w = effects_from(pd.DataFrame(rows))
    assert w.loc["C(top_k)", "pvalue"] < 0.001  # Averaged over models, top-k matters.
    assert w.loc["C(model):C(top_k)", "pvalue"] < 0.001


def _figure_check():
    """The check() cell of plots/figures.ipynb together with its setup constants."""
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path
    import nbformat
    nb = nbformat.read(Path(__file__).resolve().parents[2] / "plots/figures.ipynb", as_version=4)
    src = [c.source for c in nb.cells if c.cell_type == "code" and "def check(fig)" in c.source][0]
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.text import Text
    ns = {"plt": plt, "np": np, "matplotlib": matplotlib, "Text": Text, "FONT": 9}
    exec(src.split("def save(")[0], ns)
    return ns["check"], plt


def test_figure_check_rejects_overlap_size_and_covered_points():
    check, plt = _figure_check()
    plt.rcParams.update({k: 9 for k in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fig, ax = plt.subplots(figsize=(3, 2), layout="constrained")
    ax.scatter([0.5], [0.5])
    check(fig)                                    # a clean figure passes
    ax.text(0.2, 0.8, "label", fontsize=9, transform=ax.transAxes)
    ax.text(0.21, 0.8, "label", fontsize=9, transform=ax.transAxes)
    with pytest.raises(AssertionError, match="overlap"):
        check(fig)
    fig, ax = plt.subplots(figsize=(3, 2), layout="constrained")
    ax.set_xlabel("x", fontsize=10)
    with pytest.raises(AssertionError, match="font sizes"):
        check(fig)
    fig, ax = plt.subplots(figsize=(3, 2), layout="constrained")
    ax.tick_params(labelsize=9)
    ax.scatter([0.5], [0.5])
    ax.text(0.5, 0.5, "on the point", fontsize=9, ha="center", va="center")
    with pytest.raises(AssertionError, match="covers"):
        check(fig)
    plt.close("all")
