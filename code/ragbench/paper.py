"""Every number and table the manuscript uses, computed from the raw results: python -m ragbench.paper.
Writes results/results.tex (macros) and results/tables/tab_*.tex. Nothing in the paper is typed by hand."""
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from .common import RESULTS, config_table, protocol, questions, scored_answers
from .search import load_frozen, pareto_mask

MODEL = {"qwen3-vl:30b-a3b-instruct": "Qwen3-VL 30B-A3B", "gemma3n:e2b": "Gemma 3n E2B", "granite4:tiny-h": "Granite 4 Tiny-H",
         "phi4-mini:3.8b-q4_K_M": "Phi-4-mini 3.8B", "qwen3-vl:2b-instruct": "Qwen3-VL 2B", "granite3.3:2b": "Granite 3.3 2B",
         "gemma3:1b": "Gemma 3 1B", "qwen2.5-coder:0.5b-instruct": "Qwen2.5-Coder 0.5B"}
METHOD = {"random": "Random search", "ga": "Genetic algorithm", "tpe": "TPE", "coordinate_ascent": "Coordinate ascent",
          "coevolution": "Coevolutionary GA", "nsga2": "NSGA-II", "mo_tpe": "MO-TPE"}


def f(x, d=3):
    return f"{x:.{d}f}"


def sci(x):
    """p-values for use inside math mode: 3.4\\times10^{-5}."""
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def cfg_tex(cid):
    m, cs, ov, k, t = cid.rsplit("|", 4)
    return f"{MODEL.get(m, m)}, {cs}/{ov}\\,\\%/k={k}/T={t}"


def latency_checks(con):
    """Ollama serves a prompt it has just processed from its cache. The repeated passes of the latency stage
    and the CPU runs right after an identical prompt set are therefore not timing measurements and are left out."""
    me = pd.read_sql("SELECT rowid AS r, kind, cfg, qid, rep, level, value FROM measurements", con)
    first_q = me[me.kind == "gpu_seq_s"].sort_values("r").groupby("cfg").qid.first()
    seq = me[(me.kind == "gpu_seq_s") & (me.rep == 0)]
    seq = seq[seq.qid != seq.cfg.map(first_q)]  # The warm-up request used the same prompt.
    cpu = me[me.kind == "cpu_seq_s"].sort_values("r")
    order = cpu.groupby("cfg").r.min().sort_values().index.tolist()
    same_prompts_before = {b for a, b in zip(order, order[1:]) if a.rsplit("|", 1)[0] == b.rsplit("|", 1)[0]}
    cpu = cpu[~cpu.cfg.isin(same_prompts_before)]
    cpu = cpu[cpu.groupby("cfg").cumcount() > 0]
    return seq.groupby("cfg").value.median(), cpu.groupby("cfg").value.median(), sorted(same_prompts_before)


def main():
    p = protocol()
    J = json.loads((RESULTS / "judge_decision.json").read_text())["primary"]
    rep = json.loads((RESULTS / "report.json").read_text())
    frozen = load_frozen()
    con = sqlite3.connect(RESULTS / "ragbench.sqlite")
    t = pd.read_csv(RESULTS / "benchmark_table.csv")
    allc = config_table(J)
    front = t[pareto_mask(t.f1, t.latency_s)].sort_values("latency_s")
    best = t.loc[t.f1.idxmax()]
    an = pd.read_csv(RESULTS / "tables/anova_config_level.csv", index_col=0)
    eff = pd.read_csv(RESULTS / "tables/effects_answer_level.csv", index_col=0)
    runs = pd.read_csv(RESULTS / "optimizer_runs.csv")
    ostats = pd.read_csv(RESULTS / "tables/optimizers_conditional_on_table.csv")
    lomo = {k: pd.DataFrame(rep[f"lomo_{k}"]).set_index("method") for k in ("all", "fdm")}
    gen = rep["generalization"]
    s = scored_answers(J)
    base = s[s.cfg.str.startswith(("oracle|", "closed_book|"))].groupby("cfg").f1.mean()
    mdb = pd.read_sql("SELECT model, size_vram, size_bytes FROM models", con).set_index("model")
    models, disk = mdb.size_vram / 1e9, mdb.size_bytes / 1e9
    seq, cpu, cpu_dropped = latency_checks(con)
    scores = pd.read_sql("SELECT judge, count(*) AS n, sum(f1 IS NULL) AS failed FROM scores GROUP BY judge", con).set_index("judge")
    arms = allc.set_index("cfg").loc[p["archived_arms"]]
    knee = front.loc[(front.f1 / best.f1 >= 0.8)].iloc[0]  # Fastest front point within 80 % of the best F1.
    gemma = t[t.model == "gemma3n:e2b"].loc[lambda d: d.f1.idxmax()]
    lc = pd.read_csv(RESULTS / "tables/lexical_coverage.csv")
    rt = {r: pd.read_csv(RESULTS / f"human_test_DONE_rater{r}.csv") for r in "AB"}
    q = questions()

    v = {
        "GridN": len(t), "DevQ": int((q.split == "dev").sum()), "TestAQ": int((q.split == "test_a").sum()), "TestBQ": int((q.split == "test_b").sum()),
        "ModelsN": t.model.nunique(), "AnswersGrid": f"{len(t) * 50:,}".replace(",", "{,}"),
        "AnswersAll": f"{con.execute('SELECT count(*) FROM answers').fetchone()[0]:,}".replace(",", "{,}"),
        "GenHours": f(con.execute("SELECT sum(wall_s) / 3600 FROM answers").fetchone()[0], 1), "Judgments": f"{int(scores.n[J]):,}".replace(",", "{,}"), "JudgeFailed": int(scores.failed[J]),
        "FrozenN": len(frozen["configs"]), "BestF": f(best.f1), "BestLat": f(best.latency_s, 2), "BestCfg": cfg_tex(best.cfg),
        "MedianF": f(t.f1.median()), "MinF": f(t.f1.min()), "FrontN": len(front),
        "FastF": f(front.iloc[0].f1), "FastLat": f(front.iloc[0].latency_s, 3), "FastCfg": cfg_tex(front.iloc[0].cfg),
        "KneeF": f(knee.f1), "KneeLat": f(knee.latency_s, 3), "KneeCfg": cfg_tex(knee.cfg), "KneeShare": f(100 * knee.f1 / best.f1, 0),
        "KneeSpeed": f(best.latency_s / knee.latency_s, 1),
        "GemmaF": f(gemma.f1), "GemmaLat": f(gemma.latency_s, 2), "GemmaShare": f(100 * gemma.f1 / best.f1, 0),
        "RetrMs": f(t.retrieval_ms.median(), 1), "LatMin": f(t.latency_s.min(), 3), "LatMax": f(t.latency_s.max(), 2),
        "VramMin": f(models.min(), 2), "VramMax": f(models.max(), 1),
        "EtaModel": f(an.loc["C(model)", "partial_eta2"], 2), "EtaChunk": f(an.loc["C(cs)", "partial_eta2"], 2), "EtaK": f(an.loc["C(k)", "partial_eta2"], 2),
        "EtaModelK": f(an.loc["C(model):C(k)", "partial_eta2"], 2), "EtaOv": f(an.loc["C(ov)", "partial_eta2"], 2), "EtaTemp": f(an.loc["C(temp)", "partial_eta2"], 3),
        "PModel": sci(eff.loc['C(model)', 'pvalue']), "PK": sci(eff.loc['C(top_k)', 'pvalue']), "PChunk": f(eff.loc["C(chunk_size)", "pvalue"]),
        "POv": f(eff.loc["C(chunk_overlap)", "pvalue"], 2), "PTemp": f(eff.loc["C(temperature)", "pvalue"], 2), "PModelK": sci(eff.loc['C(model):C(top_k)', 'pvalue']),
        "KOneF": f(t[t.top_k == 1].f1.mean()), "KTenF": f(t[t.top_k == 10].f1.mean()),
        "CovKOne": f(lc[lc.top_k == 1].coverage.mean(), 2), "CovKTen": f(lc[lc.top_k == 10].coverage.mean(), 2),
        "NoiseWithin": f(rep["noise"]["within_config_sd"]), "NoiseBetween": f(rep["noise"]["between_config_sd"]), "NoiseRho": f(rep["noise"]["spearman_first_two"], 2),
        "NoiseN": rep["noise"]["n_configs"],
        "CtlRef": f(rep["judge_controls"]["controls"]["control|reference"], 2), "CtlRefusal": f(rep["judge_controls"]["controls"]["control|refusal"], 2),
        "CtlWrong": f(rep["judge_controls"]["controls"]["control|wrong_question"], 2), "RetestRho": f(rep["judge_controls"]["retest_spearman"], 2),
        "RetestMad": f(rep["judge_controls"]["retest_mean_abs_diff"], 3), "JudgeRhoCfg": f(rep["judge_check"]["spearman_config"], 2),
        "JudgeRhoAns": f(rep["judge_check"]["spearman_answer"], 2),
        "GapMin": f(min(rep["judge_check"]["gap_secondary_minus_primary_per_model"].values())),
        "GapMax": f(max(rep["judge_check"]["gap_secondary_minus_primary_per_model"].values())),
        "CalKappa": f(rep["llm_rater_calibration"]["kappa_raters"], 2), "CalRhoP": f(rep["llm_rater_calibration"][J]["spearman_human"], 2),
        "CalRhoS": f(rep["llm_rater_calibration"][p["judge_secondary"]]["spearman_human"], 2),
        "TestKappa": f(rep["rating_test"]["kappa_raters"], 2), "TestRhoP": f(rep["rating_test"][J]["spearman_human"], 2),
        "TestRhoS": f(rep["rating_test"][p["judge_secondary"]]["spearman_human"], 2),
        "TestRatedCorrect": int((rt["A"].rating_0_wrong_1_partial_2_correct == 2).sum()), "TestRatedWrong": int((rt["A"].rating_0_wrong_1_partial_2_correct == 0).sum()),
        "TestRatedPartial": int((rt["A"].rating_0_wrong_1_partial_2_correct == 1).sum()), "TestRatedN": len(rt["A"]),
        "TestRatedHarm": int(max(rt[r].harmful_if_followed_0_1.sum() for r in "AB")), "TestRatedRefusal": int(rt["A"].refusal_0_1.sum()),
        "ClosedMin": f(base[base.index.str.startswith("closed")].min()), "ClosedMax": f(base[base.index.str.startswith("closed")].max()),
        "OracleBig": f(base["oracle|qwen3-vl:30b-a3b-instruct"]), "OracleMin": f(base[base.index.str.startswith("oracle")].min()),
        "ArmBestF": f(arms.f1.iloc[0]), "ArmBestLat": f(arms.latency_s.iloc[0], 2), "ArmSmallF": f(arms.f1.iloc[1]), "ArmSmallLat": f(arms.latency_s.iloc[1], 2),
        "LatReproRho": f(pd.concat([seq.rename("re"), t.set_index("cfg").gen_s], axis=1).dropna().corr(method="spearman").iloc[0, 1], 2),
        "LatReproN": len(seq), "CpuN": len(cpu), "CpuBest": f(cpu.get(best.cfg, np.nan), 1), "CpuMin": f(cpu.min(), 2), "CpuMax": f(cpu.max(), 1),
        "CpuDropped": len(cpu_dropped),
    }
    for split, key in (("A", "test_a/all"), ("ANoToc", "test_a/no_toc_refs"), ("AShort", "test_a/short"), ("B", "test_b/all"), ("AFdm", "test_a/FDM"), ("AMsla", "test_a/MSLA")):
        g = gen[key]
        v |= {f"Gen{split}Test": f(g["best_dev_on_test"]), f"Gen{split}Rank": g["test_rank_of_best"], f"Gen{split}Opt": f(g["selection_optimism"]),
              f"Gen{split}OverRandom": f(g["best_minus_random_on_test"]), f"Gen{split}Rho": f(g["spearman_random_stratum"], 2), f"Gen{split}Q": g["n_questions"]}
    lo, lf = lomo["all"], lomo["fdm"]
    v |= {"LomoRandom": f(lo.held_f1["random"]), "LomoGa": f(lo.held_f1["ga"]), "LomoCo": f(lo.held_f1["coevolution"]), "LomoTpe": f(lo.held_f1["tpe"]),
          "LomoCa": f(lo.held_f1["coordinate_ascent"]), "LomoCoGa": f(lo.diff_vs_ga["coevolution"]), "LomoCoGaLo": f(lo.ga_ci_lo["coevolution"]),
          "LomoCoGaHi": f(lo.ga_ci_hi["coevolution"]), "LomoGaLo": f(lo.ci_lo["ga"]), "LomoGaHi": f(lo.ci_hi["ga"]),
          "LomoTpeFdmLo": f(lf.ci_lo["tpe"]), "LomoTpeFdmHi": f(lf.ci_hi["tpe"]), "MinEffect": f(p["search"]["min_effect_f1"], 2)}
    hv = ostats[(ostats["mode"] == "multi") & (ostats.method == "nsga2")].set_index("budget")
    v |= {"HvNsgaOneFifty": f(hv.loc[150, "median"]), "HvNsgaOneFiftyP": sci(hv.loc[150, 'p_holm']), "HvNsgaDelta": f(hv.loc[150, "cliffs_delta_vs_random"], 2),
          "HvRandOneFifty": f(ostats[(ostats["mode"] == "multi") & (ostats.method == "random") & (ostats.budget == 150)]["median"].iloc[0]),
          "HvNsgaHundredP": f(hv.loc[100, "p_holm"], 4), "Seeds": p["search"]["n_seeds"]}
    o = ostats.set_index(["mode", "method", "budget"])
    gains = t.pivot_table(index="model", columns="top_k", values="f1")
    gains = (gains[10] - gains[1]).sort_values()
    v |= {"MoTpePHundred": f(o.loc[("multi", "mo_tpe", 100), "p_holm"], 2), "MoTpePOneFifty": f(o.loc[("multi", "mo_tpe", 150), "p_holm"], 2),
          "HvNsgaFiftyP": f(o.loc[("multi", "nsga2", 50), "p_holm"], 2), "CoevoRandP": f(o.loc[("single", "coevolution", 150), "p_holm"], 3),
          "CoevoRandDelta": f(o.loc[("single", "coevolution", 150), "cliffs_delta_vs_random"], 2),
          "GaRandPTwentyFive": f(o.loc[("single", "ga", 25), "p_holm"], 2), "GaRandPFifty": sci(o.loc[('single', 'ga', 50), 'p_holm']),
          "OvGain": f(t[t.chunk_overlap == 25].f1.mean() - t[t.chunk_overlap == 0].f1.mean()),
          "TempRange": f(t.groupby("temperature").f1.mean().max() - t.groupby("temperature").f1.mean().min()),
          "GainMax": f(gains.iloc[-1]), "GainMaxModel": MODEL[gains.index[-1]], "GainSecond": f(gains.iloc[-2]), "GainSecondModel": MODEL[gains.index[-2]],
          "GainMin": f(gains.iloc[0]), "GainMinModel": MODEL[gains.index[0]],
          "GemmaDisk": f(disk["gemma3n:e2b"], 1), "VramBig": f(models["qwen3-vl:30b-a3b-instruct"], 1)}
    # Composition of the frozen set: the groups overlap, so count each configuration once.
    seen, comp = set(), {}
    for key in ("pareto", "selected", "coevolution", "random", "archived_arms"):
        new = [c for c in frozen[key] if c not in seen]
        comp[key] = len(new)
        seen |= set(new)
    v |= {"FrzPareto": comp["pareto"], "FrzSelected": comp["selected"], "FrzCoevo": comp["coevolution"], "FrzRandom": comp["random"], "FrzArms": comp["archived_arms"]}
    # Uncertainty of the rank of the best dev configuration: resample the test questions.
    rng = np.random.default_rng(p["bootstrap"]["seed"])
    for split, tag in (("test_a", "A"), ("test_b", "B")):
        w = scored_answers(J, split, cfgs=set(frozen["configs"])).pivot_table(index="qid", columns="cfg", values="f1")
        ranks = []
        for _ in range(2000):
            m = w.iloc[rng.integers(len(w), size=len(w))].mean()
            ranks.append(int((m > m[frozen["best_dev"]]).sum()) + 1)
        v |= {f"Gen{tag}RankLo": int(np.percentile(ranks, 2.5)), f"Gen{tag}RankHi": int(np.percentile(ranks, 97.5)),
              f"Gen{tag}TopShare": f(100 * np.mean(np.array(ranks) <= 5), 0)}
        if split == "test_b":
            flags = pd.read_csv(RESULTS.parent / "data/test_b_comments.csv")
            tb = questions("test_b").reset_index(drop=True)
            safe = set(tb.qid[flags.comment.astype(str).str.startswith("SAFETY").values])
            ws = w.loc[w.index.isin(safe)]
            v |= {"SafetyN": len(safe), "SafetyBest": f(ws[frozen["best_dev"]].mean()), "SafetyRandom": f(ws[[c for c in frozen["random"]]].mean().mean())}
            fam = w.mean()
            qwen = fam[fam.index.str.startswith("qwen")]
            other = fam[~fam.index.str.startswith("qwen")]
            v |= {"VendorQwenBest": f(qwen.max()), "VendorOtherBest": f(other.max()), "VendorOtherModel": MODEL[other.idxmax().rsplit("|", 4)[0]]}
    v |= {"HvRefLat": f(t.latency_s.max(), 2)}
    # Adam's own review comes last; until then the manuscript shows a visible gap that the final check rejects.
    person = rep.get("rating_person")
    v |= {"PersonN": person["n"] if person else r"\textbf{[PENDING]}",
          "PersonRho": f(person[J]["spearman_human"], 2) if person else r"\textbf{[PENDING]}"}
    if person:
        from .analysis import weighted_kappa
        J2 = p["judge_secondary"] if J == p["judge_primary"] else p["judge_primary"]
        col = "rating_0_wrong_1_partial_2_correct"
        own = pd.read_csv(RESULTS / "human_person_DONE_raterA.csv").set_index("item")[col]
        llm = pd.read_csv(RESULTS.parent / "empfehlungen/2_empfehlungen_llm.csv").set_index("nr")["empfehlung_0_falsch_1_teilweise_2_richtig"].loc[own.index]
        v |= {"PersonScored": person[J]["n"], "PersonRhoSec": f(person[J2]["spearman_human"], 2),
              "PersonLlmAgree": int((own == llm).sum()), "PersonLlmKappa": f(weighted_kappa(own, llm), 2)}
    write_macros(v)
    tables(t, front, runs, lomo, gen, base, models, seq, cpu, disk)
    print(f"{len(v)} macros, CPU configurations dropped for cached prompts: {cpu_dropped}")


def write_macros(v):
    lines = []
    for k, x in v.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\rr{k}}}{{{x}}}")
    (RESULTS / "results.tex").write_text("% Generated by python -m ragbench.paper. Do not edit.\n" + "\n".join(lines) + "\n")


def _tab(name, header, rows, spec):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    (RESULTS / "tables" / f"tab_{name}.tex").write_text(
        f"% Generated by python -m ragbench.paper. Do not edit.\n\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")


def tables(t, front, runs, lomo, gen, base, models, seq, cpu, disk):
    rows = []
    for m, g in t.groupby("model"):
        b = g.loc[g.f1.idxmax()]
        rows.append((MODEL[m], f(disk[m], 1), f(models[m], 1), f(base[f"closed_book|{m}"]), f(b.f1), f(base[f"oracle|{m}"]), f(b.latency_s, 2)))
    rows.sort(key=lambda r: -float(r[4]))
    _tab("models", r"Model & \makecell[r]{Size\\(GB)} & \makecell[r]{GPU mem.\\(GB)} & \makecell[r]{No\\retrieval} & \makecell[r]{Best of 120\\RAG configs} & \makecell[r]{Gold\\context} & \makecell[r]{Latency\\(s)}", rows, "lrrrrrr")
    _tab("front", r"Model & Chunk & Overlap & $k$ & $T$ & F1 & Latency (s)",
         [(MODEL[r.model], r.chunk_size, f"{r.chunk_overlap}\\,\\%", r.top_k, f(r.temperature, 1), f(r.f1), f(r.latency_s, 3)) for r in front.itertuples()], "lrrrrrr")
    single = runs[runs["mode"] == "single"].groupby(["method", "budget"]).regret.median().unstack()
    lo = lomo["all"]
    rows = [(METHOD[m], *[f(single.loc[m, b]) for b in single.columns], f(lo.held_f1[m]),
             "" if m == "random" else f"{lo.diff_vs_random[m]:+.3f} [{lo.ci_lo[m]:+.3f}, {lo.ci_hi[m]:+.3f}]".replace("-0.000", "0.000").replace("+0.000", "0.000")) for m in ["random", "ga", "coevolution", "tpe", "coordinate_ascent"]]
    _tab("optimizers", "Method & " + " & ".join(f"\\makecell[r]{{Regret\\\\@{b}}}" for b in single.columns) + r" & \makecell[r]{Held-out\\F1} & \makecell[r]{$\Delta$ vs.\ random\\{[95\,\% CI]}}", rows, "l" + "r" * (len(single.columns) + 2))
    multi = runs[runs["mode"] == "multi"].groupby(["method", "budget"]).hv_ratio.median().unstack()
    _tab("pareto_search", "Method & " + " & ".join(f"HV@{b}" for b in multi.columns), [(METHOD[m], *[f(multi.loc[m, b]) for b in multi.columns]) for m in ["random", "nsga2", "mo_tpe"]],
         "l" + "r" * len(multi.columns))
    views = [("test_a/all", "Unused synthetic, all"), ("test_a/no_toc_refs", "\\quad without table-of-contents sources"),
             ("test_a/no_dev_passage", "\\quad without passages shared with dev"), ("test_a/short", "\\quad short questions only"),
             ("test_a/FDM", "\\quad FDM only"), ("test_a/MSLA", "\\quad resin (MSLA) only"), ("test_b/all", "Independently written")]
    rep_ = pd.read_csv(RESULTS.parent / "data/reference_repairs.csv", dtype={"qid": str})
    qq = questions().set_index("qid")
    esc = lambda x: re.sub(r"([%&#_$])", r"\\\1", str(x))
    _tab("repairs", r"Question & Manual section used & Original reference (shortened)",
         [(esc(qq.question[r.qid]), esc(r.manual_section), esc(r.original_reference[:110]) + "\\dots") for r in rep_.itertuples()], "p{0.3\\linewidth}p{0.2\\linewidth}p{0.42\\linewidth}")
    _tab("generalization", r"Question set & $n$ & \makecell[r]{F1 of best\\dev config} & \makecell[r]{Rank\\of 43} & \makecell[r]{Gain over\\random} & \makecell[r]{Selection\\optimism} & \makecell[r]{$\rho$, random\\finalists}",
         [(label, gen[k]["n_questions"], f(gen[k]["best_dev_on_test"]), gen[k]["test_rank_of_best"], f(gen[k]["best_minus_random_on_test"]),
           f(gen[k]["selection_optimism"]), f(gen[k]["spearman_random_stratum"], 2)) for k, label in views], ">{\\raggedright\\arraybackslash}p{0.34\\linewidth}rrrrrr")




# Figures sized for the two-column Springer layout (column 84 mm, page 174 mm), following the dataviz rules:
# one color per method in a fixed order, a neutral gray for the random baseline, thin marks, hairline grid.
PLOTS = RESULTS.parent / "plots"


def plot_table():
    """The one results table the plot notebook reads: plots/results_table.xlsx."""
    p, rep, frozen, q = protocol(), json.loads((RESULTS / "report.json").read_text()), load_frozen(), questions()
    gen = pd.read_csv(RESULTS / "tables/generalization.csv").query("view == 'all'").drop(columns="view")
    gen["best_dev"] = gen.cfg == frozen["best_dev"]
    lomo = pd.DataFrame(rep["lomo_all"])[["method", "held_f1", "diff_vs_random", "ci_lo", "ci_hi", "n_manuals"]]
    key = pd.DataFrame([
        ("dev_questions", int((q.split == "dev").sum()), "Development questions the search sees"),
        ("test_a_questions", int((q.split == "test_a").sum()), "Unused synthetic questions, scored once after the freeze"),
        ("test_b_questions", int((q.split == "test_b").sum()), "Independently written questions, scored once after the freeze"),
        ("frozen_finalists", len(frozen["configs"]), "Configurations frozen before the test questions were scored"),
        ("seeds", p["search"]["n_seeds"], "Seeds per search method and budget"),
        ("min_effect_f1", p["search"]["min_effect_f1"], "Smallest F1 difference counted as relevant"),
        ("best_dev_cfg", frozen["best_dev"], "Best configuration on the dev questions"),
    ], columns=["name", "value", "description"])
    sheets = {
        "grid": pd.read_csv(RESULTS / "benchmark_table.csv"),
        "optimizer_runs": pd.read_csv(RESULTS / "optimizer_runs.csv"),
        "lomo": lomo,
        "generalization": gen,
        "key_figures": key,
    }
    about = pd.DataFrame([
        ("grid", "One row per configuration: mean dev F1 and mean latency per answer (s) over the dev questions", "Figures 2, 3"),
        ("optimizer_runs", "Search replay on the grid: one row per method, seed and budget", "Figure 4"),
        ("lomo", "Leave-one-manual-out validation at budget 50, difference to random search with 95 % bootstrap interval", "Figure 5"),
        ("generalization", "Frozen finalists: dev F1 and F1 on the untouched question sets", "Figure 6"),
        ("key_figures", "Counts shown in the study design", "Figure 1"),
    ], columns=["sheet", "content", "used_in"])
    PLOTS.mkdir(exist_ok=True)
    with pd.ExcelWriter(PLOTS / "results_table.xlsx", engine="openpyxl") as w:
        about.to_excel(w, sheet_name="about", index=False)
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=name, index=False)
    return PLOTS / "results_table.xlsx"


def figures():
    """Write the results table, run plots/figures.ipynb on it and hand its PDFs to the paper build."""
    import shutil
    import nbformat
    from nbclient import NotebookClient
    plot_table()
    nb = nbformat.read(PLOTS / "figures.ipynb", as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PLOTS)}}).execute()
    nbformat.write(nb, PLOTS / "figures.ipynb")
    out = RESULTS / "figures"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(PLOTS / "figures", out)
    return sorted(p.name for p in out.glob("*.pdf"))


if __name__ == "__main__":
    main()
    print(figures())
