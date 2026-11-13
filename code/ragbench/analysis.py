"""Statistics, tables, figures and LaTeX macros. Manuscript numbers come from results.tex, never by hand."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .common import RESULTS, cfg_id, config_table, db, grid_configs, parse_cfg, protocol, questions, scored_answers, sha
from .search import pareto_mask

def _out(name):
    d = RESULTS / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def optimizer_stats(runs):
    """Each method against random search at every budget; seeds are independent runs, p-values Holm-corrected."""
    rows = []
    for (mode, budget), g in runs.groupby(["mode", "budget"]):
        metric = "regret" if mode == "single" else "hv_ratio"
        base = g[g.method == "random"][metric]
        for method, x in g.groupby("method"):
            row = {"mode": mode, "budget": budget, "method": method, "median": x[metric].median(),
                   "q25": x[metric].quantile(.25), "q75": x[metric].quantile(.75)}
            if method != "random":
                row["p"] = stats.mannwhitneyu(x[metric], base).pvalue
                row["cliffs_delta_vs_random"] = cliffs_delta(x[metric], base)
            rows.append(row)
    t = pd.DataFrame(rows)
    has_p = t.p.notna()
    t.loc[has_p, "p_holm"] = multipletests(t.p[has_p], method="holm")[1]
    return t


def factorial_anova(df):
    """All grid cells are observed, so effects are not confounded by any sampler (unlike the archived GA sample)."""
    d = df.assign(ov=df.chunk_overlap, cs=df.chunk_size, k=df.top_k, temp=df.temperature)
    main = ["C(model)", "C(cs)", "C(ov)", "C(k)", "C(temp)"]
    pairs = [f"{a}:{b}" for i, a in enumerate(main) for b in main[i + 1:]]
    a = sm.stats.anova_lm(smf.ols("f1 ~ " + "+".join(main + pairs), data=d).fit(), typ=2)
    a["partial_eta2"] = a.sum_sq / (a.sum_sq + a.loc["Residual", "sum_sq"])
    a.loc["Residual", "partial_eta2"] = np.nan
    return a.sort_values("partial_eta2", ascending=False)


def lomo_stats(lomo, p):
    """Mean held-out F1 per manual and method, paired against random, CI over manuals. This, not the
    table replay, decides whether a method beats random: its lower bound must exceed min_effect_f1."""
    wide = lomo.groupby(["manual", "method"]).held_f1.mean().unstack()
    rng = np.random.default_rng(p["bootstrap"]["seed"])
    rows = []
    for m in wide.columns:
        d = (wide[m] - wide["random"]).dropna().values  # A manual whose single question has no score drops out.
        boot = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(p["bootstrap"]["n"])]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"method": m, "held_f1": wide[m].mean(), "worst_quarter_manuals": wide[m].dropna().quantile(.25),
                     "diff_vs_random": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "beats_random": bool(m != "random" and lo > p["search"]["min_effect_f1"]), "n_manuals": len(d)})
    out = pd.DataFrame(rows)
    if "coevolution" in wide:  # The specific question: does the opposing population add anything over the plain GA?
        d = (wide["coevolution"] - wide["ga"]).dropna().values
        boot = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(p["bootstrap"]["n"])]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.loc[out.method == "coevolution", ["diff_vs_ga", "ga_ci_lo", "ga_ci_hi"]] = [d.mean(), lo, hi]
        out.loc[out.method == "coevolution", "beats_ga"] = bool(lo > p["search"]["min_effect_f1"])
    return out


def generalization(judge, frozen):
    """Dev versus untouched questions, frozen configurations only. Test sets differ from dev in length
    and reference style, so raw dev minus test mixes selection with shift. Random finalists were not
    selected: the best configuration's drop beyond theirs is the selection optimism."""
    dev = config_table(judge, "dev").set_index("cfg")
    rows, summary = [], {}
    q = questions()
    for split in ("test_a", "test_b"):
        a = scored_answers(judge, split, cfgs=set(frozen["configs"]))
        if a.empty:
            continue
        a = a.merge(q[["qid", "pointer_ref", "toc_ref", "reference_contexts"]], on="qid")
        dev_passages = set(q[q.split == "dev"].reference_contexts)
        views = {"all": a}
        if split == "test_a":
            views |= {"short": a[a.query_length == "SHORT"], "no_pointer_refs": a[~a.pointer_ref], "no_toc_refs": a[~a.toc_ref], "no_dev_passage": a[~a.reference_contexts.isin(dev_passages)], "FDM": a[a.process == "FDM"], "MSLA": a[a.process == "MSLA"]}
        for view, x in views.items():
            for cid, g in x.groupby("cfg"):
                rows.append({"split": split, "view": view, "cfg": cid, "dev": dev.f1[cid], "test": g.f1.mean(), "n": g.f1.count()})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, summary
    for (split, view), g in t.groupby(["split", "view"]):
        g = g.set_index("cfg")
        rand = g.loc[[c for c in frozen["random"] if c in g.index]]
        best = g.loc[frozen["best_dev"]]
        drop_random = float((rand.dev - rand.test).mean())
        summary[f"{split}/{view}"] = {"n_questions": int(best.n), "best_dev_on_dev": best.dev, "best_dev_on_test": best.test,
                                      "raw_drop": best.dev - best.test, "drop_random_configs": drop_random,
                                      "selection_optimism": best.dev - best.test - drop_random,
                                      "best_minus_random_on_test": best.test - float(rand.test.mean()),
                                      "spearman_random_stratum": float(stats.spearmanr(rand.dev, rand.test).statistic),
                                      "test_rank_of_best": int((g.test > best.test).sum()) + 1, "n_frozen": len(g)}
    return t, summary


def noise_estimate(judge, seeds):
    tabs = [config_table(judge, "dev", s).set_index("cfg").f1.rename(s) for s in seeds]
    w = pd.concat(tabs, axis=1).dropna()
    return {"n_configs": len(w), "within_config_sd": float(w.std(axis=1).mean()),
            "between_config_sd": float(w.mean(axis=1).std()), "spearman_first_two": float(stats.spearmanr(w.iloc[:, 0], w.iloc[:, 1]).statistic)}


def failures(judge):
    """Missing judgments per model: a judge that fails more on some models would bias means."""
    a = scored_answers(judge)
    a = a[a.cfg.str.count(r"\|") == 4]
    return a.assign(model=a.cfg.str.rsplit("|", n=4).str[0]).groupby("model").f1.agg(answers="size", failed=lambda x: int(x.isna().sum())).reset_index()


def effects_from(a):
    """Question fixed effects and question-clustered errors. Main effects come from the model without
    interactions: with interactions in, a main-effect test only covers the reference level of the other factor."""
    present = [f for f in ("model", "chunk_size", "chunk_overlap", "top_k", "temperature") if a[f].nunique() > 1]
    main = " + ".join(f"C({f})" for f in present) + " + C(qid)"
    pairs = [p for p in ("C(model):C(top_k)", "C(chunk_size):C(top_k)") if all(x[2:-1] in present for x in p.split(":"))]
    groups = {"groups": pd.factorize(a.qid)[0]}
    tests = []
    for formula, keep in ((main, lambda i: ":" not in i), (main + " + " + " + ".join(pairs), lambda i: ":" in i)):
        w = smf.ols("f1 ~ " + formula, data=a).fit(cov_type="cluster", cov_kwds=groups).wald_test_terms(skip_single=False, scalar=True).table
        w = w[~w.index.str.contains("qid|Intercept")]
        tests.append(w[[keep(i) for i in w.index]])
    return pd.concat(tests)


def answer_level_effects(judge):
    a = scored_answers(judge).dropna(subset=["f1"])
    a = a[a.cfg.isin({cfg_id(c) for c in grid_configs()})]  # Not the archived arms: their levels are not in the factorial.
    return effects_from(pd.concat([a[["qid", "f1"]].reset_index(drop=True), pd.DataFrame([parse_cfg(c) for c in a.cfg])], axis=1))


def judge_check(primary, secondary, cfgs):
    """Agreement across judges on random configurations only. Same-family gap per model: the Qwen judge's
    advantage for a candidate beyond the other judge. Few models, and family is tied to size: descriptive."""
    a, b = (scored_answers(j, cfgs=set(cfgs)).dropna(subset=["f1"]) for j in (primary, secondary))
    m = a.merge(b, on=["cfg", "qid"], suffixes=("_p", "_s"))
    c = m.groupby("cfg").agg(p=("f1_p", "mean"), s=("f1_s", "mean")).reset_index()
    c["model"] = c.cfg.str.rsplit("|", n=4).str[0]
    per_model = c.assign(gap=c.s - c.p).groupby("model").gap.mean()
    return {"n_configs": len(c), "n_answers": len(m), "spearman_config": float(stats.spearmanr(c.p, c.s).statistic),
            "spearman_answer": float(stats.spearmanr(m.f1_p, m.f1_s).statistic), "gap_secondary_minus_primary_per_model": per_model.round(4).to_dict()}


def judge_controls(judge, retest_label, retest_cfgs):
    a = scored_answers(judge)
    ctl = a[a.cfg.str.startswith("control|")].groupby("cfg").f1.mean().round(3).to_dict()
    x = a[a.cfg.isin(retest_cfgs)].dropna(subset=["f1"])
    y = scored_answers(retest_label, cfgs=set(retest_cfgs)).dropna(subset=["f1"])
    m = x.merge(y, on=["cfg", "qid"], suffixes=("_1", "_2"))
    return {"controls": ctl, "retest_n": len(m), "retest_spearman": float(stats.spearmanr(m.f1_1, m.f1_2).statistic), "retest_mean_abs_diff": float((m.f1_1 - m.f1_2).abs().mean())}


def export_rating_sheet(answers, tag, n, seed):
    """Blind sheet for two raters, plus a separate key. `answers`: cfg, qid, answer. Nothing about the
    configuration or any judge score reaches the raters."""
    q = questions().set_index("qid")
    u = answers[answers.answer.notna()].drop_duplicates(["qid", "answer"])
    s = u.sample(min(n, len(u)), random_state=seed).reset_index(drop=True)
    sheet = pd.DataFrame({"item": s.index, "question": s.qid.map(q.question), "reference": s.qid.map(q.reference), "answer": s.answer,
                          "rating_0_wrong_1_partial_2_correct": "", "refusal_0_1": "", "harmful_if_followed_0_1": "", "comment": ""})
    for rater in "AB":
        sheet.to_csv(RESULTS / f"human_{tag}_SHEET_rater{rater}.csv", index=False)
    s.assign(item=s.index)[["item", "cfg", "qid", "answer"]].to_csv(RESULTS / f"human_{tag}_KEY.csv", index=False)
    return len(s)


def rating_results(tag, judges):
    """Agreement of each judge with the mean of two blind ratings. A one-person sheet may use rater A for both."""
    key = pd.read_csv(RESULTS / f"human_{tag}_KEY.csv", dtype={"qid": str}, keep_default_na=False)
    key["item"] = key["item"].astype(int)
    col = "rating_0_wrong_1_partial_2_correct"
    done_b = RESULTS / f"human_{tag}_DONE_raterB.csv"
    a = pd.read_csv(RESULTS / f"human_{tag}_DONE_raterA.csv").set_index("item")[col]
    b = pd.read_csv(done_b).set_index("item")[col] if done_b.exists() else a
    key["rating_a"], key["rating_b"] = key["item"].map(a), key["item"].map(b)
    if key.rating_a.isna().any() or key.rating_b.isna().any():
        raise SystemExit(f"Rating sheet {tag} is incomplete")
    a, b = key.rating_a, key.rating_b
    key["human"] = (a + b) / 4  # 0..1, like F1.
    q = questions().set_index("qid")
    key["ref_sha"], key["ans_sha"] = key.qid.map(q.reference).map(sha), key.answer.map(sha)
    with db() as con:
        scores = pd.read_sql("SELECT judge, ref_sha, ans_sha, f1 FROM scores", con)
    out = {"n": len(key), "kappa_raters": weighted_kappa(a, b) if done_b.exists() else None}
    for j in judges:
        m = key.merge(scores[scores.judge == j], on=["ref_sha", "ans_sha"]).dropna(subset=["f1"])
        if len(m) < 0.9 * len(key):
            raise SystemExit(f"Only {len(m)} of {len(key)} rated answers have a score from {j}")
        out[j] = {"n": len(m), "spearman_human": float(stats.spearmanr(m.f1, m.human).statistic), "mean_abs_error": float((m.f1 - m.human).abs().mean())}
    return out


def gate_decision(result, default, other, g):
    """Preregistered rule. Returns the primary judge, or None if a human has to decide."""
    # A constant judge has no correlation at all, which must count as failing, not as passing.
    d, o = (-1.0 if np.isnan(result[j]["spearman_human"]) else result[j]["spearman_human"] for j in (default, other))
    if max(d, o) < g["min_spearman"]:
        return None
    return other if d < g["min_spearman"] or o - d > g["max_gap"] else default


def weighted_kappa(a, b, k=3):
    """Quadratic-weighted Cohen's kappa for ordinal ratings 0..k-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    o = np.zeros((k, k))
    np.add.at(o, (a, b), 1)
    o /= o.sum()
    e = np.outer(o.sum(1), o.sum(0))
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    return float(1 - (w * o).sum() / (w * e).sum())


def lexical_coverage():
    """Retrieval-only diagnostic without any LLM: fraction of reference-context words present in the top-k chunks."""
    import json
    words = lambda s: set(re.findall(r"[a-z]+", str(s).lower()))
    q = questions("dev").set_index("qid")
    with db() as con:
        rows = con.execute("SELECT chunk_size, chunk_overlap, qid, docs FROM retrieval").fetchall()
    out = []
    for cs, ov, qid, docs in rows:
        if qid in q.index:
            ref, docs = words(q.reference_contexts[qid]), json.loads(docs)
            for k in protocol()["grid"]["top_k"]:
                out.append({"chunk_size": cs, "chunk_overlap": ov, "top_k": k, "coverage": len(ref & words(" ".join(docs[:k]))) / max(len(ref), 1)})
    return pd.DataFrame(out).groupby(["chunk_size", "chunk_overlap", "top_k"]).coverage.mean().reset_index()


def fig_regret(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, mode, metric, label in ((axes[0], "single", "regret", "Simple regret (F1)"), (axes[1], "multi", "hv_ratio", "Hypervolume ratio")):
        g = runs[runs["mode"] == mode]
        for m, x in g.groupby("method"):
            s = x.groupby("budget")[metric]
            ax.errorbar(s.median().index, s.median(), yerr=[s.median() - s.quantile(.25), s.quantile(.75) - s.median()], marker="o", capsize=3, label=m)
        ax.set(xlabel="Distinct configurations evaluated", ylabel=label)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_optimizers.pdf")
    plt.close(fig)


def fig_pareto(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    front = df[pareto_mask(df.f1, df.latency_s)].sort_values("latency_s")
    for m, g in df.groupby("model"):
        ax.scatter(g.latency_s, g.f1, s=10, alpha=.45, label=m)
    ax.step(front.latency_s, front.f1, where="post", color="black", lw=1.2, label="Pareto front")
    ax.set(xscale="log", xlabel="Mean sequential latency per answer (s)", ylabel="Factual Correctness F1, dev")
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_pareto.pdf")
    plt.close(fig)


def fig_heatmaps(df):
    pairs = [("top_k", "chunk_size"), ("top_k", "model"), ("temperature", "model")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (x, y) in zip(axes, pairs):
        p = df.pivot_table(index=y, columns=x, values="f1")
        im = ax.imshow(p.values, aspect="auto", vmin=df.f1.min(), vmax=df.f1.max(), cmap="viridis")
        ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), p.index, fontsize=7)
        ax.set(xlabel=x, ylabel=y)
    fig.colorbar(im, ax=axes, label="Mean F1")
    fig.savefig(RESULTS / "fig_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_generalization(t):
    t = t[t.view == "all"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for split, g in t.groupby("split"):
        ax.scatter(g.dev, g.test, s=18, label=split)
    lim = [min(t.dev.min(), t.test.min()), max(t.dev.max(), t.test.max())]
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set(xlabel="F1 on dev questions", ylabel="F1 on untouched questions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_generalization.pdf")
    plt.close(fig)


def write_macros(values):
    """values: name -> number or string. Names may contain only letters, as LaTeX requires."""
    lines = []
    for k, v in values.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\rr{k}}}{{{v:.4g}}}" if isinstance(v, (int, float, np.floating)) else f"\\newcommand{{\\rr{k}}}{{{v}}}")
    (RESULTS / "results.tex").write_text("\n".join(lines) + "\n")
