"""Optimizers replayed on the finished lookup table, so every seed sees identical scores.
Budget = number of distinct configurations a run may look at; repeats are free, as in the archived cache."""
import hashlib
import json
import random
import warnings

import numpy as np
import optuna
import pandas as pd
from deap import base, tools

from .common import CODE, DATA, FIELDS, PROTOCOL_PATH, RESULTS, cfg_id, protocol

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


class Budget(Exception):
    pass


class Table:
    """Holds objectives per configuration and counts distinct evaluations in one run."""

    def __init__(self, df, grid, answers=None):
        self.levels = [list(grid[f]) for f in FIELDS]
        self.f1 = dict(zip(df.cfg, df.f1))
        self.lat = dict(zip(df.cfg, df.latency_s))
        if answers is not None:  # Per-question scores: configurations in rows, questions in columns.
            wide = answers.pivot_table(index="cfg", columns="qid", values="f1")
            self.q = {c: r for c, r in zip(wide.index, wide.to_numpy())}
            self.n_q = wide.shape[1]
        self.reset(0)

    def reset(self, budget):
        self.budget, self.seen, self.opponents = budget, {}, None

    def on(self, cid, subset):
        return float(np.nan_to_num(np.nanmean(self.q[cid][list(subset)])))

    def pick(self):
        """Selected configuration: best plain mean, or for coevolution best against its opponents."""
        if self.opponents:
            return max(self.seen, key=lambda c: np.mean([self.on(c, s) for s in self.opponents]))
        return max(self.seen, key=lambda c: self.seen[c][0])

    def cid(self, ind):
        return cfg_id({f: self.levels[i][g] for i, (f, g) in enumerate(zip(FIELDS, ind))})

    def ev(self, ind):
        c = self.cid(ind)
        if c not in self.seen:
            if len(self.seen) >= self.budget:
                raise Budget
            self.seen[c] = (self.f1[c], self.lat[c])
        return self.seen[c]

    def random_ind(self, rng):
        return [rng.randrange(len(v)) for v in self.levels]


def mutate(ind, rng, levels, indpb):
    """Archived operator: the model jumps anywhere, ordered parameters step to a neighbor."""
    for i, vals in enumerate(levels):
        if rng.random() < indpb:
            hi = len(vals) - 1
            if FIELDS[i] == "model":
                ind[i] = (ind[i] + rng.randint(1, hi)) % (hi + 1)
            elif ind[i] == 0:
                ind[i] = 1
            elif ind[i] == hi:
                ind[i] = hi - 1
            else:
                ind[i] += rng.choice([-1, 1])
    return ind


def vary(parents, rng, t, g):
    kids = [list(x) for x in parents]
    for a, b in zip(kids[::2], kids[1::2]):
        if rng.random() < g["cxpb"]:
            for i in range(len(a)):
                if rng.random() < 0.5:
                    a[i], b[i] = b[i], a[i]
    for k in kids:
        if rng.random() < g["mutpb"]:
            mutate(k, rng, t.levels, g["indpb"])
    return kids


def random_search(t, rng, g):
    while True:
        t.ev(t.random_ind(rng))


def ga(t, rng, g):
    """Single-objective GA as archived: tournament, uniform crossover, mixed mutation, elitism."""
    pop = [t.random_ind(rng) for _ in range(g["pop_size"])]
    fit = [t.ev(x)[0] for x in pop]
    stall = 0
    while True:
        elite = pop[int(np.argmax(fit))]
        chosen = [max(rng.sample(range(len(pop)), g["tournsize"]), key=lambda i: fit[i]) for _ in range(len(pop) - g["elites"])]
        before = len(t.seen)
        pop = vary([pop[i] for i in chosen], rng, t, g) + [list(elite)]
        fit = [t.ev(x)[0] for x in pop]
        stall = stall + 1 if len(t.seen) == before else 0
        if stall > 50:  # Converged onto known points: spend the rest on fresh random ones.
            pop[0] = t.random_ind(rng)
            fit[0] = t.ev(pop[0])[0]


def coordinate_ascent(t, rng, g):
    """Random start, then one parameter at a time, restart when no parameter improves."""
    while True:
        cur = t.random_ind(rng)
        best = t.ev(cur)[0]
        improved = True
        while improved:
            improved = False
            for i in rng.sample(range(len(cur)), len(cur)):
                for v in range(len(t.levels[i])):
                    cand = cur[:i] + [v] + cur[i + 1:]
                    f = t.ev(cand)[0]
                    if f > best:
                        cur, best, improved = cand, f, True


def coevolution(t, rng, g):
    """Host-parasite coevolution. Hosts are configurations with the GA's operators. Parasites are question
    subsets that evolve toward what the current hosts fail but at least one host can solve, so that the
    search cannot settle on the easy part of the benchmark. Unsolved questions earn nothing, because a
    parasite made of impossible questions would stop telling configurations apart."""
    c = protocol()["search"]["coevolution"]
    size = max(2, round(t.n_q * c["subset_share"]))
    new_subset = lambda: tuple(sorted(rng.sample(range(t.n_q), size)))
    parasites = [new_subset() for _ in range(c["parasites"])]
    hall = []
    hosts = [t.random_ind(rng) for _ in range(g["pop_size"])]
    stall = 0
    while True:
        t.opponents = parasites + hall
        before = len(t.seen)
        for h in hosts:
            t.ev(h)
        ids = [t.cid(h) for h in hosts]
        fit = [np.mean([t.on(i, s) for s in t.opponents]) for i in ids]

        scores = np.array([t.q[i] for i in ids])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # A question without any score counts as unsolved.
            value = np.where(np.nan_to_num(np.nanmax(scores, axis=0)) > 0, 1 - np.nan_to_num(np.nanmean(scores, axis=0)), 0.0)
        worth = [value[list(s)].mean() for s in parasites]
        best = parasites[int(np.argmax(worth))]
        if best not in hall:
            hall = (hall + [best])[-c["hall_of_fame"]:]  # Memory, so the search cannot forget solved opponents.

        kids = []
        for _ in parasites:
            a, b = (max(rng.sample(range(len(parasites)), g["tournsize"]), key=lambda i: worth[i]) for _ in "ab")
            pool = sorted(set(parasites[a]) | set(parasites[b]))
            kid = set(rng.sample(pool, size))
            if rng.random() < c["swap_prob"]:
                kid.remove(rng.choice(sorted(kid)))
                kid.add(rng.choice([q for q in range(t.n_q) if q not in kid]))
            kids.append(tuple(sorted(kid)))
        parasites = kids

        elite = hosts[int(np.argmax(fit))]
        chosen = [max(rng.sample(range(len(hosts)), g["tournsize"]), key=lambda i: fit[i]) for _ in range(len(hosts) - g["elites"])]
        hosts = vary([hosts[i] for i in chosen], rng, t, g) + [list(elite)]
        stall = stall + 1 if len(t.seen) == before else 0
        if stall > 50:  # Converged onto known points: spend the rest on fresh random ones.
            hosts[0] = t.random_ind(rng)


def _optuna(t, rng, directions, value):
    study = optuna.create_study(directions=directions, sampler=optuna.samplers.TPESampler(seed=rng.randrange(2 ** 31), multivariate=True))
    repeats = 0
    while True:
        trial = study.ask()
        # Same knowledge as the GA: the model is a category, the rest are ordered.
        ind = [trial.suggest_categorical(f, list(range(len(t.levels[i])))) if f == "model"
               else trial.suggest_int(f, 0, len(t.levels[i]) - 1) for i, f in enumerate(FIELDS)]
        before = len(t.seen)
        study.tell(trial, value(t.ev(ind)))
        repeats = repeats + 1 if len(t.seen) == before else 0
        if repeats > 50:  # TPE keeps proposing known points: spend budget on a fresh random one.
            t.ev(t.random_ind(rng))
            repeats = 0


def tpe(t, rng, g):
    _optuna(t, rng, ["maximize"], lambda v: v[0])


def mo_tpe(t, rng, g):
    _optuna(t, rng, ["maximize", "minimize"], lambda v: list(v))


class _Fit(base.Fitness):
    weights = (1.0, -1.0)  # F1 up, latency down.


class _Ind(list):
    pass


def nsga2(t, rng, g):
    def mk(x):
        i = _Ind(x)
        i.fitness = _Fit(t.ev(x))
        return i

    def survive(cands):
        """Non-dominated sorting with crowding; each survivor keeps (rank, crowding) for mating."""
        out = []
        for rank, fr in enumerate(tools.sortNondominated(cands, len(cands), first_front_only=False)):
            tools.emo.assignCrowdingDist(fr)
            for i in fr:
                i.key = (-rank, i.fitness.crowding_dist)
            out.extend(sorted(fr, key=lambda i: -i.fitness.crowding_dist))
        return out[:g["pop_size"]]
    pop = survive([mk(t.random_ind(rng)) for _ in range(g["pop_size"])])
    while True:
        parents = [max(rng.sample(pop, 2), key=lambda i: i.key) for _ in pop]
        pop = survive(pop + [mk(k) for k in vary(parents, rng, t, g)])


SINGLE = {"random": random_search, "ga": ga, "coordinate_ascent": coordinate_ascent, "tpe": tpe}
MULTI = {"random": random_search, "nsga2": nsga2, "mo_tpe": mo_tpe}


def pareto_mask(f1, lat):
    """True for points no other point beats on both quality (max) and latency (min)."""
    f1, lat = np.asarray(f1), np.asarray(lat)
    return np.array([not np.any((f1 >= a) & (lat <= b) & ((f1 > a) | (lat < b))) for a, b in zip(f1, lat)])


def hypervolume(points, ref_lat):
    """Area above F1=0 and left of ref_lat on a log-latency axis, so one very slow model cannot
    dominate the score. Points past ref_lat contribute nothing."""
    hv, best = 0.0, 0.0
    ordered = sorted((lat, f) for f, lat in points if lat < ref_lat)
    for i, (lat, f) in enumerate(ordered):
        best = max(best, f)
        nxt = ordered[i + 1][0] if i + 1 < len(ordered) else ref_lat
        hv += best * (np.log10(nxt) - np.log10(lat))
    return hv


def replay(df, p=None, answers=None):
    """Runs every optimizer, budget and seed. Coevolution needs per-question `answers`."""
    p = p or protocol()
    s = p["search"]
    t = Table(df, p["grid"], answers)
    best = df.f1.max()
    ref = p["search"].get("hv_ref_latency_s") or df.latency_s.max()
    true_hv = hypervolume(list(zip(df.f1, df.latency_s)), ref)
    rows = []
    for mode, methods in (("single", {**SINGLE, "coevolution": coevolution} if answers is not None else SINGLE), ("multi", MULTI)):
        for name, fn in methods.items():
            for seed in range(s["n_seeds"]):
                for budget in s["budgets"]:
                    t.reset(budget)
                    rng = random.Random(f"{name}-{seed}")
                    try:
                        fn(t, rng, s["ga"])
                    except Budget:
                        pass
                    if mode == "single":
                        cid = t.pick()
                        rows.append({"mode": mode, "method": name, "seed": seed, "budget": budget, "selected": cid,
                                     "f1": t.seen[cid][0], "regret": best - t.seen[cid][0], "n_eval": len(t.seen)})
                    else:
                        hv = hypervolume(list(t.seen.values()), ref)
                        rows.append({"mode": mode, "method": name, "seed": seed, "budget": budget, "selected": "",
                                     "hv": hv, "hv_ratio": hv / true_hv, "n_eval": len(t.seen)})
    return pd.DataFrame(rows)


def replay_lomo(answers, p=None, process=None):
    """Leave-one-manual-out. Search sees only the other manuals; the winner is scored on the held-out
    one. Replaying many seeds on one table only measures optimizer luck; this measures whether a
    search result carries over to questions it never saw. `answers`: cfg, qid, source_file, process, f1, wall_s."""
    p = p or protocol()
    s = p["search"]
    a = answers if process is None else answers[answers.process == process]
    rows = []
    for manual, test in a.groupby("source_file"):
        rest = a[a.source_file != manual]
        train = rest.groupby("cfg").agg(f1=("f1", "mean"), latency_s=("wall_s", "mean")).reset_index()
        held = test.groupby("cfg").f1.mean()
        t = Table(train, p["grid"], rest)
        for name, fn in {**SINGLE, "coevolution": coevolution}.items():
            for seed in range(s["lomo_seeds"]):
                t.reset(s["freeze_budget"])
                try:
                    fn(t, random.Random(f"{name}-{manual}-{seed}"), s["ga"])
                except Budget:
                    pass
                cid = t.pick()
                rows.append({"manual": manual, "method": name, "seed": seed, "selected": cid, "train_f1": t.seen[cid][0],
                             "held_f1": held.get(cid, np.nan), "n_held": len(test.qid.unique())})
    return pd.DataFrame(rows)


def _sha_files(paths):
    h = hashlib.sha256()
    for f in paths:
        if f.exists():
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def _data_sha():
    return _sha_files([DATA / n for n in ("questions.csv", "reference_repairs.csv", "test_b_human.csv")])


def freeze(df, runs, p=None):
    """Fixes the configurations allowed to touch the test questions, before any test score exists."""
    p = p or protocol()
    n = p["finalists"]
    front = df[pareto_mask(df.f1, df.latency_s)].sort_values("latency_s")
    idx = np.unique(np.linspace(0, len(front) - 1, min(n["n_pareto"], len(front))).round().astype(int))
    pareto = list(front.iloc[idx].cfg)
    sel = runs[(runs["mode"] == "single") & (runs.budget == p["search"]["freeze_budget"]) & (runs.method != "coevolution")]
    selected = list(sel.selected.value_counts().head(n["n_selected"]).index)
    co = runs[(runs["mode"] == "single") & (runs.budget == p["search"]["freeze_budget"]) & (runs.method == "coevolution")]
    coevolved = list(co.selected.value_counts().head(n["n_coevolution"]).index)
    rng = random.Random(n["sample_seed"])
    rest = sorted(set(df.cfg) - set(pareto) - set(selected) - set(coevolved))
    rand = rng.sample(rest, min(n["n_random"], len(rest)))
    best = df.loc[df.f1.idxmax(), "cfg"]
    configs = list(dict.fromkeys([best] + pareto + selected + coevolved + rand + list(p.get("archived_arms", []))))
    out = {"configs": configs, "best_dev": best, "pareto": pareto, "selected": selected, "coevolution": coevolved, "random": rand,
           "archived_arms": list(p.get("archived_arms", [])),
           "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
           "code_sha256": _sha_files(sorted(CODE.glob("ragbench/*.py"))),
           "data_sha256": _data_sha()}
    path = RESULTS / "frozen_finalists.json"
    if path.exists():
        raise SystemExit(f"{path} already exists. Test scores may depend on it, so it is not overwritten.")
    path.write_text(json.dumps(out, indent=1))
    return out


def load_frozen():
    frozen = json.loads((RESULTS / "frozen_finalists.json").read_text())
    if frozen["data_sha256"] != _data_sha() or frozen["protocol_sha256"] != hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest():
        raise SystemExit("Questions, repairs or protocol changed after the freeze. Stop: the test results would no longer be untouched.")
    return frozen
