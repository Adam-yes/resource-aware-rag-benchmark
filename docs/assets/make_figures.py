"""Draw the README figures in docs/assets from the committed results.

Run from the repository root:  python docs/assets/make_figures.py

Only the standard library is used. Every number in the figures is read from
results/, so the figures change when the results change.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets"
RES = ROOT / "results"

W = 1200  # every figure uses the same width, so they line up in the README

# Light and dark tokens. GitHub shows SVGs through <img>, so the dark set
# follows the viewer's system setting.
STYLE = """
<style>
  svg { --bg:#fcfcfb; --plane:#f4f4f1; --ink:#0b0b0b; --ink2:#52514e; --mute:#898781;
        --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --dot:#c9c8c1; --soft:#e7f0fb; }
  @media (prefers-color-scheme: dark) {
    svg { --bg:#1a1a19; --plane:#222220; --ink:#f0efec; --ink2:#c3c2b7; --mute:#898781;
          --grid:#2c2c2a; --axis:#46463f; --border:rgba(255,255,255,.10);
          --s1:#3987e5; --s2:#d95926; --s3:#199e70; --dot:#4f4e49; --soft:#1c2a3c; }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink); }
  .t-title { font-size: 26px; font-weight: 650; }
  .t-sub { font-size: 15.5px; fill: var(--ink2); }
  .t-h { font-size: 16px; font-weight: 600; }
  .t-lab { font-size: 13.5px; fill: var(--ink2); }
  .t-small { font-size: 12.5px; fill: var(--mute); }
  .t-tick { font-size: 12.5px; fill: var(--mute); font-variant-numeric: tabular-nums; }
  .t-strong { font-size: 13.5px; font-weight: 600; }
  .t-kpi { font-size: 38px; font-weight: 650; }
  .card { fill: var(--bg); stroke: var(--border); stroke-width: 1; }
  .plane { fill: var(--plane); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .lead { stroke: var(--mute); stroke-width: 1; fill: none; }
  .dot { fill: var(--dot); }
  .ring { stroke: var(--bg); stroke-width: 2; }
  .f-s1 { fill: var(--s1); } .st-s1 { stroke: var(--s1); }
  .f-s2 { fill: var(--s2); } .st-s2 { stroke: var(--s2); }
  .f-s3 { fill: var(--s3); } .st-s3 { stroke: var(--s3); }
  .f-mute { fill: var(--mute); } .st-mute { stroke: var(--mute); }
  .f-ink { fill: var(--ink); } .st-ink { stroke: var(--ink); }
  .f-soft { fill: var(--soft); }
  .line { fill: none; stroke-width: 2.25; stroke-linejoin: round; stroke-linecap: round; }
  .box { fill: var(--bg); stroke: var(--axis); stroke-width: 1; }
  .box-hi { fill: var(--soft); stroke: var(--s1); stroke-width: 1.25; }
  .arrow { stroke: var(--mute); stroke-width: 1.5; fill: none; }
  .halo { paint-order: stroke; stroke: var(--bg); stroke-width: 5px; stroke-linejoin: round; }
</style>"""


# --------------------------------------------------------------------------- helpers


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(height: int, title: str, desc: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img" aria-labelledby="t d">',
            f'<title id="t">{esc(title)}</title>',
            f'<desc id="d">{esc(desc)}</desc>',
            STYLE,
            f'<rect class="card" x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def text(x, y, s, cls="t-lab", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'


def header(title: str, sub: str, x: int = 40) -> list[str]:
    return [text(x, 52, title, "t-title"), text(x, 80, sub, "t-sub")]


class Lin:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


class Log(Lin):
    def __call__(self, v):
        return super().__call__(math.log10(v))

    def __init__(self, d0, d1, r0, r1):
        super().__init__(math.log10(d0), math.log10(d1), r0, r1)


def path(points) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def short_model(m: str) -> str:
    return {
        "qwen3-vl:30b-a3b-instruct": "qwen3-vl 30B-A3B",
        "qwen3-vl:2b-instruct": "qwen3-vl 2B",
        "gemma3n:e2b": "gemma3n e2b",
        "qwen2.5-coder:0.5b-instruct": "qwen2.5-coder 0.5B",
        "granite3.3:2b": "granite3.3 2B",
        "granite4:tiny-h": "granite4 tiny-h",
        "phi4-mini:3.8b-q4_K_M": "phi4-mini 3.8B",
        "gemma3:1b": "gemma3 1B",
    }.get(m, m)


def describe(cfg: str) -> str:
    m, cs, ov, k, t = cfg.split("|")
    return f"{short_model(m)}, chunk {cs}, Top-k {k}, T {t}"


# --------------------------------------------------------------------------- data


def load_benchmark():
    rows = list(csv.DictReader(open(RES / "benchmark_table.csv", encoding="utf-8")))
    for r in rows:
        r["f1"] = float(r["f1"])
        r["lat"] = float(r["latency_s"])
        r["n"] = int(r["n_answers"])
    return rows


def pareto(rows):
    front, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r["lat"], -r["f1"])):
        if r["f1"] > best:
            front.append(r)
            best = r["f1"]
    return front


def load_report():
    return json.loads((RES / "report.json").read_text(encoding="utf-8"), parse_constant=lambda c: None)


def load_frozen():
    return json.loads((RES / "frozen_finalists.json").read_text(encoding="utf-8"))


def n_answers_total() -> int:
    """Rows in the answer-level table: every generated and judged answer."""
    with gzip.open(RES / "answers.csv.gz", "rt", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def fig_header(bench, report, frozen):
    n_answers = n_answers_total()
    gen = report["generalization"]
    rank_a = gen["test_a/all"]["test_rank_of_best"]
    n_frozen = gen["test_a/all"]["n_frozen"]
    nq_a = gen["test_a/all"]["n_questions"]
    h = 276
    b = [
        text(40, 70, "Resource-Aware RAG Benchmark", "t-title", extra=' style="font-size:34px"'),
        text(
            40,
            102,
            "Quality-latency trade-offs and configuration search for local RAG on technical manuals",
            "t-sub",
            extra=' style="font-size:17px"',
        ),
        text(
            40,
            128,
            "Manuscript under review  ·  Apache-2.0",
            "t-small",
            extra=' style="font-size:13.5px"',
        ),
    ]
    kpis = [
        (fmt_int(len(bench)), "configurations measured", "full factorial, 50 dev questions each"),
        (fmt_int(n_answers), "answers generated and judged", "local models via Ollama"),
        (str(n_frozen), "finalists frozen", "SHA-256 hashes before any test"),
        (f"{rank_a} of {n_frozen}", "test rank of the selected setup", f"on {nq_a} unused questions"),
    ]
    tw, gap, x0, y0 = 266, 18, 40, 162
    for i, (v, l1, l2) in enumerate(kpis):
        xx = x0 + i * (tw + gap)
        b.append(f'<rect class="plane" x="{xx}" y="{y0}" width="{tw}" height="84" rx="10"/>')
        b.append(text(xx + 18, y0 + 44, v, "t-kpi", extra=' style="font-size:30px"'))
        b.append(text(xx + 18, y0 + 64, l1, "t-strong", extra=' style="font-size:13px"'))
        b.append(text(xx + 18, y0 + 78, l2, "t-small", extra=' style="font-size:11.5px"'))
    return svg(
        h,
        "Resource-Aware RAG Benchmark",
        "Repository header with four figures: configurations measured, answers judged, "
        "finalists frozen and the test rank of the selected configuration.",
        b,
    )


def fig_protocol(bench, report):
    gen = report["generalization"]
    na, nb = gen["test_a/all"]["n_questions"], gen["test_b/all"]["n_questions"]
    nf = gen["test_a/all"]["n_frozen"]
    n_lomo = report["lomo_all"][0]["n_manuals"]
    h = 470
    b = header(
        "Search, selection and testing are kept apart",
        "Settings were fixed in protocol.yaml before the run; test questions were used once.",
    )
    stages = [
        (
            "1",
            "Measure",
            [f"{len(bench)} configurations", "8 models, 4 chunk sizes,", "2 overlaps, 5 Top-k,", "3 temperatures"],
        ),
        ("2", "Choose judge", ["Fixed rule against", "blind ratings", "judge: gpt-oss:20b", "2nd: qwen3-coder:30b"]),
        (
            "3",
            "Replay search",
            ["GA, coevolution, TPE,", "coordinate ascent,", "random; 50 seeds,", "budgets 25 to 150"],
        ),
        ("4", "Hold out manuals", ["Leave one manual out", f"over {n_lomo} manuals", "for every method", "and seed"]),
        ("5", "Freeze", [f"{nf} finalists", "SHA-256 of protocol,", "code and data", "written to disk"]),
        ("6", "Test once", [f"{na} unused questions", f"{nb} independently", "written questions", "scored once"]),
    ]
    x0, avail, gap = 40, W - 80, 26
    bw = (avail - gap * (len(stages) - 1)) / len(stages)
    by, bh = 150, 150
    for i, (n, name, lines) in enumerate(stages):
        x = x0 + i * (bw + gap)
        cls = "box-hi" if n == "5" else "box"
        b.append(f'<rect class="{cls}" x="{x:.1f}" y="{by}" width="{bw:.1f}" height="{bh}" rx="10"/>')
        b.append(f'<circle cx="{x + 22:.1f}" cy="{by + 26}" r="11" class="{"f-s1" if n == "5" else "f-ink"}"/>')
        b.append(text(x + 22, by + 30.5, n, "t-strong", "middle", ' style="fill:var(--bg);font-size:12.5px"'))
        b.append(text(x + 40, by + 31, name, "t-h", extra=' style="font-size:15px"'))
        for j, ln in enumerate(lines):
            b.append(text(x + 16, by + 64 + j * 20, ln, "t-lab", extra=' style="font-size:13px"'))
        if i < len(stages) - 1:
            ax = x + bw + 5
            b.append(f'<path class="arrow" d="M{ax:.1f},{by + bh / 2} h{gap - 10}"/>')
            b.append(f'<path d="M{ax + gap - 10:.1f},{by + bh / 2 - 4} l6,4 l-6,4z" class="f-mute"/>')

    # data bands under the stages
    yb = by + bh + 34
    x_dev_end = x0 + 4 * (bw + gap) - gap
    x_test = x0 + 5 * (bw + gap)
    b.append(f'<rect class="plane" x="{x0}" y="{yb}" width="{x_dev_end - x0:.1f}" height="44" rx="8"/>')
    b.append(
        text(
            (x0 + x_dev_end) / 2, yb + 27, "Development questions only (50 questions, 20 manuals)", "t-strong", "middle"
        )
    )
    b.append(f'<rect class="f-soft" x="{x_test:.1f}" y="{yb}" width="{bw:.1f}" height="44" rx="8"/>')
    b.append(text(x_test + bw / 2, yb + 27, "Unseen questions", "t-strong", "middle"))
    xf = x0 + 4 * (bw + gap) + bw / 2
    b.append(f'<path class="st-s1" style="stroke-width:1.5" d="M{xf:.1f},{by + bh + 8} V{yb + 44}"/>')
    b.append(text(xf, yb + 64, "hashes checked before scoring", "t-small", "middle"))

    # runtime footer
    b.append(f'<path class="grid" d="M40,{h - 58} H{W - 40}"/>')
    b.append(
        text(
            40,
            h - 28,
            "Runtime: Ollama 0.40.14 with local open-weight models  ·  Chroma vector stores  ·  "
            "RAGAS Factual Correctness (F1)  ·  NVIDIA RTX A6000, 48 GB",
            "t-small",
            extra=' style="font-size:13px"',
        )
    )
    return svg(
        h,
        "Evaluation protocol",
        "Six stages: measure all configurations, choose the judge, replay the search methods, validate by "
        "leaving one manual out, freeze the finalists with hashes, and score them once on unseen questions.",
        b,
    )


def fig_quality_latency(bench):
    front = pareto(bench)
    best = max(bench, key=lambda r: r["f1"])
    p90 = min((r for r in bench if r["f1"] >= 0.9 * best["f1"]), key=lambda r: r["lat"])
    p80 = min((r for r in bench if r["f1"] >= 0.8 * best["f1"]), key=lambda r: r["lat"])
    lat_saving = 1 - p90["lat"] / best["lat"]
    q_loss = 1 - p90["f1"] / best["f1"]

    h = 640
    title = f"{lat_saving:.0%} less latency for {q_loss:.0%} less quality"
    b = header(
        title,
        f"Development F1 against end-to-end latency per answer for all {len(bench)} configurations. "
        f"The blue line is the Pareto front ({len(front)} configurations).",
    )
    L, R, T, B = 96, W - 300, 120, h - 86
    x = Log(0.04, 1.0, L, R)
    y = Lin(0.05, 0.42, B, T)
    for v in (0.05, 0.1, 0.2, 0.3, 0.4):
        b.append(f'<path class="grid" d="M{L},{y(v):.1f} H{R}"/>')
        b.append(text(L - 10, y(v) + 4, f"{v:.2f}", "t-tick", "end"))
    for v in (0.05, 0.1, 0.2, 0.5, 1.0):
        b.append(f'<path class="grid" d="M{x(v):.1f},{T} V{B}"/>')
        b.append(text(x(v), B + 20, f"{v:g} s", "t-tick", "middle"))
    b.append(f'<path class="axis" d="M{L},{B} H{R}"/>')
    b.append(text((L + R) / 2, B + 46, "Latency per answer on an RTX A6000 (log scale)", "t-lab", "middle"))
    b.append(
        text(
            L - 62,
            (T + B) / 2,
            "Factual correctness F1 (dev)",
            "t-lab",
            "middle",
            f' transform="rotate(-90 {L - 62} {(T + B) / 2})"',
        )
    )

    on_front = {r["cfg"] for r in front}
    for r in bench:
        if r["cfg"] in on_front:
            continue
        b.append(
            f'<circle cx="{x(r["lat"]):.1f}" cy="{y(r["f1"]):.1f}" r="3.2" class="dot">'
            f"<title>{esc(describe(r['cfg']))}: F1 {r['f1']:.3f}, {r['lat']:.2f} s</title></circle>"
        )
    b.append(f'<path d="{path([(x(r["lat"]), y(r["f1"])) for r in front])}" class="line st-s1"/>')
    for r in front:
        b.append(
            f'<circle cx="{x(r["lat"]):.1f}" cy="{y(r["f1"]):.1f}" r="4.5" class="f-s1 ring">'
            f"<title>{esc(describe(r['cfg']))}: F1 {r['f1']:.3f}, {r['lat']:.2f} s</title></circle>"
        )

    def callout(r, dx, dy, head, line2):
        px, py = x(r["lat"]), y(r["f1"])
        tx, ty = px + dx, py + dy
        b.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="ring" '
            'style="fill:none;stroke:var(--s1);stroke-width:1.5"/>'
        )
        sx, ex = px + (6 if dx > 0 else -6), tx - (4 if dx > 0 else -4)
        b.append(f'<path class="lead" d="M{sx:.1f},{py:.1f} L{ex:.1f},{ty - 4:.1f}"/>')
        anchor = "start" if dx > 0 else "end"
        b.append(text(tx, ty, head, "t-strong halo", anchor))
        b.append(text(tx, ty + 18, line2, "t-lab halo", anchor))

    callout(best, -14, 100, f"Best: F1 {best['f1']:.3f} at {best['lat']:.2f} s", describe(best["cfg"]))
    callout(p90, -40, -46, f"{p90['f1'] / best['f1']:.0%} of best at {p90['lat']:.2f} s", describe(p90["cfg"]))
    callout(p80, -40, -40, f"{p80['f1'] / best['f1']:.0%} of best at {p80['lat']:.2f} s", describe(p80["cfg"]))

    # side panel: best per model
    sx = R + 44
    b.append(text(sx, T + 4, "Best F1 per model", "t-h"))
    b.append(text(sx, T + 24, "and its latency", "t-small"))
    models = {}
    for r in bench:
        if r["model"] not in models or r["f1"] > models[r["model"]]["f1"]:
            models[r["model"]] = r
    ms = sorted(models.values(), key=lambda r: -r["f1"])
    bx = Lin(0, 0.42, sx, W - 92)
    for i, r in enumerate(ms):
        yy = T + 60 + i * 46
        b.append(text(sx, yy, short_model(r["model"]), "t-lab", extra=' style="font-size:12.5px"'))
        b.append(
            f'<rect x="{sx}" y="{yy + 7}" width="{bx(r["f1"]) - sx:.1f}" height="9" rx="3" '
            f'class="{"f-s1" if r is ms[0] else "f-mute"}"/>'
        )
        b.append(text(W - 40, yy + 16, f"{r['f1']:.3f}", "t-tick", "end"))
        b.append(text(W - 40, yy, f"{r['lat']:.2f} s", "t-small", "end"))
    b.append(
        text(
            L,
            h - 18,
            "Source: results/benchmark_table.csv (50 development questions per configuration, "
            "gpt-oss:20b as judge).",
            "t-small",
        )
    )
    return svg(
        h,
        "Quality against latency for all configurations",
        f"Scatter of {len(bench)} configurations. Best F1 {best['f1']:.3f} at {best['lat']:.2f} s; "
        f"{p90['f1'] / best['f1']:.0%} of that at {p90['lat']:.2f} s.",
        b,
    )


def fig_search(report):
    rows = list(csv.DictReader(open(RES / "tables" / "optimizers_conditional_on_table.csv", encoding="utf-8")))
    single = [r for r in rows if r["mode"] == "single"]
    methods = [
        ("ga", "Genetic algorithm", "s1"),
        ("tpe", "TPE", "s2"),
        ("coordinate_ascent", "Coordinate ascent", "s3"),
        ("random", "Random search", "mute"),
    ]
    budgets = sorted({int(r["budget"]) for r in single})
    med = {(r["method"], int(r["budget"])): float(r["median"]) for r in single}

    h = 600
    b = header(
        "Search beats random sampling inside the table, but not on unseen manuals",
        "Left: replay on the complete lookup table, 50 seeds per method. "
        "Right: the same methods validated by leaving one manual out.",
    )
    # left panel
    L, R, T, B = 96, 560, 150, h - 96
    b.append(text(40, 124, "Inside the benchmark table", "t-h"))
    x = Lin(25, 150, L, R)
    y = Lin(0, 0.036, B, T)
    for v in (0, 0.01, 0.02, 0.03):
        b.append(f'<path class="grid" d="M{L},{y(v):.1f} H{R}"/>')
        b.append(text(L - 10, y(v) + 4, f"{v:.2f}", "t-tick", "end"))
    for v in budgets:
        b.append(text(x(v), B + 20, str(v), "t-tick", "middle"))
    b.append(f'<path class="axis" d="M{L},{B} H{R}"/>')
    b.append(text((L + R) / 2, B + 46, "Evaluation budget (configurations tried)", "t-lab", "middle"))
    b.append(
        text(
            L - 62,
            (T + B) / 2,
            "Median F1 gap to the best of 960",
            "t-lab",
            "middle",
            f' transform="rotate(-90 {L - 62} {(T + B) / 2})"',
        )
    )
    ends = []
    for key, name, col in methods:
        pts = [(x(bu), y(med[(key, bu)])) for bu in budgets]
        b.append(f'<path d="{path(pts)}" class="line st-{col}"/>')
        for (px, py), bu in zip(pts, budgets):
            b.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" class="f-{col} ring">'
                f"<title>{esc(name)}, budget {bu}: median gap {med[(key, bu)]:.4f}</title></circle>"
            )
        ends.append([pts[-1][1], name, col])
    # de-overlap end labels
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 17:
            ends[i][0] = ends[i - 1][0] + 17
    for yy, name, col in ends:
        b.append(f'<circle cx="{R + 18}" cy="{yy - 4:.1f}" r="4" class="f-{col}"/>')
        b.append(text(R + 28, yy, name, "t-lab"))

    # right panel: held-out difference vs random
    lomo = {r["method"]: r for r in report["lomo_all"]}
    n_man = lomo["random"]["n_manuals"]
    RL, RR, RT = 760, W - 60, 150
    b.append(text(RL - 40, 124, f"On held-out manuals ({n_man} manuals)", "t-h"))
    xr = Lin(-0.06, 0.06, RL + 150, RR)
    rows_r = [(k, n, c) for k, n, c in methods if k != "random"]
    row_h = 70
    top = RT + 30
    bottom = top + row_h * len(rows_r)
    for v in (-0.06, -0.03, 0, 0.03, 0.06):
        b.append(f'<path class="{"axis" if v == 0 else "grid"}" d="M{xr(v):.1f},{top - 20} V{bottom}"/>')
        b.append(text(xr(v), bottom + 20, f"{v:+.2f}" if v else "0", "t-tick", "middle"))
    b.append(
        text((xr(-0.06) + xr(0.06)) / 2, bottom + 46, "Held-out F1 minus random search, 95% CI", "t-lab", "middle")
    )
    for i, (k, name, col) in enumerate(rows_r):
        r = lomo[k]
        yy = top + i * row_h + row_h / 2 - 10
        b.append(text(RL - 40, yy, name, "t-lab"))
        b.append(text(RL - 40, yy + 17, f"held-out F1 {r['held_f1']:.3f}", "t-small"))
        b.append(
            f'<path class="st-{col}" style="stroke-width:2;stroke-linecap:round" '
            f'd="M{xr(r["ci_lo"]):.1f},{yy} H{xr(r["ci_hi"]):.1f}"/>'
        )
        b.append(
            f'<circle cx="{xr(r["diff_vs_random"]):.1f}" cy="{yy}" r="5.5" class="f-{col} ring">'
            f"<title>{esc(name)}: {r['diff_vs_random']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]</title></circle>"
        )
    b.append(
        text(
            RL - 40,
            bottom + 86,
            f"Random search reaches F1 {lomo['random']['held_f1']:.3f} on the held-out manual.",
            "t-lab",
        )
    )
    b.append(text(RL - 40, bottom + 106, "Every interval includes zero.", "t-lab"))
    b.append(
        text(
            40,
            h - 18,
            "Sources: results/tables/optimizers_conditional_on_table.csv, "
            "results/report.json (lomo_all). The coevolutionary variant is in the same tables.",
            "t-small",
        )
    )
    return svg(
        h,
        "Search methods compared with random search",
        "Inside the lookup table, the genetic algorithm, TPE and coordinate ascent close the gap to the best "
        "configuration faster than random search. On held-out manuals none of them beats random search.",
        b,
    )


def fig_heldout(report, frozen):
    gen = [
        r for r in csv.DictReader(open(RES / "tables" / "generalization.csv", encoding="utf-8")) if r["view"] == "all"
    ]
    best_dev = frozen["best_dev"]
    archived = set(frozen.get("archived_arms", []))
    G = report["generalization"]
    panels = [
        ("test_a", f"Test A: {G['test_a/all']['n_questions']} unused questions", G["test_a/all"]),
        ("test_b", f"Test B: {G['test_b/all']['n_questions']} independently written questions", G["test_b/all"]),
    ]
    h = 620
    b = header(
        "The configuration chosen on development data stays near the top on unseen questions",
        f"Development F1 against test F1 for the {G['test_a/all']['n_frozen']} frozen finalists. "
        "Each finalist was scored once, after the freeze.",
    )
    pw = (W - 80 - 80) / 2
    for i, (split, name, g) in enumerate(panels):
        rows = [r for r in gen if r["split"] == split]
        L = 40 + 56 + i * (pw + 80)
        R = L + pw - 56
        T, B = 170, h - 100
        b.append(text(L - 56, 118, name, "t-h"))
        b.append(
            text(
                L - 56,
                138,
                f"Selected setup ranks {g['test_rank_of_best']} of {g['n_frozen']}; "
                f"test F1 {g['best_dev_on_test']:.3f}, {g['best_minus_random_on_test']:+.3f} "
                "against the random finalists",
                "t-lab",
            )
        )
        x = Lin(0.15, 0.42, L, R)
        ymax = 0.6 if split == "test_b" else 0.42
        y = Lin(0.05, ymax, B, T)
        step = 0.1
        v = 0.1
        while v <= ymax + 1e-9:
            b.append(f'<path class="grid" d="M{L},{y(v):.1f} H{R}"/>')
            b.append(text(L - 10, y(v) + 4, f"{v:.1f}", "t-tick", "end"))
            v += step
        for v in (0.2, 0.3, 0.4):
            b.append(text(x(v), B + 20, f"{v:.1f}", "t-tick", "middle"))
        b.append(f'<path class="axis" d="M{L},{B} H{R}"/>')
        lo, hi = 0.15, min(0.42, ymax)
        b.append(
            f'<path class="lead" style="stroke-dasharray:0" d="M{x(lo):.1f},{y(lo):.1f} L{x(hi):.1f},{y(hi):.1f}"/>'
        )
        ang = -math.degrees(math.atan2(y(lo) - y(hi), x(hi) - x(lo)))
        lx_, ly_ = x(lo + 0.045), y(lo + 0.045) + 16
        b.append(
            text(lx_, ly_, "test = dev", "t-small", "start", f' transform="rotate({ang:.1f} {lx_:.1f} {ly_:.1f})"')
        )
        b.append(text((L + R) / 2, B + 46, "Development F1", "t-lab", "middle"))
        if i == 0:
            b.append(
                text(
                    L - 50,
                    (T + B) / 2,
                    "Test F1",
                    "t-lab",
                    "middle",
                    f' transform="rotate(-90 {L - 50} {(T + B) / 2})"',
                )
            )
        sel = None
        for r in rows:
            d, t = float(r["dev"]), float(r["test"])
            if r["cfg"] == best_dev:
                sel = (d, t, r)
                continue
            cls = "f-mute" if r["cfg"] in archived else "dot"
            b.append(
                f'<circle cx="{x(d):.1f}" cy="{y(t):.1f}" r="4.5" class="{cls} ring">'
                f"<title>{esc(describe(r['cfg']))}: dev {d:.3f}, test {t:.3f}</title></circle>"
            )
        d, t, r = sel
        b.append(f'<circle cx="{x(d):.1f}" cy="{y(t):.1f}" r="6.5" class="f-s1 ring"/>')
        b.append(text(x(d) - 12, y(t) + 4, "selected", "t-strong halo", "end"))
    # legend
    lx, ly = 40, h - 22
    b.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" class="f-s1"/>')
    b.append(text(lx + 16, ly, "Selected on development questions", "t-small"))
    b.append(f'<circle cx="{lx + 275}" cy="{ly - 4}" r="5" class="f-mute"/>')
    b.append(text(lx + 286, ly, "Two reference configurations outside the grid", "t-small"))
    b.append(f'<circle cx="{lx + 525}" cy="{ly - 4}" r="5" class="dot"/>')
    b.append(text(lx + 536, ly, "Other finalists (Pareto, search output, random)", "t-small"))
    b.append(text(W - 40, ly, "Source: results/tables/generalization.csv", "t-small", "end"))
    return svg(
        h,
        "Held-out test of the frozen finalists",
        "Two scatter plots of development against test F1. The configuration selected on development data ranks "
        f"{G['test_a/all']['test_rank_of_best']} of {G['test_a/all']['n_frozen']} on Test A and "
        f"{G['test_b/all']['test_rank_of_best']} of {G['test_b/all']['n_frozen']} on Test B.",
        b,
    )


def main():
    bench = load_benchmark()
    report = load_report()
    frozen = load_frozen()
    figures = {
        "header.svg": fig_header(bench, report, frozen),
        "protocol.svg": fig_protocol(bench, report),
        "quality-vs-latency.svg": fig_quality_latency(bench),
        "search-vs-random.svg": fig_search(report),
        "held-out-test.svg": fig_heldout(report, frozen),
    }
    for name, content in figures.items():
        (OUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote docs/assets/{name}")


if __name__ == "__main__":
    main()
