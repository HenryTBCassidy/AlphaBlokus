"""Render the profiling investigation results as a visual HTML report.

Turns the measured numbers from ``scripts/profile_self_play.py`` (P7 of
``docs/plans/profiling-investigation.md``) into charts you can actually look at,
with the raw cProfile evidence appended so every figure is traceable.

The DATA dicts below ARE the measurements — measured 2026-06-05 on the RTX 3060 Ti
at the production config (300 sims, K=16, conv 64f×4b, CUDA, F2 move-gen on),
via cProfile + the built-in MCTS phase timers + a synthetic-buffer training run.

Usage:  uv run python -m scripts.profile_report   ->  docs/research/profiling-report.html
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

_RESEARCH = Path(__file__).resolve().parent.parent / "docs" / "research"
OUT = _RESEARCH / "profiling-report.html"
PNG_DIR = _RESEARCH / "profiling"  # static charts for the GitHub-rendered markdown report

# ---- Measured data (source: scripts/profile_self_play.py on the PC, 2026-06-05) ----

# Full training cycle, per generation (15-gen x 1000-game config).
# self-play wall = measured gen-1 of the crashed run; training = synthetic-buffer
# measurement (12.3 s); arena+elo estimated from per-game cost x game counts.
FULL_CYCLE = [  # (phase, minutes_per_gen, helped_by_mcts_opt)
    ("Self-play (1000 games)", 19.5, True),
    ("Arena + Elo (50+50)", 2.0, True),
    ("Training (2 epochs, 57k ex)", 0.2, False),
    ("Diagnostics", 0.1, False),
]

# Inside one self-play game (7.93 s measured), % of game wall.
SELF_PLAY = [  # (slice, pct)
    ("NN inference", 33),
    ("Move generation (F2)", 24),
    ("_select_action (UCB loop)", 22),
    ("Board transitions (copy)", 10),
    ("Tree descend/expand/backprop", 10),
    ("Episode overhead", 1),
]

# Decomposition of the 33% NN-inference slice — the surprise: transfers > compute.
INFERENCE = [  # (component, pct_of_self_play)
    (".cpu() transfer (GPU→CPU)", 14),
    (".cuda() transfer (CPU→GPU)", 3),
    ("compute (conv/bn/linear/relu)", 7),
    ("other torch overhead", 9),
]

# Amdahl: whole-self-play speedup if a slice is driven to zero (1 / (1 - p)).
AMDAHL = [  # (lever, slice_pct, effort)
    ("_select_action (UCB)", 22, "Low"),
    ("Move-gen (bitboard)", 24, "High"),
    ("Inference transfers", 17, "Medium"),
    ("All NN inference", 33, "—"),
    ("Training (of full cycle)", 1, "—"),
]

# Raw cProfile top functions (1 game, 300 sims, F2 on; total 8.941 s) — the evidence.
CPROFILE_RAW = """\
   ncalls  tottime  cumtime  function
    57224    1.788    1.842  movegen_runtime.py:201(_fill_legal_actions_for_anchor)   <- move-gen core
    14444    1.695    1.938  mcts.py:346(_select_action)                              <- UCB loop (pure Python)
      888    1.242    1.242  {method 'cpu' of torch.TensorBase}                       <- GPU->CPU transfer
     4884    0.369    0.369  {built-in method torch.conv2d}                           <- actual NN compute
      444    0.292    0.292  {method 'cuda' of torch.TensorBase}                      <- CPU->GPU transfer
     6746    0.247    2.531  mcts.py:388(_expand_leaf)
    64794    0.209    0.332  game.py:238(_placements_at_point)
   218731    0.208    0.208  game.py:208(_all_cells_valid)
    14162    0.127    0.823  board.py:228(with_piece)                                 <- board copy
  2518748    0.100    0.100  {built-in method math.sqrt}                              <- UCB, 2.5M calls/game
cumulative: predict_batch 2.927 (33%) | valid_move_mask 2.188 (24%) | _select_action 1.938 (22%)
"""

BLUE, GREEN, AMBER, GREY, RED = "#2563eb", "#16a34a", "#d97706", "#9ca3af", "#dc2626"


def _bar(labels, values, colors, title, xtitle, texts=None):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=texts or [f"{v}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title, xaxis_title=xtitle, height=360,
        margin=dict(l=10, r=30, t=50, b=40), yaxis=dict(autorange="reversed"),
        template="plotly_white", font=dict(size=13),
    )
    return fig


def build() -> None:
    total_min = sum(m for _, m, _ in FULL_CYCLE)

    # 1. Full-cycle phase split
    labels = [p for p, _, _ in FULL_CYCLE]
    mins = [m for _, m, _ in FULL_CYCLE]
    colors = [GREEN if h else GREY for _, _, h in FULL_CYCLE]
    pct = [m / total_min * 100 for m in mins]
    f1 = _bar(labels, [round(p, 1) for p in pct], colors,
              f"Full training cycle — where a generation's ~{total_min:.0f} min goes "
              "(green = sped up by MCTS work)", "% of generation",
              texts=[f"{p:.1f}%  ({m:.1f} min)" for p, m in zip(pct, mins, strict=True)])

    # 2. Self-play breakdown
    f2 = _bar([s for s, _ in SELF_PLAY], [p for _, p in SELF_PLAY],
              [BLUE, GREEN, AMBER, GREY, GREY, GREY],
              "Inside one self-play game (7.93 s) — % of game", "% of self-play",
              texts=[f"{p}%" for _, p in SELF_PLAY])

    # 3. Inference decomposition
    icolors = [RED, RED, GREEN, GREY]
    f3 = _bar([c for c, _ in INFERENCE], [p for _, p in INFERENCE], icolors,
              "The 33% inference slice — transfers (red) cost ~3× the compute (green)",
              "% of self-play", texts=[f"{p}%" for _, p in INFERENCE])

    # 4. Amdahl ceilings
    ceil = [round(1 / (1 - p / 100), 2) for _, p, _ in AMDAHL]
    f4 = _bar([f"{lv}  ({eff})" for lv, _, eff in AMDAHL], ceil,
              [AMBER, RED, BLUE, BLUE, GREY],
              "Amdahl ceiling — max whole-cycle speedup if that slice → 0",
              "× speedup ceiling", texts=[f"{c}×" for c in ceil])

    # Static PNGs for the markdown report (renders on GitHub, unlike the HTML).
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in [("1-full-cycle", f1), ("2-self-play", f2),
                      ("3-inference", f3), ("4-amdahl", f4)]:
        fig.write_image(str(PNG_DIR / f"{name}.png"), width=900, height=360, scale=2)
    print(f"wrote 4 PNGs to {PNG_DIR}")

    parts = []
    for i, fig in enumerate([f1, f2, f3, f4]):
        parts.append(fig.to_html(full_html=False,
                                 include_plotlyjs="cdn" if i == 0 else False))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AlphaBlokus — Profiling Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;
       margin:32px auto;padding:0 20px;color:#111;line-height:1.5}}
 h1{{margin-bottom:4px}} .sub{{color:#555;margin-top:0}}
 .box{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;margin:18px 0}}
 .verdict{{background:#ecfdf5;border-color:#a7f3d0}}
 pre{{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:8px;overflow-x:auto;font-size:12px}}
 code{{background:#eef;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>AlphaBlokus — Training-Cycle Profiling Report</h1>
<p class="sub">Measured 2026-06-05 on the RTX 3060 Ti at the production config
(300 sims, K=16, conv 64f×4b, CUDA, F2 move-gen on) via <code>scripts/profile_self_play.py</code>
— cProfile + built-in MCTS phase timers + a synthetic-buffer training run.</p>

<div class="box"><b>Headline.</b> Self-play is <b>CPU-bound</b> (GPU idles ~0–30%). No single
bottleneck — inference, move-gen, and the UCB loop are three co-equal thirds. Training is
~1% of the cycle. Surprise: inference cost is dominated by GPU↔CPU <b>transfers</b>, not compute.</div>

<h2>1. Where a generation's time goes</h2>{parts[0]}
<h2>2. Inside one self-play game</h2>{parts[1]}
<h2>3. The inference slice — transfers vs compute</h2>{parts[2]}
<h2>4. Amdahl ceilings — how much each lever could ever buy</h2>{parts[3]}

<div class="box verdict"><b>Verdict (P8): RUN NOW.</b> The 15-gen run projects to ~5.3 h —
comfortably overnight. Don't optimise training (~1%, ceiling 1.01×). When tighter iteration
is wanted, the ranked levers are: <b>1)</b> vectorise <code>_select_action</code> (~1.25×, low effort),
<b>2)</b> cut inference transfer overhead (~1.2×, medium), <b>3)</b> bitboard move-gen (~1.3×, high).</div>

<h2>Raw evidence — cProfile top functions (1 game, 300 sims, F2 on; total 8.941 s)</h2>
<pre>{CPROFILE_RAW}</pre>
<p class="sub">Source: <code>docs/plans/profiling-investigation.md</code> (P7/P8) ·
harness <code>scripts/profile_self_play.py</code> · this page <code>scripts/profile_report.py</code>.</p>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
