"""J5/J6 throughput benchmark for the JAX spike kernels.

Measures, with jit warm-up excluded and ``block_until_ready`` around every
timed region:

1. ``legal_mask_batch`` throughput (masks/s) over a batch-size sweep, on
   states drawn from the dev-5000 cache (mixed game phases).
2. Full random self-play rollouts — a jitted ``fori_loop`` over a fixed ply
   horizon of mask -> masked-categorical sample -> step, vmapped over the
   batch. Reports board-steps/s and completed games/s.
3. Python baselines on the same machine: F2/numba ``valid_move_mask`` per-call
   latency over dev-cache positions, and a single-process Python random
   rollout (F2 enabled) in games/s.
4. (J6, ``--net``) the same rollout loop with a dummy run3-shaped ResNet
   (44ch -> 128f x 8 blocks, conv policy head) choosing moves: the inner loop
   of a batched self-play actor minus tree search. fp32 and bf16 variants.

Usage::

    uv run python -m experiments.jax_spike.benchmark            # CPU dev box
    uv run python -m experiments.jax_spike.benchmark --net      # + J6 net loop
    uv run python -m experiments.jax_spike.benchmark --out temp/benchmarks/jax_spike.html

Writes an HTML report (charts inlined base64, ``scripts/benchmark.py``
conventions) and logs a summary table.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.jax_spike.bridge import numpy_state_from_board
from experiments.jax_spike.kernels import GameState, JaxKernels, make_kernels
from experiments.jax_spike.tables import JaxTables, build_jax_tables
from games.blokusduo.game import BlokusDuoGame

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIECES_PATH = REPO_ROOT / "games" / "blokusduo" / "pieces.json"
DEV_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "blokus_duo_positions" / "dev_5000.npz"

#: Fixed rollout horizon. Random Duo games average ~32 plies; terminal states
#: absorb (pass-only mask, pass step is a no-op), so a generous bound costs
#: linear time but never correctness. Completion fraction is reported.
MAX_PLIES = 72

#: Timed repetitions per measurement; the median is reported.
REPS = 5

DEFAULT_BATCH_SIZES = (64, 256, 1024, 4096, 8192)

#: run3 net shape for the J6 dummy net.
NET_FILTERS = 128
NET_BLOCKS = 8


@dataclass
class Measurement:
    """One timed configuration."""

    name: str
    batch_size: int
    seconds: float
    items_per_second: float
    detail: dict = field(default_factory=dict)


@dataclass
class Report:
    device: str
    jax_version: str
    git_commit: str
    timestamp: str
    measurements: list[Measurement] = field(default_factory=list)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _median_time(fn, reps: int = REPS) -> float:
    """Median wall time of ``fn()`` over ``reps`` runs (fn must block)."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _load_cache_states(game: BlokusDuoGame, limit: int = 5000) -> GameState:
    """Stack dev-cache positions into one big numpy GameState batch."""
    from tests.fixtures.blokus_positions import iter_cached_positions

    rows = []
    for index, (board, player, _sequence) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if index >= limit:
            break
        rows.append(numpy_state_from_board(board, player))
    return GameState(*(np.stack([row[i] for row in rows]) for i in range(4)))


def _tile_states(states: GameState, batch_size: int) -> GameState:
    """Cycle cache states up/down to exactly ``batch_size`` rows."""
    n = states.ppb.shape[0]
    idx = np.arange(batch_size) % n
    return GameState(*(np.asarray(a)[idx] for a in states))


# ---------------------------------------------------------------------------
# JAX measurements
# ---------------------------------------------------------------------------

def measure_mask_throughput(
    kernels: JaxKernels, cache_states: GameState, batch_sizes: tuple[int, ...],
) -> list[Measurement]:
    import jax

    results = []
    for batch_size in batch_sizes:
        batch = jax.device_put(_tile_states(cache_states, batch_size))
        out = kernels.legal_mask_batch(batch)  # warm-up / compile
        out.block_until_ready()
        seconds = _median_time(lambda b=batch: kernels.legal_mask_batch(b).block_until_ready())
        results.append(Measurement(
            name="legal_mask", batch_size=batch_size, seconds=seconds,
            items_per_second=batch_size / seconds,
        ))
        logger.info("legal_mask  B={:>5}: {:>12,.0f} masks/s", batch_size, batch_size / seconds)
    return results


def make_random_rollout(kernels: JaxKernels, max_plies: int = MAX_PLIES):
    """Jitted batched random self-play: mask -> categorical sample -> step."""
    from functools import partial

    import jax
    import jax.numpy as jnp

    def initial_batch(batch_size: int) -> GameState:
        single = kernels.initial_state()
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), single)

    @partial(jax.jit, static_argnums=(1,))
    def rollout(key: jax.Array, batch_size: int) -> GameState:
        state = initial_batch(batch_size)

        def body(ply: int, carry):
            state, key = carry
            key, subkey = jax.random.split(key)
            masks = kernels.legal_mask_batch(state)
            logits = jnp.where(masks, 0.0, -jnp.inf)
            actions = jax.random.categorical(subkey, logits, axis=-1)
            return kernels.step_batch(state, actions), key

        state, _ = jax.lax.fori_loop(0, max_plies, body, (state, key))
        return state

    return rollout


def measure_random_rollout(
    kernels: JaxKernels, batch_sizes: tuple[int, ...],
) -> list[Measurement]:
    import jax

    rollout = make_random_rollout(kernels)
    results = []
    for batch_size in batch_sizes:
        key = jax.random.PRNGKey(42)
        final = rollout(key, batch_size)  # warm-up / compile
        jax.block_until_ready(final)
        completed = float(np.mean(np.asarray(kernels.game_result_batch(final, np.int8(1))) != 0.0))
        seconds = _median_time(lambda k=key, b=batch_size: jax.block_until_ready(rollout(k, b)))
        steps_per_second = batch_size * MAX_PLIES / seconds
        games_per_second = batch_size * completed / seconds
        results.append(Measurement(
            name="random_rollout", batch_size=batch_size, seconds=seconds,
            items_per_second=steps_per_second,
            detail={"games_per_second": games_per_second, "completed_fraction": completed},
        ))
        logger.info(
            "rollout     B={:>5}: {:>12,.0f} steps/s, {:>9,.1f} games/s (completed {:.1%})",
            batch_size, steps_per_second, games_per_second, completed,
        )
    return results


# ---------------------------------------------------------------------------
# J6: dummy net loop
# ---------------------------------------------------------------------------

def _init_dummy_net_params(key, filters: int, blocks: int, action_size: int, dtype):
    """Random weights for a run3-shaped ResNet in plain jnp arrays."""
    import jax

    keys = iter(jax.random.split(key, 4 + 2 * blocks + 2))
    scale = 0.05

    def conv(k, cin, cout, ksize):
        return (jax.random.normal(k, (cout, cin, ksize, ksize), dtype) * scale)

    params = {
        "trunk": conv(next(keys), 44, filters, 3),
        "blocks": [
            (conv(next(keys), filters, filters, 3), conv(next(keys), filters, filters, 3))
            for _ in range(blocks)
        ],
        # Conv policy head: 1x1 conv to 91 orientation planes + pass logit.
        "policy": conv(next(keys), filters, 91, 1),
        "pass_head": jax.random.normal(next(keys), (filters,), dtype) * scale,
        "value": jax.random.normal(next(keys), (filters,), dtype) * scale,
    }
    del action_size
    return params


def make_net_rollout(
    kernels: JaxKernels, tables: JaxTables, dtype_name: str, max_plies: int = MAX_PLIES,
):
    """Batched pseudo-self-play: mask -> net forward -> masked sample -> step.

    The net is run3-shaped (44ch -> 128f x 8 residual blocks, 1x1-conv policy
    head to 91 orientation planes + pass logit, scalar value head) with random
    weights — throughput measurement only. This is the inner loop of a batched
    self-play actor minus tree search, i.e. the optimistic per-node cost of a
    future mctx pipeline.
    """
    from functools import partial

    import jax
    import jax.numpy as jnp

    dtype = jnp.float32 if dtype_name == "fp32" else jnp.bfloat16
    n = 14
    params = _init_dummy_net_params(jax.random.PRNGKey(7), NET_FILTERS, NET_BLOCKS, kernels.action_size, dtype)
    piece_planes = jnp.arange(1, 22, dtype=jnp.int8)  # (21,)

    def encode(ppb: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
        """44-channel encoding straight from the signed board (canonical)."""
        signed = (ppb * player).reshape(n, n)  # current player's pieces positive
        own = (signed[None, :, :] == piece_planes[:, None, None])
        opp = (signed[None, :, :] == -piece_planes[:, None, None])
        aggregates = jnp.stack([signed > 0, signed < 0])
        return jnp.concatenate([own, opp, aggregates]).astype(dtype)

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """x: (B, 44, 14, 14) -> (policy logits (B, A), value (B,))."""
        dn = ("NCHW", "OIHW", "NCHW")

        def conv(x, w):
            return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=dn)

        h = jax.nn.relu(conv(x, params["trunk"]))
        for w1, w2 in params["blocks"]:
            h = jax.nn.relu(h + conv(jax.nn.relu(conv(h, w1)), w2))
        planes = conv(h, params["policy"])  # (B, 91, 14, 14)
        placement_logits = planes.reshape(planes.shape[0], -1)  # (B, 91*196)
        pooled = h.mean(axis=(2, 3))  # (B, filters)
        pass_logit = (pooled * params["pass_head"]).sum(axis=1, keepdims=True)
        value = jnp.tanh((pooled * params["value"]).sum(axis=1))
        # NOTE: plane order vs ActionCodec order differs by a fixed permutation
        # (see games/blokusduo/neuralnets/net.py::build_action_permutation);
        # irrelevant for throughput, omitted here.
        logits = jnp.concatenate([placement_logits, pass_logit], axis=1)
        return logits, value

    def initial_batch(batch_size: int) -> GameState:
        single = kernels.initial_state()
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), single)

    @partial(jax.jit, static_argnums=(1,))
    def rollout(key: jax.Array, batch_size: int) -> GameState:
        state = initial_batch(batch_size)

        def body(ply: int, carry):
            state, key = carry
            key, subkey = jax.random.split(key)
            masks = kernels.legal_mask_batch(state)
            x = jax.vmap(encode)(state.ppb, state.current_player)
            logits, _value = forward(x)
            masked = jnp.where(masks, logits.astype(jnp.float32), -jnp.inf)
            actions = jax.random.categorical(subkey, masked, axis=-1)
            return kernels.step_batch(state, actions), key

        state, _ = jax.lax.fori_loop(0, max_plies, body, (state, key))
        return state

    @partial(jax.jit, static_argnums=(1,))
    def forward_only(key: jax.Array, batch_size: int) -> jnp.ndarray:
        x = jax.random.normal(key, (batch_size, 44, n, n), dtype)
        logits, value = forward(x)
        return logits.sum() + value.sum()

    return rollout, forward_only


def measure_net_rollout(
    kernels: JaxKernels, tables: JaxTables, batch_sizes: tuple[int, ...], dtype_name: str,
) -> list[Measurement]:
    import jax

    rollout, forward_only = make_net_rollout(kernels, tables, dtype_name)
    results = []
    for batch_size in batch_sizes:
        key = jax.random.PRNGKey(42)
        try:
            final = rollout(key, batch_size)
            jax.block_until_ready(final)
        except Exception as error:  # noqa: BLE001 — XlaRuntimeError (OOM) has no stable import path
            logger.warning(
                "net-rollout[{}] B={} SKIPPED ({}: {})",
                dtype_name, batch_size, type(error).__name__, str(error).splitlines()[0][:120],
            )
            _measure_forward_only(forward_only, batch_size, dtype_name, results)
            continue
        completed = float(np.mean(np.asarray(kernels.game_result_batch(final, np.int8(1))) != 0.0))
        seconds = _median_time(lambda k=key, b=batch_size: jax.block_until_ready(rollout(k, b)))
        steps_per_second = batch_size * MAX_PLIES / seconds
        results.append(Measurement(
            name=f"net_rollout_{dtype_name}", batch_size=batch_size, seconds=seconds,
            items_per_second=steps_per_second,
            detail={
                "games_per_second": batch_size * completed / seconds,
                "completed_fraction": completed,
            },
        ))
        logger.info(
            "net-rollout[{}] B={:>5}: {:>12,.0f} steps/s ({:>8,.1f} games/s)",
            dtype_name, batch_size, steps_per_second, batch_size * completed / seconds,
        )
        _measure_forward_only(forward_only, batch_size, dtype_name, results)
    return results


def _measure_forward_only(forward_only, batch_size: int, dtype_name: str, results: list[Measurement]) -> None:
    """Time the bare net forward at ``batch_size``, skipping on OOM."""
    import jax

    key = jax.random.PRNGKey(42)
    try:
        forward_only(key, batch_size)  # compile
        fwd_seconds = _median_time(lambda k=key, b=batch_size: jax.block_until_ready(forward_only(k, b)))
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "net-forward[{}] B={} SKIPPED ({}: {})",
            dtype_name, batch_size, type(error).__name__, str(error).splitlines()[0][:120],
        )
        return
    results.append(Measurement(
        name=f"net_forward_{dtype_name}", batch_size=batch_size, seconds=fwd_seconds,
        items_per_second=batch_size / fwd_seconds,
    ))
    logger.info(
        "net-forward[{}] B={:>5}: {:>12,.0f} forwards/s",
        dtype_name, batch_size, batch_size / fwd_seconds,
    )


# ---------------------------------------------------------------------------
# Python baselines
# ---------------------------------------------------------------------------

def measure_python_mask_baseline(game: BlokusDuoGame, sample: int = 500) -> Measurement:
    """Per-call latency of the production F2/numba mask over dev positions."""
    from games.blokusduo.movegen_runtime import get_default_generator
    from tests.fixtures.blokus_positions import iter_cached_positions

    generator = get_default_generator()
    positions = []
    for index, (board, player, _sequence) in enumerate(iter_cached_positions(game, DEV_CACHE_PATH)):
        if index >= sample:
            break
        positions.append((board, player))
    generator.valid_move_mask(game, positions[0][0], positions[0][1])  # numba warm-up
    t0 = time.perf_counter()
    for board, player in positions:
        generator.valid_move_mask(game, board, player)
    seconds = time.perf_counter() - t0
    per_second = len(positions) / seconds
    logger.info("python F2/numba mask: {:>10,.0f} masks/s (single process)", per_second)
    return Measurement(name="python_f2_mask", batch_size=1, seconds=seconds, items_per_second=per_second)


def measure_python_rollout_baseline(game: BlokusDuoGame, games: int = 20) -> Measurement:
    """Single-process Python random self-play (F2 enabled), games/s + steps/s."""
    rng = np.random.default_rng(42)
    total_plies = 0
    t0 = time.perf_counter()
    for _ in range(games):
        board = game.initialise_board()
        player = 1
        while game.get_game_ended(board, player) == 0:
            mask = game.valid_move_masking(board, player)
            action = int(rng.choice(np.flatnonzero(mask > 0)))
            board, player = game.get_next_state(board, player, action)
            total_plies += 1
    seconds = time.perf_counter() - t0
    logger.info(
        "python rollout: {:.2f} games/s, {:,.0f} steps/s (single process)",
        games / seconds, total_plies / seconds,
    )
    return Measurement(
        name="python_rollout", batch_size=1, seconds=seconds,
        items_per_second=total_plies / seconds,
        detail={"games_per_second": games / seconds, "games": games},
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fig_b64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_chart(report: Report) -> str:
    """Return an ``<img>`` tag with the throughput chart, or '' if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — report will have no chart")
        return ""

    by_name: dict[str, list[Measurement]] = {}
    for m in report.measurements:
        by_name.setdefault(m.name, []).append(m)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, ms in by_name.items():
        if name.startswith("python"):
            continue
        xs = [m.batch_size for m in ms]
        ys = [m.items_per_second for m in ms]
        ax.plot(xs, ys, marker="o", label=name)
    for name, style in (("python_f2_mask", ":"), ("python_rollout", "--")):
        if name in by_name:
            ax.axhline(by_name[name][0].items_per_second, linestyle=style, color="grey", label=f"{name} (1 proc)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("items/s (masks, board-steps, forwards)")
    ax.set_title(f"JAX spike throughput — {report.device}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return f'<img src="data:image/png;base64,{_fig_b64(fig)}" alt="throughput chart">'


def render_html(report: Report, out_path: Path) -> None:
    chart_tag = _render_chart(report)

    rows = "\n".join(
        f"<tr><td>{m.name}</td><td>{m.batch_size}</td><td>{m.seconds:.4f}</td>"
        f"<td>{m.items_per_second:,.0f}</td><td>{json.dumps(m.detail)}</td></tr>"
        for m in report.measurements
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>JAX spike benchmark</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}th{{background:#eee}}</style></head>
<body>
<h1>JAX spike benchmark</h1>
<p>device: <b>{report.device}</b> | jax {report.jax_version} | commit {report.git_commit} | {report.timestamp}</p>
{chart_tag}
<table><tr><th>measurement</th><th>batch</th><th>median s</th><th>items/s</th><th>detail</th></tr>
{rows}</table>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    logger.info("HTML report written to {}", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--net", action="store_true", help="include the J6 dummy-net rollout")
    parser.add_argument("--no-python-baselines", action="store_true")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import jax

    device = jax.devices()[0]
    device_label = f"{device.platform}:{device.device_kind}"
    logger.info("jax {} on {}", jax.__version__, device_label)

    batch_sizes = tuple(args.batch_sizes) if args.batch_sizes else DEFAULT_BATCH_SIZES
    if device.platform == "cpu" and args.batch_sizes is None:
        batch_sizes = (64, 256, 1024)  # CPU sweep — big batches just repeat the answer slowly

    game = BlokusDuoGame(pieces_config_path=PIECES_PATH)
    tables = build_jax_tables(game)
    kernels = make_kernels(tables)
    cache_states = _load_cache_states(game)

    report = Report(
        device=device_label,
        jax_version=jax.__version__,
        git_commit=_git_commit(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    report.measurements += measure_mask_throughput(kernels, cache_states, batch_sizes)
    report.measurements += measure_random_rollout(kernels, batch_sizes)
    if args.net:
        if device.platform == "cpu":
            logger.warning(
                "--net on CPU: XLA-CPU schedules convs inside fori_loop poorly (~40x slower than "
                "the bare forward) and bf16 may be emulated — numbers are not meaningful. "
                "The net rollout is a GPU measurement."
            )
        for dtype_name in ("fp32", "bf16"):
            report.measurements += measure_net_rollout(kernels, tables, batch_sizes, dtype_name)
    if not args.no_python_baselines:
        game_f2 = BlokusDuoGame(pieces_config_path=PIECES_PATH)
        game_f2.enable_optimised_movegen()
        report.measurements.append(measure_python_mask_baseline(game_f2))
        report.measurements.append(measure_python_rollout_baseline(game_f2))

    out = args.out or REPO_ROOT / "temp" / "benchmarks" / f"jax_spike_{_git_commit()}_{device.platform}.html"
    render_html(report, out)
    (out.with_suffix(".json")).write_text(json.dumps(asdict(report), indent=2))
    logger.info("JSON written to {}", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
