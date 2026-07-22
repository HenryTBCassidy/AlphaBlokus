"""Capacity probe: supervised `large` vs `xl` fit on a frozen self-play buffer (P6/P8).

The decisive, free experiment behind the `xl` decision
(docs/research/regression-and-next-steps.md §3.4): train each candidate
architecture *supervised* on the same frozen buffer of gen-40 self-play games,
with a game-level held-out split, and compare **held-out policy CE + value
MSE**. If `xl` fits the same data clearly better out-of-sample, the targets
contain structure `large` cannot absorb — capacity is binding and the paid
`xl` run (P9) is justified on evidence. A clear tie finally grounds the A4
demotion properly (→ P10, Pentobi distillation).

Pre-registered verdict rule (docs/plans/post-regression-recovery.md P8):
`xl` best CE ≤ `large`'s − 0.03 nats → capacity_binding; gap < 0.01 → tie;
between → ambiguous (default to P9, which is config-only).

One command on the box (GPU, no cloud), e.g.::

    uv run python scripts/capacity_probe.py \
        --config run_configurations/blokus_cloud_v2.json \
        --history-dir temp/runs/blokus/<run>/SelfPlayHistory --file-indices 16 \
        --arms large,xl --max-epochs 20 --out temp/benchmarks/capacity_probe.json

If no ``SelfPlayHistory`` parquets exist locally, regenerate ~10k games with
the gen-40 net first (~1 h on the box) — the probe only needs the parquet, not
a training run. The optional ``large-warm`` arm (``--warm-start <ckpt>``)
answers a secondary question: does gen-40 warm-start still have SL headroom on
frozen data? (Relevant to P10's design.)
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from alphablokus.config import NET_PRESETS, RunConfig, load_args
from alphablokus.registry import instantiate_game_and_network
from alphablokus.storage.selfplay_store import SelfPlayStore
from alphablokus.training.holdout import HoldoutMetrics, evaluate_holdout, split_games_holdout

if TYPE_CHECKING:
    from alphablokus.interfaces import IGame, INeuralNetWrapper
    from alphablokus.selfplay.episode import GameExamples, ProcessedExample

# P8's pre-registered verdict thresholds, in nats of held-out policy CE.
CAPACITY_BINDING_DELTA = 0.03
TIE_DELTA = 0.01

KNOWN_ARMS = ("large", "xl", "large-warm")


def _parse_indices(spec: str) -> list[int]:
    """Parse a file-index spec — ``"16"``, ``"12-16"``, or ``"12,14,16"``."""
    indices: list[int] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        start, _, end = token.partition("-")
        if end:
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(start))
    if not indices:
        raise ValueError(f"Bad file-indices spec {spec!r} (e.g. '16', '12-16', '12,14').")
    return indices


def _load_games(history_dir: Path, indices: list[int]) -> list[GameExamples]:
    store = SelfPlayStore(history_dir)
    games: list[GameExamples] = []
    for index in indices:
        loaded = store.load_games(index)
        if loaded is None:
            raise FileNotFoundError(f"{history_dir}/self_play_{index}.parquet not found")
        games.extend(loaded)
    return games


def _arm_config(base: RunConfig, preset: str, args: argparse.Namespace) -> RunConfig:
    """Base run config re-shaped for one supervised probe arm."""
    net_config = replace(
        base.net_config,
        preset=preset,
        num_filters=NET_PRESETS[preset]["num_filters"],
        num_residual_blocks=NET_PRESETS[preset]["num_residual_blocks"],
        learning_rate=args.lr,
        epochs=1,  # train() is called once per probe epoch so eval runs between passes
        batch_size=args.batch_size,
        lr_scheduler="constant",
    )
    return replace(base, net_config=net_config)


def _seed_everything(seed: int) -> None:
    """Identical init RNG per arm construction (weight init aside, arms differ only in architecture)."""
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release(wrapper: INeuralNetWrapper) -> None:
    """Free one arm's GPU memory before the next arm builds."""
    import torch

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_arm(
    name: str,
    config: RunConfig,
    args: argparse.Namespace,
    train_flat: list[ProcessedExample],
    holdout_flat: list[ProcessedExample],
) -> dict[str, Any]:
    """Train one architecture to (early-stopped) asymptote; return its curve + best."""
    preset = "large" if name == "large-warm" else name
    _seed_everything(args.seed)
    game: IGame
    game, wrapper = instantiate_game_and_network(_arm_config(config, preset, args))
    if name == "large-warm":
        wrapper.load_weights(str(Path(args.warm_start).resolve()))
        logger.info("Arm {}: warm-started from {}", name, args.warm_start)

    def evaluate() -> HoldoutMetrics:
        return evaluate_holdout(
            wrapper,  # type: ignore[arg-type]  # BaseNNetWrapper satisfies SupportsEncodedPrediction
            holdout_flat,
            encode_fn=game.encode_compact,
            action_size=game.get_action_size(),
            batch_size=args.eval_batch_size,
        )

    curve: list[dict[str, Any]] = []
    baseline = evaluate()
    curve.append({"epoch": 0, **asdict(baseline)})
    logger.info("Arm {} epoch 0 (baseline): CE {:.4f}, value MSE {:.4f}", name, baseline.policy_ce, baseline.value_mse)

    best = baseline
    best_epoch = 0
    epochs_since_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        wrapper.train(train_flat, generation=epoch, metrics=None, eval_set=None)
        metrics = evaluate()
        curve.append({"epoch": epoch, **asdict(metrics)})
        logger.info(
            "Arm {} epoch {}: CE {:.4f} (KL {:.4f}), value MSE {:.4f}",
            name,
            epoch,
            metrics.policy_ce,
            metrics.policy_kl,
            metrics.value_mse,
        )
        if metrics.policy_ce <= best.policy_ce - args.min_delta:
            best, best_epoch, epochs_since_improvement = metrics, epoch, 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logger.info("Arm {}: early stop at epoch {} (patience {})", name, epoch, args.patience)
                break

    num_params = sum(p.numel() for p in wrapper.nnet.parameters())  # type: ignore[attr-defined]
    _release(wrapper)
    return {
        "preset": preset,
        "warm_start": args.warm_start if name == "large-warm" else None,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best": asdict(best),
        "curve": curve,
    }


def _verdict(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply P8's pre-registered rule to the large-vs-xl best held-out CE."""
    if "large" not in arms or "xl" not in arms:
        return {"verdict": "incomplete", "note": "needs both 'large' and 'xl' arms"}
    delta = arms["large"]["best"]["policy_ce"] - arms["xl"]["best"]["policy_ce"]
    if delta >= CAPACITY_BINDING_DELTA:
        verdict = "capacity_binding"  # → P9: launch the xl run
    elif abs(delta) < TIE_DELTA:
        verdict = "tie"  # → P10: the A4 demotion stands; go distillation
    else:
        verdict = "ambiguous"  # → default to P9 (config-only) per the plan
    return {
        "verdict": verdict,
        "ce_delta_large_minus_xl": delta,
        "rule": f"binding ≥ {CAPACITY_BINDING_DELTA}, tie < {TIE_DELTA} (nats held-out policy CE)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised large-vs-xl capacity probe on a frozen buffer")
    parser.add_argument("--config", required=True, help="Base run config JSON (game + dirs + perf knobs)")
    parser.add_argument("--history-dir", type=Path, required=True, help="SelfPlayHistory directory with parquets")
    parser.add_argument("--file-indices", required=True, help="self_play_<i>.parquet indices: '16', '12-16', '12,14'")
    parser.add_argument("--arms", default="large,xl", help=f"Comma list from {KNOWN_ARMS} (default large,xl)")
    parser.add_argument("--warm-start", default=None, help="Checkpoint path for the large-warm arm")
    parser.add_argument("--holdout-frac", type=float, default=0.05, help="Fraction of games held out (default 0.05)")
    parser.add_argument("--seed", type=int, default=7, help="Split + init seed (default 7)")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max full passes per arm (default 20)")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop patience in epochs (default 3)")
    parser.add_argument("--min-delta", type=float, default=0.002, help="CE improvement that resets patience (nats)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Probe learning rate (default 1e-3, constant)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("temp/benchmarks/capacity_probe.json"))
    args = parser.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"Unknown arms {unknown}; expected a subset of {KNOWN_ARMS}.")
    if "large-warm" in arm_names and not args.warm_start:
        raise SystemExit("--warm-start <checkpoint> is required for the large-warm arm.")

    config = load_args(args.config)
    indices = _parse_indices(args.file_indices)
    games = _load_games(args.history_dir, indices)
    train_games, holdout_games = split_games_holdout(games, args.holdout_frac, args.seed)
    train_flat = [example for game_examples in train_games for example in game_examples]
    holdout_flat = [example for game_examples in holdout_games for example in game_examples]
    logger.info(
        "Probe data: {} games → train {} games ({} positions) / holdout {} games ({} positions)",
        len(games),
        len(train_games),
        len(train_flat),
        len(holdout_games),
        len(holdout_flat),
    )

    arms: dict[str, dict[str, Any]] = {}
    for name in arm_names:
        logger.info("=== Arm {} ===", name)
        arms[name] = _run_arm(name, config, args, train_flat, holdout_flat)

    payload = {
        "config": args.config,
        "history_dir": str(args.history_dir),
        "file_indices": indices,
        "holdout_fraction": args.holdout_frac,
        "seed": args.seed,
        "lr": args.lr,
        "timestamp": datetime.now(UTC).isoformat(),
        "arms": arms,
        "verdict": _verdict(arms),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Verdict: {} → {}", payload["verdict"].get("verdict"), args.out)


if __name__ == "__main__":
    main()
