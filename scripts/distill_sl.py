"""SL distillation trainer: fine-tune (or freshly fit) a net on the Pentobi corpus (D7).

The training half of the Pentobi-distillation plan
(``docs/plans/pentobi-distillation.md`` D6/D7): supervised behavioural cloning on the
expert corpus — policy cross-entropy against Pentobi's move (label-smoothed over legal
moves at batch-build time, plan D6) + value MSE against the game outcome. ``margin`` is
stored in the corpus for a later margin-aware experiment but deliberately **not** part
of the v1 loss.

Two arms, one config each:

- ``warm``: fine-tune the current best net (v3 gen-40; weights only via
  ``load_weights``, so the optimiser starts fresh at this script's LR with AdamW weight
  decay at the config default).
- ``scratch``: a fresh random init at the same net size — v3's operator-ceiling history
  suggests a fresh net may imitate better than a converged one.

Both arms train through the *existing* ``BaseNNetWrapper.train`` path (lazy board
re-encode, sparse-target densify per batch, AdamW) on identical data: a game-granular
held-out split (no position of a held-out game leaks into training), 2× order-2
symmetry augmentation on the train side only. Early stop on held-out policy CE; every
improvement checkpoints the arm, so the run's product is the best-CE net per arm. Also
tracked per epoch: held-out top-1 accuracy vs Pentobi's move and value calibration
split by side-to-move (Blokus outcomes are colour-skewed, so calibration is only
readable per colour).

One command on the box (GPU), e.g.::

    uv run python scripts/distill_sl.py \
        --config run_configurations/blokus_cloud_v2.json \
        --corpus ~/corpora/pentobi_l9_stage1 \
        --arms warm,scratch --warm-start ~/best_nets/v3_gen40.pth.tar \
        --out temp/benchmarks/distill_sl.json --ckpt-dir temp/distill_sl

**v2 corpora are detected automatically** (they keep their games under ``games/``) and
change the data, not the loss — ``loss_pi`` was always a KL against a full distribution.
On a v2 corpus the policy target is Pentobi's stored distribution (label smoothing
defaults to 0), ``--tau`` softens it at load, the opening dataset is mixed in at
``--opening-mix`` (its value label chosen by ``--opening-value``), the v1 corpus can be
mixed in as mid-game data via ``--v1-corpus``/``--v1-mix``, and **the held-out split is by
opening subtree rather than by game** — v2 deliberately shares openings across games, so a
game-level split would leak. See ``docs/plans/pentobi-corpus-v2.md`` V9.

The verdict is **not** computed here: it is the D8 mini-ladder
(``scripts/mini_ladder.py`` over both arms' checkpoints + the v3 gen-40 baseline;
gate = +10 pp at any of L5–L7 after SL alone).
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

from alphablokus.calibration import parse_net_sizes
from alphablokus.config import RunConfig, load_args
from alphablokus.games.blokusduo.pentobi.corpus import corpus_shards
from alphablokus.games.blokusduo.pentobi.corpus_v2 import game_shards, opening_shards
from alphablokus.games.blokusduo.pentobi.distill import (
    CorpusGameRows,
    build_training_examples,
    load_corpus_games,
    load_corpus_games_v2,
    load_opening_examples,
    measure_holdout_leakage,
    mix_examples,
    partition_by_unit,
    sample_games,
    split_opening_units,
)
from alphablokus.registry import instantiate_game, instantiate_game_and_network
from alphablokus.training.holdout import (
    HoldoutMetrics,
    ImitationDiagnostics,
    evaluate_holdout,
    evaluate_imitation_diagnostics,
    split_games_holdout,
)

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame
    from alphablokus.games.blokusduo.pentobi.corpus import CorpusExample
    from alphablokus.interfaces import INeuralNetWrapper

KNOWN_ARMS = ("warm", "scratch")


def _arm_config(base: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Base run config re-shaped for one supervised distillation arm.

    ``epochs: 1`` because ``train()`` is called once per SL epoch so held-out
    evaluation (and the early-stop decision) runs between passes; constant LR at
    the script's ``--lr`` (default 1e-4 — a fine-tune rate, not the self-play
    peak). AdamW weight decay and the perf knobs stay the config's; net size is
    the config's unless ``--net-size <F>x<B>`` overrides it (the sizing sweep).
    """
    num_filters = base.net_config.num_filters
    num_residual_blocks = base.net_config.num_residual_blocks
    if args.net_size:
        _, num_filters, num_residual_blocks = parse_net_sizes(args.net_size)[0]
    net_config = replace(
        base.net_config,
        learning_rate=args.lr,
        epochs=1,
        batch_size=args.batch_size,
        lr_scheduler="constant",
        num_filters=num_filters,
        num_residual_blocks=num_residual_blocks,
    )
    return replace(base, net_config=net_config)


def _seed_everything(seed: int) -> None:
    """Identical init RNG per arm construction (arms differ only in their weights)."""
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


def _diagnostics_summary(diagnostics: ImitationDiagnostics) -> str:
    """One log line: top-1 + per-colour calibration bias (predicted − actual)."""
    biases = ", ".join(
        f"player {c.player:+d}: bias {c.mean_predicted - c.mean_outcome:+.3f} (mse {c.value_mse:.3f})"
        for c in diagnostics.calibration
    )
    return f"top-1 {diagnostics.top1_accuracy:.3f} | {biases}"


def _run_arm(
    name: str,
    config: RunConfig,
    args: argparse.Namespace,
    game: BlokusDuoGame,
    train_flat: list[CorpusExample],
    holdout_flat: list[CorpusExample],
    holdout_actions: list[int],
    holdout_players: list[int],
) -> dict[str, Any]:
    """Train one arm to its early-stopped best; checkpoint every CE improvement."""
    _seed_everything(args.seed)
    _, wrapper = instantiate_game_and_network(config)
    if name == "warm":
        wrapper.load_weights(str(Path(args.warm_start).expanduser().resolve()))
        logger.info("Arm {}: warm-started from {}", name, args.warm_start)

    ckpt_path = (Path(args.ckpt_dir) / f"distill_{name}.pth.tar").resolve()

    def evaluate() -> tuple[HoldoutMetrics, ImitationDiagnostics]:
        metrics = evaluate_holdout(
            wrapper,  # type: ignore[arg-type]  # BaseNNetWrapper satisfies SupportsEncodedPrediction
            holdout_flat,
            encode_fn=game.encode_compact,
            action_size=game.get_action_size(),
            batch_size=args.eval_batch_size,
        )
        diagnostics = evaluate_imitation_diagnostics(
            wrapper,  # type: ignore[arg-type]
            holdout_flat,
            holdout_actions,
            holdout_players,
            encode_fn=game.encode_compact,
            batch_size=args.eval_batch_size,
        )
        return metrics, diagnostics

    def curve_row(epoch: int, metrics: HoldoutMetrics, diagnostics: ImitationDiagnostics) -> dict[str, Any]:
        return {"epoch": epoch, **asdict(metrics), "diagnostics": asdict(diagnostics)}

    curve: list[dict[str, Any]] = []
    best, best_diagnostics = evaluate()
    best_epoch = 0
    curve.append(curve_row(0, best, best_diagnostics))
    logger.info(
        "Arm {} epoch 0 (baseline): CE {:.4f}, value MSE {:.4f} | {}",
        name,
        best.policy_ce,
        best.value_mse,
        _diagnostics_summary(best_diagnostics),
    )

    epochs_since_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        wrapper.train(train_flat, generation=epoch, metrics=None, eval_set=None)
        metrics, diagnostics = evaluate()
        curve.append(curve_row(epoch, metrics, diagnostics))
        logger.info(
            "Arm {} epoch {}: CE {:.4f} (KL {:.4f}), value MSE {:.4f} | {}",
            name,
            epoch,
            metrics.policy_ce,
            metrics.policy_kl,
            metrics.value_mse,
            _diagnostics_summary(diagnostics),
        )
        if metrics.policy_ce <= best.policy_ce - args.min_delta:
            best, best_diagnostics, best_epoch, epochs_since_improvement = metrics, diagnostics, epoch, 0
            # ``save_checkpoint`` joins its filename onto the config's net
            # directory, so an absolute path lands exactly where asked.
            wrapper.save_checkpoint(str(ckpt_path))
            logger.info("Arm {}: new best CE {:.4f} at epoch {} → {}", name, best.policy_ce, epoch, ckpt_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logger.info("Arm {}: early stop at epoch {} (patience {})", name, epoch, args.patience)
                break

    if best_epoch == 0:
        # Nothing beat the baseline (a warm start can already sit at its floor);
        # still persist the arm so the D8 ladder always has a checkpoint to run.
        wrapper.save_checkpoint(str(ckpt_path))
        logger.warning("Arm {}: no epoch improved on the baseline — checkpointed the final state.", name)

    num_params = sum(p.numel() for p in wrapper.nnet.parameters())  # type: ignore[attr-defined]
    _release(wrapper)
    return {
        "warm_start": args.warm_start if name == "warm" else None,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best": asdict(best),
        "best_diagnostics": asdict(best_diagnostics),
        "checkpoint": str(ckpt_path),
        "curve": curve,
    }


def _corpus_version(corpus: Path) -> str:
    """Detect the corpus format: v2 keeps its games under a ``games/`` subdirectory."""
    return "v2" if (corpus / "games").is_dir() else "v1"


def _flatten(games: list[CorpusGameRows], attribute: str) -> list[Any]:
    """Flatten one per-position attribute of grouped games, preserving order."""
    return [item for rows in games for item in getattr(rows, attribute)]


def main() -> None:
    parser = argparse.ArgumentParser(description="SL distillation of the Pentobi corpus (warm + scratch arms)")
    parser.add_argument("--config", required=True, help="Base run config JSON (net arch + device + perf knobs)")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus directory of corpus_*.parquet shards")
    parser.add_argument("--arms", default="warm,scratch", help=f"Comma list from {KNOWN_ARMS} (default warm,scratch)")
    parser.add_argument("--warm-start", default=None, help="Checkpoint path for the warm arm (v3 gen-40)")
    parser.add_argument(
        "--net-size",
        default=None,
        help="Override net size as <F>x<B> (e.g. 160x10) for the sizing sweep; scratch arm only.",
    )
    parser.add_argument(
        "--corpus-version",
        choices=("auto", "v1", "v2"),
        default="auto",
        help="Corpus format (auto-detects v2 by its games/ subdirectory)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Label-smoothing mass over legal moves (default 0.1 for v1, 0 for v2's soft targets)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="v2 target temperature: p^(1/tau) over the stored support. Softens confidence; "
        "order-preserving, so it cannot fix a misordered target (v2 plan V9)",
    )
    parser.add_argument(
        "--opening-value",
        choices=("blend", "outcome", "search"),
        default="blend",
        help="v2 opening-row value label: count-shrunk blend (default), pure outcomes, or the teacher",
    )
    parser.add_argument(
        "--opening-mix",
        type=float,
        default=0.05,
        help="v2 fraction of sampled examples drawn from opening rows (~0.6%% by natural count)",
    )
    parser.add_argument("--v1-corpus", type=Path, default=None, help="v1 corpus directory to mix in as mid-game data")
    parser.add_argument("--v1-mix", type=float, default=0.0, help="Fraction of sampled examples from the v1 corpus")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2x order-2 symmetry augmentation on the train side (default on)",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Subsample the corpus to this many games")
    parser.add_argument("--holdout-frac", type=float, default=0.05, help="Fraction of games held out (default 0.05)")
    parser.add_argument("--seed", type=int, default=7, help="Split + init + subsample seed (default 7)")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max full passes per arm (default 20)")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop patience in epochs (default 3)")
    parser.add_argument("--min-delta", type=float, default=0.002, help="CE improvement that resets patience (nats)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Constant SL learning rate (default 1e-4)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("temp/distill_sl"), help="Where arm checkpoints land")
    parser.add_argument("--out", type=Path, default=Path("temp/benchmarks/distill_sl.json"))
    args = parser.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"Unknown arms {unknown}; expected a subset of {KNOWN_ARMS}.")
    if "warm" in arm_names and not args.warm_start:
        raise SystemExit("--warm-start <checkpoint> is required for the warm arm.")
    if args.net_size:
        parse_net_sizes(args.net_size)  # validate the F×B spec early
        if "warm" in arm_names:
            raise SystemExit(
                "--net-size resizes the net, incompatible with the warm arm (v3 gen-40 weights are 192x12); "
                "use --arms scratch for the sizing sweep."
            )

    config = load_args(args.config)
    if config.game != "blokusduo":
        raise SystemExit(f"The Pentobi corpus is Blokus-only; config game is {config.game!r}.")
    config = _arm_config(config, args)

    # One game instance (no net yet) builds the examples for every arm
    # (identical data); its static ``encode_compact`` is what the training
    # DataLoader ships to workers. Building here pays the corpus's one
    # legal-mask pass up front, before any GPU memory is claimed.
    game_typed: BlokusDuoGame = instantiate_game(config)  # type: ignore[assignment]  # blokusduo guaranteed above
    corpus = args.corpus.expanduser()
    version = _corpus_version(corpus) if args.corpus_version == "auto" else args.corpus_version
    epsilon = args.epsilon if args.epsilon is not None else (0.1 if version == "v1" else 0.0)
    logger.info("Corpus {} detected as {} (epsilon {}, target temperature {})", corpus, version, epsilon, args.tau)

    if version == "v1":
        shards = corpus_shards(corpus)
        if not shards:
            raise SystemExit(f"No corpus shards found in {corpus}")
        games = load_corpus_games(shards)
        if args.max_games is not None:
            games = sample_games(games, args.max_games, args.seed)
        train_games, holdout_games = split_games_holdout(games, args.holdout_frac, args.seed)
        train_flat = build_training_examples(game_typed, train_games, epsilon=epsilon, augment=args.augment)
    else:
        shards = game_shards(corpus / "games")
        if not shards:
            raise SystemExit(f"No v2 games shards found in {corpus / 'games'}")
        games = load_corpus_games_v2(shards, game_typed)
        if args.max_games is not None:
            games = sample_games(games, args.max_games, args.seed)
        # Split by opening subtree, not by game: v2 gives many games a shared opening,
        # so a game-level boundary would leak identical early positions across it.
        holdout_units = split_opening_units(
            [rows.opening_unit for rows in games],
            [float(len(rows)) for rows in games],
            args.holdout_frac,
            args.seed,
        )
        train_games, holdout_games = partition_by_unit(games, holdout_units)
        pools: dict[str, list[CorpusExample]] = {
            "games": build_training_examples(
                game_typed,
                train_games,
                epsilon=epsilon,
                augment=args.augment,
                temperature=args.tau,
            ),
        }
        weights = {"games": max(0.0, 1.0 - args.opening_mix - args.v1_mix)}
        opening_examples, opening_units = load_opening_examples(
            opening_shards(corpus / "opening"),
            game_typed,
            value_target=args.opening_value,
            temperature=args.tau,
            epsilon=epsilon,
            augment=args.augment,
        )
        pools["opening"] = [
            example for example, unit in zip(opening_examples, opening_units, strict=True) if unit not in holdout_units
        ]
        weights["opening"] = args.opening_mix
        if args.v1_corpus is not None and args.v1_mix > 0.0:
            v1_games = load_corpus_games(corpus_shards(args.v1_corpus.expanduser()))
            pools["v1"] = build_training_examples(game_typed, v1_games, epsilon=0.1, augment=args.augment)
            weights["v1"] = args.v1_mix
        logger.info(
            "Source pools: {} → mixed at {}",
            {name: len(pool) for name, pool in pools.items()},
            weights,
        )
        train_flat = mix_examples(pools, weights, seed=args.seed)
    holdout_flat = build_training_examples(
        game_typed,
        holdout_games,
        epsilon=epsilon,
        augment=False,
        temperature=args.tau,
    )
    holdout_actions: list[int] = _flatten(holdout_games, "actions")
    holdout_players: list[int] = _flatten(holdout_games, "players")

    # How much of the exam has the model already seen? Splitting by opening subtree keeps
    # whole lines apart but cannot keep whole positions apart — two openings can transpose
    # into the same board — so measure it rather than assume. Reported, never enforced:
    # the number qualifies the held-out score that feeds the ladder gate.
    leakage = measure_holdout_leakage(
        (board for rows in train_games for board in rows.boards),
        (board for rows in holdout_games for board in rows.boards),
    )
    logger.info(
        "Holdout leakage: {}/{} held-out rows share a position with training ({:.3%}); "
        "{:.3%} counting mirrors; {} distinct positions on both sides",
        leakage.leaked_rows,
        leakage.holdout_rows,
        leakage.leaked_fraction,
        leakage.leaked_fraction_mirror,
        leakage.shared_positions,
    )
    if leakage.leaked_fraction_mirror > 0.01:
        logger.warning(
            "More than 1% of the held-out set is also in training — the held-out score is "
            "flattered by that much, and the gate verdict should be read accordingly",
        )
    logger.info(
        "Corpus: {} games from {} shards → train {} games ({} examples, augment={}) / holdout {} games ({} positions)",
        len(games),
        len(shards),
        len(train_games),
        len(train_flat),
        args.augment,
        len(holdout_games),
        len(holdout_flat),
    )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, Any]] = {}
    for name in arm_names:
        logger.info("=== Arm {} ===", name)
        arms[name] = _run_arm(
            name,
            config,
            args,
            game_typed,
            train_flat,
            holdout_flat,
            holdout_actions,
            holdout_players,
        )

    payload = {
        "config": args.config,
        "corpus": str(args.corpus),
        "num_games": len(games),
        "holdout_fraction": args.holdout_frac,
        "corpus_version": version,
        "epsilon": epsilon,
        "target_temperature": args.tau,
        "opening_value": args.opening_value,
        "opening_mix": args.opening_mix,
        "v1_mix": args.v1_mix,
        "augment": args.augment,
        "seed": args.seed,
        "lr": args.lr,
        "timestamp": datetime.now(UTC).isoformat(),
        "holdout_leakage": leakage.to_dict(),
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, arm in arms.items():
        logger.info(
            "Arm {}: best CE {:.4f} at epoch {} → {}",
            name,
            arm["best"]["policy_ce"],
            arm["best_epoch"],
            arm["checkpoint"],
        )
    logger.info("Results → {} (verdict comes from the D8 mini-ladder, not this script)", args.out)


if __name__ == "__main__":
    main()
