"""C5 probe: optimizer continuation semantics across the Coach's generation cycle.

Empirically drives the real wrapper through the Coach's exact sequence:
  save temp -> train candidate -> (reject) load temp -> train next candidate
and checks:
  1. Adam moments persist across an ACCEPTED generation (no reset).
  2. On REJECT, weights AND Adam moments revert exactly to the pre-training
     snapshot; the LR schedule clock does NOT rewind.
  3. weight_decay is present and equal to config in every param group, before
     and after a reject-reload of a legacy (decay-less) checkpoint.
  4. The cosine schedule steps once per epoch and never restarts within a run.
  5. `load_weights` (warm start) leaves optimizer + scheduler fresh — i.e. a
     warm continuation DOES restart the LR at peak (known A3 concern; recorded).
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab-audit"))

from alphablokus.config import MCTSConfig, NetConfig, RunConfig
from alphablokus.games.tictactoe.game import TicTacToeGame
from alphablokus.games.tictactoe.nn.wrapper import NNetWrapper

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


tmp = Path(tempfile.mkdtemp())
config = RunConfig(
    game="tictactoe",
    run_name="c5_probe",
    num_generations=10,
    num_eps=1,
    temp_threshold=3,
    update_threshold=0.55,
    num_arena_matches=2,
    root_directory=tmp,
    load_model=False,
    mcts_config=MCTSConfig(num_mcts_sims=4, cpuct=1.0),
    net_config=NetConfig(
        learning_rate=1e-3,
        dropout=0.0,
        epochs=2,
        batch_size=8,
        cuda=False,
        num_filters=8,
        num_residual_blocks=1,
        lr_scheduler="cosine",
        lr_eta_min=1e-4,
    ),
)
torch.manual_seed(0)
np.random.seed(0)
game = TicTacToeGame()
w = NNetWrapper(game, config)


def make_examples(n=64):
    out = []
    for _ in range(n):
        board = game.initialise_board()
        pi = np.random.dirichlet(np.ones(game.get_action_size()))
        out.append((board.to_compact(), (np.arange(game.get_action_size(), dtype=np.int32), pi.astype(np.float32)), 1.0))
    return out


def opt_snapshot(wrapper):
    st = wrapper.optimizer.state_dict()["state"]
    return {k: {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()} for k, v in st.items()}


def snapshots_equal(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        for kk in a[k]:
            va, vb = a[k][kk], b[k][kk]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


print("== 1. moments persist across an accepted generation")
w.train(make_examples(), generation=1)
snap_after_g1 = opt_snapshot(w)
check("moments exist after gen 1", len(snap_after_g1) > 0)
lr_after_g1 = w.optimizer.param_groups[0]["lr"]
clock_after_g1 = w.scheduler.last_epoch

# Coach cycle for gen 2: save temp FIRST (pre-training incumbent), then train.
w.save_checkpoint(filename="temp.pth.tar")
w.train(make_examples(), generation=2)
snap_after_g2 = opt_snapshot(w)
check("moments advanced during gen 2 (no silent reset)", not snapshots_equal(snap_after_g1, snap_after_g2))
clock_after_g2 = w.scheduler.last_epoch
check("scheduler stepped once per epoch", clock_after_g2 == clock_after_g1 + config.net_config.epochs, f"{clock_after_g1} -> {clock_after_g2}")

print("== 2. reject-reload reverts weights + moments, keeps the LR clock")
weights_pre_reject = {k: v.clone() for k, v in w.nnet.state_dict().items()}
w.load_checkpoint(filename="temp.pth.tar", restore_lr_schedule=False)
snap_after_revert = opt_snapshot(w)
check("Adam moments reverted to the pre-training snapshot", snapshots_equal(snap_after_revert, snap_after_g1))
check("weights actually changed by the revert", any(not torch.equal(weights_pre_reject[k], v) for k, v in w.nnet.state_dict().items()))
check("scheduler clock NOT rewound", w.scheduler.last_epoch == clock_after_g2, f"clock {w.scheduler.last_epoch}")
lr_now = w.optimizer.param_groups[0]["lr"]
check("optimizer LR re-synced to the scheduler (not the stale saved LR)", abs(lr_now - w.scheduler.get_last_lr()[0]) < 1e-12, f"lr {lr_now:.2e}")
check("LR is not the gen-1 saved value", lr_now != lr_after_g1, f"{lr_now:.6e} vs saved {lr_after_g1:.6e}")

print("== 3. weight decay live in every param group")
check("weight_decay == config in all groups", all(g["weight_decay"] == config.net_config.weight_decay for g in w.optimizer.param_groups), f"{config.net_config.weight_decay}")
# simulate a legacy checkpoint written with decay off
for g in w.optimizer.param_groups:
    g["weight_decay"] = 0.0
w.save_checkpoint(filename="legacy.pth.tar")
w.load_checkpoint(filename="legacy.pth.tar", restore_lr_schedule=False)
check("legacy decay-less checkpoint re-asserted to config on load", all(g["weight_decay"] == config.net_config.weight_decay for g in w.optimizer.param_groups))

print("== 4. warm start (load_weights) resets optimizer + schedule to peak")
w2 = NNetWrapper(game, config)
w2.train(make_examples(), generation=1)
w2.save_checkpoint(filename="donor.pth.tar")
w3 = NNetWrapper(game, config)
w3.load_weights(filename="donor.pth.tar")
check("fresh optimizer (no moments)", len(opt_snapshot(w3)) == 0)
check("LR back at configured peak", w3.optimizer.param_groups[0]["lr"] == config.net_config.learning_rate)
check("scheduler clock at 0", w3.scheduler.last_epoch == 0)
print("  (=> every warm continuation restarts the schedule at peak LR — by design; A3's concern)")

print()
if FAILURES:
    print("FAILURES:", FAILURES)
    sys.exit(1)
print("C5 probe: all checks passed")
