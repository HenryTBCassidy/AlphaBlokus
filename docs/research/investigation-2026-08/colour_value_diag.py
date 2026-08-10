"""Colour-conditional value diagnostic on the frozen eval sets.

For each run's eval set (200 gen-1 self-play positions, targets = final outcome
from the mover's perspective) and a checkpoint, compute:
  - the net's value MSE vs targets
  - a colour-only baseline MSE (predict the per-colour mean target)
  - value skill = 1 - mse/colour_only_mse (holdout.py's definition)
  - mean prediction / mean target per inferred mover colour
Mover colour inferred from the canonical compact board: mover piece count ==
opponent piece count -> White to move; opponent one more -> Black. Positions
where neither holds (post-pass endgames) are dropped.
"""
import sys
import numpy as np

REPO = "/Users/henrycassidy/code/personal projects/AlphaBlokus"
sys.path.insert(0, REPO + "/src")

from alphablokus.config import MCTSConfig, NetConfig, RunConfig  # noqa: E402
from alphablokus.games.blokusduo.game import BlokusDuoGame  # noqa: E402
from alphablokus.games.blokusduo.nn.wrapper import NNetWrapper  # noqa: E402
from alphablokus.games.blokusduo.pieces import default_pieces_path  # noqa: E402
from pathlib import Path  # noqa: E402

game = BlokusDuoGame(pieces_config_path=default_pieces_path())
config = RunConfig(
    game="blokusduo", run_name="colour_diag", num_generations=1, num_eps=1,
    temp_threshold=12, update_threshold=0.55, num_arena_matches=1,
    root_directory=Path(REPO) / "temp", load_model=False,
    mcts_config=MCTSConfig(num_mcts_sims=8, cpuct=2.5, mcts_batch_size=16),
    net_config=NetConfig(learning_rate=1e-3, dropout=0.0, epochs=1, batch_size=8,
                         cuda=False, num_filters=192, num_residual_blocks=12,
                         fp16_inference=False),
)


def mover_colour(compact):
    """+1 if White to move, -1 if Black, 0 if ambiguous (pass happened)."""
    mine = {int(v) for v in np.unique(compact) if v > 0}
    theirs = {int(-v) for v in np.unique(compact) if v < 0}
    if len(mine) == len(theirs):
        return 1
    if len(theirs) == len(mine) + 1:
        return -1
    return 0


def run(eval_dir, ckpt, label):
    boards = np.load(eval_dir + "/boards.npy")
    compact = np.load(eval_dir + "/compact_boards.npy")
    targets = np.load(eval_dir + "/target_values.npy")
    nnet = NNetWrapper(game, config)
    nnet.load_checkpoint(filename=ckpt)
    preds = []
    for i in range(len(boards)):
        _, v = nnet.predict_encoded(boards[i])
        preds.append(float(np.asarray(v).reshape(-1)[0]))
    preds = np.array(preds)
    colours = np.array([mover_colour(c) for c in compact])
    keep = colours != 0
    p, t, c = preds[keep], targets[keep], colours[keep]
    mse = float(np.mean((p - t) ** 2))
    colour_means = {col: t[c == col].mean() for col in (1, -1)}
    baseline = np.array([colour_means[col] for col in c])
    colour_mse = float(np.mean((baseline - t) ** 2))
    skill = 1 - mse / colour_mse
    # correlation structure
    corr_ct = float(np.corrcoef(c, t)[0, 1])
    corr_cp = float(np.corrcoef(c, p)[0, 1])
    # partial: does p predict t beyond colour? residualise both on colour
    rp = p - np.array([p[c == col].mean() for col in c])
    rt = t - np.array([t[c == col].mean() for col in c])
    corr_resid = float(np.corrcoef(rp, rt)[0, 1])
    print(f"\n== {label} ==  n={keep.sum()}/{len(boards)} (dropped {int((~keep).sum())} ambiguous)")
    print(f"  target mean | White-to-move: {t[c==1].mean():+.3f} (n={int((c==1).sum())})   Black-to-move: {t[c==-1].mean():+.3f} (n={int((c==-1).sum())})")
    print(f"  pred   mean | White-to-move: {p[c==1].mean():+.3f}   Black-to-move: {p[c==-1].mean():+.3f}")
    print(f"  net MSE {mse:.4f}   colour-only MSE {colour_mse:.4f}   value skill {skill:+.4f}")
    print(f"  corr(colour,target) {corr_ct:+.3f}   corr(colour,pred) {corr_cp:+.3f}   corr(pred,target | colour) {corr_resid:+.3f}")


V3 = REPO + "/temp/runs/blokus/blokus_cloud_v3"
RR = REPO + "/temp/runs/blokus/blokus_paired_gate_rerun"
run(V3 + "/EvalSet", V3 + "/Nets/accepted_40.pth.tar", "v3 gen-40 on v3 eval set")
run(V3 + "/EvalSet", V3 + "/Nets/accepted_1.pth.tar", "v3 gen-1 on v3 eval set")
run(RR + "/EvalSet", RR + "/Nets/accepted_20.pth.tar", "rerun gen-20 on rerun eval set")
run(RR + "/EvalSet", V3 + "/Nets/accepted_40.pth.tar", "v3 gen-40 on rerun eval set")
