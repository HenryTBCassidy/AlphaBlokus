# GPU Training POC

Add W&B reporting alongside the existing Plotly HTML reports, then prove the end-to-end training pipeline runs on the home PC's RTX 3060 Ti using TicTacToe as a smoke test. Companion to `docs/plans/archive/remote-training-setup.md` (network/SSH layer, already complete) and `docs/08-TRAINING-ESTIMATES.md` (the timing numbers in there need to be replaced with measurements taken on the actual hardware once the smoke test runs).

The plan spans two phases. Phase 1 (`W1–W7`) is code work on this branch (`feat/wandb-integration`) and ends at a merge into main. Phase 2 (`W8–W14`) is operational work on the home PC over SSH, with no new branch — it's running scripts and editing one or two config files. `W15` archives the plan.

Prerequisites: Tailscale + WSL2 + OpenSSH + key auth are already working between Mac and PC (`ssh gpu-cmd nvidia-smi` confirmed 2026-05-15). The PC's WSL2 Ubuntu has `python3` and `git`, but no `uv`, no torch, and no repo cloned.

---

## Checklist

| #   | Item                                                                                                          | Effort | Priority | Done |
|-----|---------------------------------------------------------------------------------------------------------------|--------|----------|------|
| W1  | Add `wandb` to `pyproject.toml`, run `uv sync`, commit lockfile                                               | 10 min | High     |      |
| W2  | Add `WandbConfig` frozen dataclass and optional field on `RunConfig`; wire through the JSON config loader     | 30 min | High     |      |
| W3  | Extend `MetricsCollector` (`core/storage.py`) to mirror existing `log_*` calls into `wandb.log`; add `close()` and call it from `Coach` in a `try/finally` | 1.5 hr | High | |
| W4  | Add a `run_configurations/smoke_test_gpu.json` (TicTacToe, `cuda: true`, W&B enabled) and document the new W&B fields in `smoke_test.json` | 15 min | High | |
| W5  | Run W4's CPU equivalent locally on the Mac (3 generations TicTacToe), verify the W&B dashboard receives metrics end-to-end and the HTML report still renders | 30 min | High | |
| W6  | Update `docs/07-DATA-STORAGE.md` with a W&B section explaining what's logged and when to prefer it over the HTML reports | 20 min | Medium | |
| W7  | Open PR for `feat/wandb-integration`, merge into main, delete branch                                          | 10 min | High     |      |
| W8  | On the PC: set OpenSSH default shell to WSL2 bash (T5 from the archived remote-setup-windows plan), reconnect, verify `ssh gpu` lands in bash | 10 min | Medium | |
| W9  | On the PC inside WSL2: clone the repo, install `uv`, `uv sync`, `apt install -y tmux`, run `uv run pytest -m "not slow"` and confirm green | 20 min | High | |
| W10 | On the PC: confirm `torch.cuda.is_available()` returns `True` and `torch.cuda.get_device_name(0)` reports the RTX 3060 Ti | 5 min | High | |
| W11 | Design a TicTacToe GPU training config sized to load the GPU meaningfully but finish in under an hour (10–20 generations, moderate net), commit as `run_configurations/ttt_gpu_first_run.json` | 20 min | High | |
| W12 | Run W11 in `tmux` on the PC; in a second SSH session, watch `nvidia-smi` to confirm the python process is listed and GPU utilisation is non-zero; capture a screenshot for the writeup later | 1 hr | High | |
| W13 | Post-run verification: W&B dashboard populated end-to-end, parquet metrics on disk, HTML report regenerates locally after `rsync`, arena win-rates sensible for TicTacToe self-play | 20 min | High | |
| W14 | Re-run `scripts/mcts_profiling.py` on the PC against Blokus and replace the estimated 3080 Ti numbers in `docs/08-TRAINING-ESTIMATES.md` with measurements | 30 min | Medium | |
| W15 | `git mv docs/plans/gpu-training-poc.md docs/plans/archive/gpu-training-poc.md`                                 | 2 min  | Low      |      |

---

## W1. Add wandb dependency

`uv add wandb` (or hand-edit `pyproject.toml` and `uv lock`). No code changes. Commit `pyproject.toml` and `uv.lock`. Confirm `uv run python -c "import wandb; print(wandb.__version__)"` succeeds inside the project venv.

---

## W2. WandbConfig and RunConfig wiring

**Current state:** `core/config.py:35` defines `NetConfig` and `core/config.py` defines `RunConfig` as frozen dataclasses loaded from JSON. There is currently no concept of reporting backends — Plotly HTML output is hard-wired through `reporting/training.py`.

**Fix:** Add a `WandbConfig` frozen dataclass and an optional `wandb` field on `RunConfig`. Make the absence of the field equivalent to "W&B disabled" so existing configs (`test_run.json`, `smoke_test.json`, `full_run.json`) keep working unchanged.

```python
@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str | None = None
    tags: tuple[str, ...] = ()
    mode: Literal["online", "offline", "disabled"] = "online"
```

`RunConfig` gains `wandb: WandbConfig | None = None`. JSON loader maps a missing or null `wandb` key to `None`.

---

## W3. MetricsCollector W&B integration

**Current state:** `core/storage.py:56` defines `MetricsCollector` with `log_training`, `log_arena`, `log_timing`, `log_self_play_profiling`, `log_resource_usage`, `log_training_throughput`, and `flush`. `Coach` (`core/coach.py:94`) holds an instance and routes everything through it. Output is parquet on disk plus an HTML report generated from those parquets at end of run.

**Fix:** Add an optional W&B client to `MetricsCollector`. If `config.wandb is not None` at construction, call `wandb.init(project=..., entity=..., tags=..., mode=..., config=asdict(config))` and store the run handle. In each existing `log_*` method, after the parquet append, also call `wandb.log({...}, step=generation)` with the same key/value pairs. Add a `close()` method that calls `wandb.finish()` when the W&B client is present.

`Coach` needs to call `self.metrics.close()` in a `try/finally` around the generation loop in `core/coach.py` so a crash mid-run still flushes the W&B state.

Don't replace the parquet writes or the HTML report — W&B is additive.

---

## W4. Smoke-test configs

Two files:

- `run_configurations/smoke_test_gpu.json`: copy of `smoke_test.json` with `net_config.cuda: true` and a `wandb` block (`project: "alphablokus-poc"`, `tags: ["smoke", "gpu", "ttt"]`).
- `run_configurations/smoke_test.json`: leave `cuda: false` but add the `wandb` block (with `mode: "disabled"`) as a documented example so future configs have something to copy.

Both run TicTacToe — Blokus is not in scope for this plan.

---

## W5. Local end-to-end smoke test (Mac, CPU)

Run the (CPU) variant of W4 on the MacBook. Expectation: the existing parquet outputs and HTML report look identical to a pre-W&B run, and the W&B dashboard at the project URL shows the same metrics live. Catch any wiring bugs here before pushing to the PC.

If `wandb.init` prompts for a login on first run, capture the API key into `~/.netrc` so subsequent runs are non-interactive.

---

## W6. Documentation

Add a "W&B integration" section to `docs/07-DATA-STORAGE.md` covering: what gets logged, where parquet vs. W&B sit in the data flow, when to use the dashboard vs. the HTML report (rule of thumb: dashboard for live monitoring of unattended runs on the PC; HTML report for retrospective deep-dives once results are `rsync`'d back to the Mac), and how to disable W&B for offline/airgapped runs.

---

## W7. PR and merge

Open a PR against main with a short description linking to this plan. Merge, delete `feat/wandb-integration`. Phase 2 starts on a clean main checkout.

---

## W8. OpenSSH default shell on the PC

`ssh gpu` currently lands in PowerShell because step T5 of the archived remote-setup-windows plan was never executed. From the Mac:

```bash
ssh gpu-cmd 'powershell -Command "New-ItemProperty -Path HKLM:\\SOFTWARE\\OpenSSH -Name DefaultShell -Value C:\\Windows\\System32\\wsl.exe -PropertyType String -Force; Restart-Service sshd"'
```

Then reconnect with `ssh gpu` and verify the prompt is WSL2 bash. Future SSH sessions skip the `wsl -d Ubuntu -- bash -lc '...'` wrapper.

---

## W9. PC repo and dependency install

Inside WSL2 over SSH:

```bash
cd ~
git clone https://github.com/HenryTBCassidy/AlphaBlokus.git
cd AlphaBlokus
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
sudo apt install -y tmux
uv run pytest -m "not slow"
```

Test suite must pass. If `uv sync` fails on torch CUDA wheels, fall back to the CPU wheel and retry — this is a common WSL2 gotcha and worth noting in `docs/07-DATA-STORAGE.md` if it happens.

---

## W10. GPU + PyTorch sanity check

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')"
```

Expected: `True NVIDIA GeForce RTX 3060 Ti 8 GB`. If `False`, debug WSL2 CUDA driver before going further — no point running training that silently falls back to CPU.

---

## W11. TicTacToe GPU training config

Goal: a config that exercises the full pipeline (self-play → train → arena → reporting → W&B) and clearly loads the GPU, but completes in under an hour so we can iterate. Suggested starting point: 15 generations, 30 self-play games/gen, 50 MCTS sims, 128 filters / 3 residual blocks, batch size 64, 10 arena games. Tune up or down based on observed GPU utilisation in W12.

Commit as `run_configurations/ttt_gpu_first_run.json` on main (no branch — single config file, low risk).

---

## W12. Smoke run on the PC

```bash
ssh gpu
cd ~/AlphaBlokus && git pull
tmux new -s ttt-poc
uv run python main.py --config run_configurations/ttt_gpu_first_run.json
# Ctrl-B D to detach
```

In a second SSH session:

```bash
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "watch -n 1 nvidia-smi"'
```

Confirm the python process appears in the process list and `GPU-Util` is non-zero during training epochs. Capture a screenshot of the W&B dashboard mid-run for the writeup.

---

## W13. Post-run verification

Sanity checks after the run completes:

- W&B dashboard shows all expected metric series end-to-end, no gaps.
- Parquet files exist under `temp/ttt_gpu_first_run/`.
- `rsync` results back to Mac and regenerate the HTML report — must look identical in shape to historical TicTacToe runs.
- Arena win rates are within the range we'd expect for self-play converging on TicTacToe (the game is small enough that strong play emerges in tens of generations, so monotonic improvement is a reasonable smell test).

If any of these fail, fix and re-run before moving on.

---

## W14. Re-profile MCTS on the PC

Run `uv run python scripts/mcts_profiling.py` on the PC. Use the output to replace the estimated 3080 Ti numbers in `docs/08-TRAINING-ESTIMATES.md` with measured 3060 Ti numbers, and remove the "needs re-profiling" banner. This makes the doc honest for the first time since the project started.

---

## W15. Archive

`git mv docs/plans/gpu-training-poc.md docs/plans/archive/gpu-training-poc.md` in the same commit as marking W14 ✅, per the plan-format invariant.
