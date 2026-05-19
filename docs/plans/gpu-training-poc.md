# GPU Training POC

Add W&B reporting alongside the existing Plotly HTML reports, then prove the end-to-end training pipeline runs on the home PC's RTX 3060 Ti using TicTacToe as a smoke test. Companion to `docs/plans/archive/remote-training-setup.md` (network/SSH layer, already complete) and `docs/08-TRAINING-ESTIMATES.md` (the estimated 3080 Ti timing numbers in there need to be replaced with measurements taken on the actual hardware).

The plan spans three phases. **Phase 1 (`W1–W6`)** is code work on this branch (`feat/wandb-integration`) — W&B integration + the two run configs. **Phase 2 (`W7–W17`)** is the operational PC test on the *feature branch* (not main): commit/push the code, set up the PC, run training, verify. The branch only merges to main *after* the PC validation passes. **Phase 3 (`W18–W19`)** is post-merge follow-up — re-profile MCTS with real measurements, archive the plan.

Prerequisites: Tailscale + WSL2 + OpenSSH + key auth are already working between Mac and PC (`ssh gpu-cmd nvidia-smi` confirmed 2026-05-15). The PC's WSL2 Ubuntu has `python3` and `git`, but no `uv`, no torch, and no repo cloned. W&B Pro account already set up; API key stored at `local/wandb_api_key.txt` on the Mac (gitignored).

---

## Checklist

| #   | Item                                                                                                          | Effort | Priority | Done |
|-----|---------------------------------------------------------------------------------------------------------------|--------|----------|------|
| W1  | Add `wandb` to `pyproject.toml`, run `uv sync`, commit lockfile                                               | 10 min | High     | ✅    |
| W2  | Add `WandbConfig` frozen dataclass and optional field on `RunConfig`; wire through the JSON config loader     | 30 min | High     | ✅    |
| W3  | Extend `MetricsCollector` (`core/storage.py`) to mirror existing `log_*` calls into `wandb.log`; add `close()` and call it from `Coach` in a `try/finally` | 1.5 hr | High | ✅ |
| W4  | Add `run_configurations/smoke_test_gpu.json` (TicTacToe, `cuda: true`, W&B enabled); document W&B fields in `smoke_test.json` | 15 min | High | ✅ |
| W5  | Run W4's CPU equivalent locally on the Mac (3 generations), verify the W&B dashboard receives metrics end-to-end and the HTML report still renders | 30 min | High | ✅ |
| W6  | Add `run_configurations/ttt_full.json` for the PC run (30 gens, 50 eps, 80 sims, 256 filters / 3 blocks, CUDA on, W&B online) | 5 min | High | ✅ |
| W7  | Commit the entire W1–W6 working tree as the planned five one-sentence commits: wandb dep, WandbConfig, MetricsCollector W&B integration, smoke configs, mac+full TTT configs | 10 min | High | ✅ |
| W8  | Push `feat/wandb-integration` to origin                                                                       | 2 min  | High     | ✅    |
| W9  | Update `docs/07-DATA-STORAGE.md` with a W&B section (what is logged, when to use dashboard vs HTML); commit + push | 25 min | Medium | ✅ |
| W10 | On the PC: set OpenSSH default shell to WSL2 bash (T5 from the archived remote-setup-windows plan), reconnect, verify `ssh gpu` lands in bash | 10 min | Medium | Deferred |
| W11 | On the PC inside WSL2: clone the repo, install `uv`, `uv sync --extra dev`, run `uv run pytest -q` and confirm green (tmux already installed) | 20 min | High | ✅ |
| W12 | On the PC: `git fetch origin && git checkout feat/wandb-integration` to get the W&B code on the test branch | 3 min | High | ✅ |
| W13 | On the PC: verify `torch.cuda.is_available()` is `True` and `torch.cuda.get_device_name(0)` reports the RTX 3060 Ti | 5 min | High | ✅ |
| W14 | On the PC: `wandb login` using the API key from `local/wandb_api_key.txt` (copy over via `scp` from Mac, then `wandb.login(key=...)` via Python — CLI path unreliable) | 5 min | High | ✅ |
| W15 | On the PC: run `ttt_full.json` via a long-running foreground SSH session (NOT `nohup & disown` — disowned WSL processes get SIGHUP'd when SSH closes). Predicted 22 min, actual **6.6 min**. | 30 min | High | ✅ |
| W16 | Post-run verification: W&B dashboard populated end-to-end, `best.pth.tar` exists in `Nets/` (gen 11), all 7 parquet directories populated for all 30 generations (210 files) | 15 min | High | ✅ |
| W17 | Pull results back to Mac. Rsync doesn't work cleanly through PowerShell→WSL relay, so use tar+scp: `wsl tar czf /mnt/c/Users/Henry/ttt_full.tar.gz`, `scp gpu-cmd:ttt_full.tar.gz /tmp/`, untar locally. Confirmed `best.pth.tar` loads (2.58M params) and HTML report (1.4 MB) is present. | 10 min | Medium | ✅ |
| W18 | Open PR for `feat/wandb-integration`, merge into main, delete branch                                          | 10 min | High     |      |
| W19 | Re-run `scripts/mcts_profiling.py` on the PC against Blokus and replace the estimated 3080 Ti numbers in `docs/08-TRAINING-ESTIMATES.md` with measurements. Surprise finding: PC is ~1.7× slower per sim than Mac because move gen (CPU) dominates and the PC's CPU is slower; GPU only helps at large net sizes. | 30 min | Medium | ✅ |
| W20 | `git mv docs/plans/gpu-training-poc.md docs/plans/archive/gpu-training-poc.md`                                 | 2 min  | Low      |      |

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
- `run_configurations/smoke_test.json`: add a `wandb` block (`mode: "online"`) so the Mac smoke test exercises the full W&B path.

Both run TicTacToe — Blokus is not in scope for this plan.

---

## W5. Local end-to-end smoke test (Mac, CPU)

Run the CPU smoke test (`smoke_test.json`) on the Mac with `mode: "online"`. Expectation: the existing parquet outputs and HTML report look identical to a pre-W&B run, and the W&B dashboard at the project URL shows the same metrics live. Catch any wiring bugs here before pushing to the PC.

If `wandb.init` prompts for a login on first run, capture the API key into `~/.netrc` so subsequent runs are non-interactive.

---

## W6. Mac demo + full PC config files

Two more configs:

- `run_configurations/ttt_mac_demo.json` — written for the Mac calibration run (25 gens, 40 eps, 60 sims, 128 filters / 2 blocks). Used to measure actual wall clock against a prediction so PC sizing can be calibrated. Completed 2026-05-15 in 305s (~5 min) vs an 8-min prediction; model converged at gen 10 with `best.pth.tar` saved.
- `run_configurations/ttt_full.json` — the PC target config: 30 generations, 50 episodes, 80 MCTS sims, 20 arena matches, 256 filters / 3 residual blocks, batch 64, 6 epochs, CUDA on, W&B online. Sized to actually exercise the GPU (the Mac demo's smaller net is kernel-launch bound and barely uses the GPU). Predicted wall clock on the 3060 Ti: **22 minutes** (range 15-30 min).

---

## W7. Commit the integration work

The W1–W6 working tree is currently uncommitted on `feat/wandb-integration` (Henry reviews diffs first). Once approved, commit as five one-sentence commits matching the W1–W4 + W6 boundaries:

1. `Add wandb dependency` — `pyproject.toml`, `uv.lock`
2. `Add optional WandbConfig field on RunConfig` — `core/config.py`
3. `Mirror MetricsCollector log_* calls to W&B when configured` — `core/storage.py`, `core/coach.py`
4. `Add W&B settings to smoke_test and smoke_test_gpu configs` — `run_configurations/smoke_test.json`, `run_configurations/smoke_test_gpu.json`
5. `Add TicTacToe Mac demo and full PC run configs` — `run_configurations/ttt_mac_demo.json`, `run_configurations/ttt_full.json`

Plus the gitignore tweak (`wandb/`, `local/wandb_api_key.txt` already covered) and the plan-format tweak (tick rule) as their own one-sentence commits if not already covered.

---

## W8. Push the branch

`git push -u origin feat/wandb-integration`. Confirm the branch is visible on GitHub.

---

## W9. Documentation

Add a "W&B integration" section to `docs/07-DATA-STORAGE.md` covering:

- What gets logged to which namespace (`training/`, `arena/`, `timing/`, `self_play/`, `resources/`, `throughput/`) — table form.
- How parquet and W&B coexist (W&B is additive; parquet is still the source of truth for HTML reports).
- When to use which (dashboard for live monitoring of unattended runs; HTML report for retrospective deep-dives once `rsync`'d back).
- How to disable W&B for offline/airgapped runs (`mode: "disabled"` in the JSON config, or no `wandb` block at all).
- Where the API key lives (`local/wandb_api_key.txt`, gitignored).

Commit + push as its own one-sentence commit so it can be reviewed independently of the code.

---

## W10. OpenSSH default shell on the PC

> **Deferred 2026-05-15.** Setting the registry key under `HKLM:\SOFTWARE\OpenSSH` requires Administrator elevation, which isn't reliably available over a non-interactive SSH session. All subsequent PC commands in this plan are wrapped in `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "..."'`, which works regardless of the default shell. This is a QoL fix Henry can do interactively later from an Admin PowerShell.

### Original instructions (kept for reference)


`ssh gpu` currently lands in PowerShell because step T5 of the archived `remote-setup-windows` plan was never executed. From the Mac:

```bash
ssh gpu-cmd 'powershell -Command "New-ItemProperty -Path HKLM:\\SOFTWARE\\OpenSSH -Name DefaultShell -Value C:\\Windows\\System32\\wsl.exe -PropertyType String -Force; Restart-Service sshd"'
```

Then reconnect with `ssh gpu` and verify the prompt is WSL2 bash. Future SSH sessions skip the `wsl -d Ubuntu -- bash -lc '...'` wrapper.

If this is fiddly or fails, skip it — every PC command for the rest of the plan can be wrapped in `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "..."'`. It's a QoL fix, not a blocker.

---

## W11. PC repo and dependency install

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

Test suite must pass (170 tests, ~3s on Mac, similar on PC). If `uv sync` fails on torch CUDA wheels, fall back to the CPU wheel and retry — this is a common WSL2 gotcha. Note any deviations in `docs/07-DATA-STORAGE.md` follow-up.

---

## W12. Check out the feature branch

```bash
git fetch origin
git checkout feat/wandb-integration
git log --oneline -5  # sanity: should show the W&B commits
```

We deliberately test on the feature branch before merging to main. If the PC reveals a CUDA-specific bug in our W&B integration, fixing it on the same branch keeps history clean.

---

## W13. GPU + PyTorch sanity check

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // (1024**3), 'GB')"
```

Expected: `True NVIDIA GeForce RTX 3060 Ti 8 GB`. If `False`, debug WSL2 CUDA driver before going further — no point running training that silently falls back to CPU.

---

## W14. W&B login on the PC

From the Mac, copy the key:

```bash
scp local/wandb_api_key.txt gpu:~/wandb_api_key.txt
ssh gpu 'WANDB_API_KEY=$(cat ~/wandb_api_key.txt) wandb login --relogin && rm ~/wandb_api_key.txt'
```

The `rm` at the end avoids leaving the key in the PC's home directory once `~/.netrc` is populated. Verify with `ssh gpu 'wandb status'` or by running a tiny `wandb.Api().viewer.entity` check.

---

## W15. PC training run

```bash
ssh gpu
cd ~/AlphaBlokus
tmux new -s ttt-full
uv run python main.py --config run_configurations/ttt_full.json
# Ctrl-B D to detach
```

In a second SSH session (don't re-attach to tmux — just run side-by-side):

```bash
ssh gpu 'watch -n 2 nvidia-smi'
```

Confirm the python process appears in the GPU process list and `GPU-Util` is non-zero during training epochs. Take a screenshot of the W&B dashboard mid-run for the writeup.

**Record actual wall clock at the end.** Compare to the 22 min prediction. If it's much faster, the PC is underloaded and we should size up for future runs. If much slower, future runs need scaling down.

---

## W16. Post-run verification

Sanity checks while still SSH'd into the PC:

- W&B dashboard shows all 6 namespaces (`training/`, `arena/`, `timing/`, `self_play/`, `resources/`, `throughput/`) populated for all 30 generations, no gaps.
- `temp/ttt_full/Nets/best.pth.tar` exists (means at least one generation was accepted).
- `find temp/ttt_full -name "*.parquet" | wc -l` — expect 30 generations × 6 parquet types + 30 self-play files = 210ish.
- Arena win rates look sane (early gens: real wins; later: plateauing toward all-draws).
- No tracebacks or crashes in `temp/ttt_full/Logs/alpha.log`.

If anything looks wrong, fix on the feature branch and re-run before merging.

---

## W17. Pull results back to the Mac

```bash
rsync -avz gpu:~/AlphaBlokus/temp/ttt_full/ ./temp/ttt_full/
```

Then on the Mac, regenerate the HTML report locally (Coach already wrote it on the PC, but rendering on the Mac confirms the parquets are portable):

```bash
# If we have a standalone report-regen entry point, use it; otherwise inspect by opening the report.html
open temp/ttt_full/Reporting/report.html
```

Optionally `torch.load('temp/ttt_full/Nets/best.pth.tar')` from a notebook on the Mac to confirm the checkpoint loads cleanly. Once everything is verified the PC's `temp/ttt_full/` can be deleted if disk space matters.

---

## W18. PR + merge

Open a PR for `feat/wandb-integration` → `main` with a short description linking to this plan and the PC run's W&B dashboard URL as evidence of working end-to-end. Merge, delete branch, both locally and on origin.

After merge, on the PC: `git checkout main && git pull` so the PC is back on a clean main checkout for future runs.

---

## W19. Re-profile MCTS on the PC

Run `uv run python scripts/mcts_profiling.py` on the PC. Use the output to replace the estimated 3080 Ti numbers in `docs/08-TRAINING-ESTIMATES.md` with measured 3060 Ti numbers, and remove the "needs re-profiling" banner. This makes the doc honest for the first time since the project started.

Commit as a separate one-line commit on main (no branch needed for a one-file doc update).

---

## W20. Archive

`git mv docs/plans/gpu-training-poc.md docs/plans/archive/gpu-training-poc.md` in the same commit as marking W19 ✅, per the plan-format invariant.
