# Remote Training Runbook

How to run AlphaBlokus training jobs on the home PC (RTX 3060 Ti) from the MacBook over Tailscale + WSL2 + SSH. Companion to `docs/guides/STYLE-GUIDE.md` (code conventions) and `docs/plans/archive/remote-training-setup.md` (the original infrastructure setup, since superseded by this runbook for operational guidance).

This guide assumes Tailscale + WSL2 + OpenSSH + key auth are already working between the two machines (per the archived setup plan). It covers everything from logging in to a fresh PC environment through to pulling trained models back to the Mac.

---

## At-a-glance: a normal run, end-to-end

```bash
# 0. Make sure code is on a branch reachable from origin and the PC has it
git push -u origin <branch>
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && git fetch origin && git checkout <branch>"'

# 1. Launch training in a long-running SSH session (held open as a background task)
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 gpu-cmd \
  'wsl -d Ubuntu -- bash -lc "source /home/henry/.local/bin/env && cd /home/henry/AlphaBlokus && uv run python main.py --config run_configurations/<config>.json"' \
  > /tmp/training_local.log 2>&1

# 2. While that's running, in a separate session, capture the live W&B URL
grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[^[:space:]]+' /tmp/training_local.log | head -1

# 3. After the SSH command returns (= training finished or crashed), rsync results back
rsync -avz gpu:~/AlphaBlokus/temp/<run_name>/ ./temp/<run_name>/
```

The rest of this document explains why each piece is shaped the way it is and what to do when things break.

---

## Prerequisites checklist

A fresh PC environment needs all of:

| # | Item | Test |
|---|------|------|
| 1 | Windows 11 with OpenSSH Server, Tailscale connected | `ssh gpu-cmd 'echo ok'` from Mac |
| 2 | WSL2 + Ubuntu, CUDA visible inside WSL | `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "nvidia-smi"'` |
| 3 | `.wslconfig` at `C:\Users\<user>\.wslconfig` keeping the WSL VM alive (see below) | `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "cat /mnt/c/Users/Henry/.wslconfig"'` |
| 4 | Repo cloned in WSL: `~/AlphaBlokus` | `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "ls ~/AlphaBlokus"'` |
| 5 | `uv` installed in WSL (`~/.local/bin/uv`) | `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "source /home/henry/.local/bin/env && uv --version"'` |
| 6 | Project deps installed including dev extras (pytest) | `uv sync --extra dev` in `~/AlphaBlokus` |
| 7 | W&B logged in (`~/.netrc` in WSL has `api.wandb.ai`) | `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "grep -c api.wandb.ai ~/.netrc"'` returns ≥1 |

Items 1-3 are one-time host-level setup; 4-7 are per-environment.

---

## One-time PC setup

### `.wslconfig` to stop WSL killing your processes

The single biggest footgun. WSL2's VM shuts down 60 seconds after the last user session ends, killing *everything* inside — including `nohup`'d background processes. You **must** write `C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=-1
```

(Even with this, the cleanest training approach is to keep an SSH session alive — see below. The config is belt-and-braces.)

After editing, run `wsl --shutdown` to force re-read on next launch.

### WSL2 environment bootstrap

Inside WSL2 (via `ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "..."'`), install uv and clone:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~
git clone https://github.com/HenryTBCassidy/AlphaBlokus.git
cd AlphaBlokus
source /home/henry/.local/bin/env   # NOT $HOME/.local/bin/env — see "PowerShell mangling" below
uv sync --extra dev                  # --extra dev is required for pytest
uv run pytest -q                     # confirm 171 pass
```

### W&B login on the PC

The W&B CLI's `wandb login` doesn't reliably accept `WANDB_API_KEY` from env on this stack — use the Python API instead. From the Mac:

```bash
# Copy key to Windows home (ends up at C:\Users\Henry\)
scp local/wandb_api_key.txt gpu-cmd:wandb_api_key.txt

# Login from WSL via Python (CLI path is fragile)
KEY_B64=$(echo "import wandb; from pathlib import Path; \
  print('ok:', wandb.login(key=Path('/mnt/c/Users/Henry/wandb_api_key.txt').read_text().strip(), relogin=True))" \
  | base64)
ssh gpu-cmd "wsl -d Ubuntu -- bash -lc 'source /home/henry/.local/bin/env && echo $KEY_B64 | base64 -d | uv run --project /home/henry/AlphaBlokus python -'"

# Delete the temp key file
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "rm /mnt/c/Users/Henry/wandb_api_key.txt"'
```

The credentials land in WSL's `~/.netrc` and are persistent across restarts.

---

## Running a training job

### The launch pattern

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 gpu-cmd \
  'wsl -d Ubuntu -- bash -lc "source /home/henry/.local/bin/env && cd /home/henry/AlphaBlokus && uv run python main.py --config run_configurations/<config>.json"' \
  > /tmp/training_local.log 2>&1
```

Run this in a **long-running foreground SSH session** held by some process on the Mac (e.g. a background task in Claude Code, or a separate terminal you don't close, or in `tmux` on the Mac). Don't `nohup` it on the PC — disowned background processes inside WSL get killed when the SSH session closes, even with `nohup` (this seems to be a SIGHUP-leak via the WSL relay process, not strictly an idle timeout).

`ServerAliveInterval=30` keeps the TCP connection alive across network blips (hotel WiFi, sleeping the Mac briefly, etc).

### Get the W&B URL once it appears

```bash
grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[^[:space:]]+' /tmp/training_local.log | head -1
```

The URL prints ~10-15 seconds into the run.

### Check live progress

While running, from the Mac:

```bash
# Streamed output (full python stdout/stderr)
tail -f /tmp/training_local.log

# GPU utilisation snapshot
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"'

# Process check
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'
```

Or just open the W&B URL — that's the easiest live view.

### After it finishes

```bash
# Pull results back to Mac
rsync -avz gpu:~/AlphaBlokus/temp/<run_name>/ ./temp/<run_name>/

# Regenerate / re-open the HTML report locally
open temp/<run_name>/Reporting/report.html

# Load best.pth.tar for inference / play
uv run python -c "import torch; print(torch.load('temp/<run_name>/Nets/best.pth.tar').keys())"
```

---

## PowerShell mangling — the quoting tax

`ssh gpu-cmd '...'` lands in **Windows PowerShell**, not bash. Everything inside the single quotes is parsed by PowerShell before being handed to `wsl -- bash -lc "..."`. PowerShell will eat:

| PowerShell sees | What it does |
|-----------------|--------------|
| `$HOME` | Substitutes `C:\Users\<user>` *with the slashes stripped* — `C:UsersHenry`. **Always use literal absolute paths** in WSL commands (`/home/henry/...`). |
| `()` empty parens | Tries to interpret as a function call. Breaks Python like `torch.cuda.is_available()`. |
| `**` | Tries to interpret as a wildcard. Breaks Python like `total_memory >> 30`. |
| `\"foo bar\"` | Eats the escape, then re-splits on space. Breaks pytest filters like `-m "not slow"`. |
| `&&` / `||` | Not recognised as separators in older PowerShell. Use `;` for sequencing, or split into separate `ssh` calls. |

**The general escape hatch: base64-encode any Python you want to run remotely** and pipe through `base64 -d`:

```bash
SCRIPT_B64=$(echo "import torch; print(torch.cuda.is_available())" | base64)
ssh gpu-cmd "wsl -d Ubuntu -- bash -lc 'source /home/henry/.local/bin/env && echo $SCRIPT_B64 | base64 -d | uv run --project /home/henry/AlphaBlokus python -'"
```

Base64 contains only `[A-Za-z0-9+/=]` so PowerShell can't mangle it. This is also the cleanest pattern for ad-hoc remote queries (W&B API checks, log inspection, etc).

For shell scripts longer than one line: write to a file locally, `scp` to PC, run from there. **Always test the shebang line on the PC** — Windows-edited files can get CRLF line endings that break `bash`. `file ~/script.sh` should report `ASCII text executable`, not `with CRLF line terminators`. If it does have CRLF, fix with `sed -i 's/\r$//' ~/script.sh`.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Training dies after ~30s with no traceback in the log | Disowned process killed when SSH session closed | Use a long-running SSH session (don't `nohup & disown`) |
| `tmux ls` says no session right after creating one | WSL idle timeout (60s default) wiped `/tmp/tmux-*` | `vmIdleTimeout=-1` in `.wslconfig`, then `wsl --shutdown` |
| `error: Failed to spawn: pytest` | Dev extras not installed | `uv sync --extra dev` |
| `wandb login` CLI hangs or returns "No API key configured" | CLI auth via env is unreliable on this stack | Use `wandb.login(key=...)` from Python |
| `cannot open /tmp/something.sh` | `/tmp` is wiped between WSL VM lifecycles | Use `~/script.sh` instead |
| `bash: line 1: C:UsersHenry/.local/bin/env: No such file` | PowerShell substituted `$HOME` | Use literal `/home/henry/...` |
| `An expression was expected after '('` | PowerShell parsed `()` in your Python | Base64-encode the Python |
| nvidia-smi from Windows shows no python process | Windows nvidia-smi only sees Windows processes | Query inside WSL: `wsl -- bash -lc "nvidia-smi"` |
| Two W&B runs with the same name | `wandb.init(name=config.run_name)` collides on re-runs | We now append a UTC timestamp suffix — see `core/storage.py` `_init_wandb` |

---

## Killing a stuck or runaway run

```bash
# Find the process
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'

# Kill it cleanly (lets Coach._learn_loop's finally close W&B)
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pkill -INT -f main.py"'

# If that doesn't work after 30s
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pkill -9 -f main.py"'

# Confirm gone
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'
```

If WSL itself is wedged: `ssh gpu-cmd 'wsl --shutdown'` from PowerShell, then re-launch. This wipes `/tmp` but leaves `~/` intact (training results in `~/AlphaBlokus/temp/` survive).

---

## Updating code on the PC mid-session

```bash
# From Mac: push your changes to the feature branch
git push

# On PC: pull and verify
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && git pull && git log --oneline -3"'
```

If there's an in-flight training run when you pull, the on-disk code change won't affect it (Python has already imported its modules) — the next launch picks up the new code.

---

## Sanity-check commands to keep handy

```bash
# "Is the PC alive at all?"
ssh gpu-cmd 'echo ok'

# "Is WSL alive?"
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "uptime; whoami"'

# "Is CUDA working?"
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"'

# "Is there a training run going?"
ssh gpu-cmd 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'

# "What's the latest W&B run?"
ssh gpu-cmd "wsl -d Ubuntu -- bash -lc 'cd /home/henry/AlphaBlokus && uv run python -c \"import wandb; print(list(wandb.Api().runs(\\\"henrycassidy/alphablokus-poc\\\", per_page=1))[0].url)\"'"
```

The last one's quote-nesting is painful — prefer the base64 pattern from the "PowerShell mangling" section above for anything more complex.

---

## When something this guide doesn't cover comes up

Update this guide. Operational knowledge erodes fast — if you find yourself debugging the same WSL/PowerShell/W&B oddity twice, add a row to one of the tables above. The point of having this file is so the next session doesn't have to relearn it.
