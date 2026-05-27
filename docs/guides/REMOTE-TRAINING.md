# Remote Training Runbook

How to run AlphaBlokus training jobs on a home PC (with CUDA-capable GPU) from a development machine over Tailscale + WSL2 + SSH. Companion to `docs/guides/STYLE-GUIDE.md` (code conventions) and `docs/plans/archive/remote-training-setup.md` (the original infrastructure setup, since superseded by this runbook for operational guidance).

This guide assumes Tailscale + WSL2 + OpenSSH + key auth are already working between the two machines (per the archived setup plan). It covers everything from logging in to a fresh PC environment through to pulling trained models back to the dev machine.

> **For maintainers' own setup:** placeholders below (`<wsl-user>`, `<windows-user>`, `<wandb-entity>`, etc) are deliberately generic so this guide stays portable. Your project's actual mapping should live in `local/MY-SETUP.md` (gitignored). On a fresh checkout, see that file (or recreate from the contents of [[reference_training_hardware]] in Claude memory).

---

## At-a-glance: a normal run, end-to-end

```bash
# 0. Make sure code is on a branch reachable from origin and the PC has it
git push -u origin <branch>
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && git fetch origin && git checkout <branch>"'

# 1. Launch training in a long-running SSH session (held open as a background task)
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 <gpu-host> \
  'wsl -d Ubuntu -- bash -lc "source /home/<wsl-user>/.local/bin/env && cd /home/<wsl-user>/AlphaBlokus && uv run python main.py --config run_configurations/<config>.json"' \
  > /tmp/training_local.log 2>&1

# 2. While that's running, in a separate session, capture the live W&B URL
grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[^[:space:]]+' /tmp/training_local.log | head -1

# 3. After the SSH command returns (= training finished or crashed), rsync results back
rsync -avz <gpu-host>:~/AlphaBlokus/temp/<run_name>/ ./temp/<run_name>/
```

Convention used below: `<gpu-host>` is your SSH alias for the PC (you can pick anything; we use a short name like `gpu`). The rest of this document explains why each piece is shaped the way it is and what to do when things break.

---

## Prerequisites checklist

A fresh PC environment needs all of:

| # | Item | Test |
|---|------|------|
| 1 | Windows 11 with OpenSSH Server, Tailscale connected | `ssh <gpu-host> 'echo ok'` from dev machine |
| 2 | WSL2 + Ubuntu, CUDA visible inside WSL | `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "nvidia-smi"'` |
| 3 | `.wslconfig` at `C:\Users\<windows-user>\.wslconfig` keeping the WSL VM alive (see below) | `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "cat /mnt/c/Users/<windows-user>/.wslconfig"'` |
| 4 | Repo cloned in WSL: `~/AlphaBlokus` | `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "ls ~/AlphaBlokus"'` |
| 5 | `uv` installed in WSL (`~/.local/bin/uv`) | `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "source /home/<wsl-user>/.local/bin/env && uv --version"'` |
| 6 | Project deps installed including dev extras (pytest) | `uv sync --extra dev` in `~/AlphaBlokus` |
| 7 | W&B logged in (`~/.netrc` in WSL has `api.wandb.ai`) | `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "grep -c api.wandb.ai ~/.netrc"'` returns ≥1 |

Items 1-3 are one-time host-level setup; 4-7 are per-environment.

---

## One-time PC setup

### `.wslconfig` to stop WSL killing your processes

The single biggest footgun. WSL2's VM shuts down 60 seconds after the last user session ends, killing *everything* inside — including `nohup`'d background processes. You **must** write `C:\Users\<windows-user>\.wslconfig`:

```ini
[wsl2]
vmIdleTimeout=-1
```

(Even with this, the cleanest training approach is to keep an SSH session alive — see below. The config is belt-and-braces.)

After editing, run `wsl --shutdown` to force re-read on next launch.

### WSL2 environment bootstrap

Inside WSL2 (via `ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "..."'`), install uv and clone:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~
git clone https://github.com/<your-fork>/AlphaBlokus.git
cd AlphaBlokus
source /home/<wsl-user>/.local/bin/env   # NOT $HOME/.local/bin/env — see "PowerShell mangling" below
uv sync --extra dev                       # --extra dev is required for pytest
uv run pytest -q                          # confirm tests pass
```

### W&B login on the PC

The W&B CLI's `wandb login` doesn't reliably accept `WANDB_API_KEY` from env on this stack — use the Python API instead. From the dev machine:

```bash
# Copy key to Windows home (ends up at C:\Users\<windows-user>\)
scp local/wandb_api_key.txt <gpu-host>:wandb_api_key.txt

# Login from WSL via Python (CLI path is fragile)
KEY_B64=$(echo "import wandb; from pathlib import Path; \
  print('ok:', wandb.login(key=Path('/mnt/c/Users/<windows-user>/wandb_api_key.txt').read_text().strip(), relogin=True))" \
  | base64)
ssh <gpu-host> "wsl -d Ubuntu -- bash -lc 'source /home/<wsl-user>/.local/bin/env && echo $KEY_B64 | base64 -d | uv run --project /home/<wsl-user>/AlphaBlokus python -'"

# Delete the temp key file
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "rm /mnt/c/Users/<windows-user>/wandb_api_key.txt"'
```

The credentials land in WSL's `~/.netrc` and are persistent across restarts.

---

## Running a training job

### The launch pattern

```bash
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 <gpu-host> \
  'wsl -d Ubuntu -- bash -lc "source /home/<wsl-user>/.local/bin/env && cd /home/<wsl-user>/AlphaBlokus && uv run python main.py --config run_configurations/<config>.json"' \
  > /tmp/training_local.log 2>&1
```

Run this in a **long-running foreground SSH session** held by some process on the dev machine (e.g. a background task in Claude Code, or a separate terminal you don't close, or in `tmux` locally). Don't `nohup` it on the PC — disowned background processes inside WSL get killed when the SSH session closes, even with `nohup` (this seems to be a SIGHUP-leak via the WSL relay process, not strictly an idle timeout).

`ServerAliveInterval=30` keeps the TCP connection alive across network blips (hotel WiFi, sleeping the dev machine briefly, etc).

### Get the W&B URL once it appears

```bash
grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[^[:space:]]+' /tmp/training_local.log | head -1
```

The URL prints ~10-15 seconds into the run.

### Check live progress

While running, from the dev machine:

```bash
# Streamed output (full python stdout/stderr)
tail -f /tmp/training_local.log

# GPU utilisation snapshot
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"'

# Process check
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'
```

Or just open the W&B URL — that's the easiest live view.

### After it finishes

```bash
# Pull results back to dev machine
rsync -avz <gpu-host>:~/AlphaBlokus/temp/<run_name>/ ./temp/<run_name>/

# Regenerate / re-open the HTML report locally
open temp/<run_name>/Reporting/report.html

# Load best.pth.tar for inference / play
uv run python -c "import torch; print(torch.load('temp/<run_name>/Nets/best.pth.tar').keys())"
```

---

## PowerShell mangling — the quoting tax

`ssh <gpu-host> '...'` lands in **Windows PowerShell**, not bash. Everything inside the single quotes is parsed by PowerShell before being handed to `wsl -- bash -lc "..."`. PowerShell will eat:

| PowerShell sees | What it does |
|-----------------|--------------|
| `$HOME` | Substitutes `C:\Users\<windows-user>` *with the slashes stripped* — e.g. `C:Users<windows-user>`. **Always use literal absolute paths** in WSL commands (`/home/<wsl-user>/...`). |
| `()` empty parens | Tries to interpret as a function call. Breaks Python like `torch.cuda.is_available()`. |
| `**` | Tries to interpret as a wildcard. Breaks Python like `total_memory >> 30`. |
| `\"foo bar\"` | Eats the escape, then re-splits on space. Breaks pytest filters like `-m "not slow"`. |
| `&&` / `||` | Not recognised as separators in older PowerShell. Use `;` for sequencing, or split into separate `ssh` calls. |

**The general escape hatch: base64-encode any Python you want to run remotely** and pipe through `base64 -d`:

```bash
SCRIPT_B64=$(echo "import torch; print(torch.cuda.is_available())" | base64)
ssh <gpu-host> "wsl -d Ubuntu -- bash -lc 'source /home/<wsl-user>/.local/bin/env && echo $SCRIPT_B64 | base64 -d | uv run --project /home/<wsl-user>/AlphaBlokus python -'"
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
| `bash: line 1: C:Users<windows-user>/.local/bin/env: No such file` | PowerShell substituted `$HOME` | Use literal `/home/<wsl-user>/...` |
| `An expression was expected after '('` | PowerShell parsed `()` in your Python | Base64-encode the Python |
| nvidia-smi from Windows shows no python process | Windows nvidia-smi only sees Windows processes | Query inside WSL: `wsl -- bash -lc "nvidia-smi"` |
| Two W&B runs with the same name | `wandb.init(name=config.run_name)` collides on re-runs | We now append a UTC timestamp suffix — see `core/storage.py` `_init_wandb` |
| `ssh: Operation timed out` to `<gpu-host>` | PC dropped off the tailnet (sleep, Tailscale service stopped, VPN interference) — `tailscale status` shows `offline, last seen Xh ago` | See [Tailscale connectivity recovery](#tailscale-connectivity-recovery) below |
| PC `active` in `tailscale status` but `tailscale ping` and SSH both time out | Mac-side Tailscale system extension stuck — typically a `MagicSock function ReceiveIPv4 is not running` health warning | Force-kill **both** `Tailscale` and `IPNExtension` on the Mac, then relaunch — see recovery section below |

---

## Tailscale connectivity recovery

The Tailscale tunnel between Mac and PC has two failure modes that look superficially the same (SSH times out) but need very different fixes.

### Read the health check first

Always start with:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale status
```

The line for `desktop-5nut9e9` (the PC) tells you which side is broken:

| Status line for PC | Meaning | Where to fix |
|--------------------|---------|--------------|
| `offline, last seen Xh ago` | PC is not talking to the control plane at all | **PC side** — see "PC-side recovery" below |
| `active; relay "lhr"` *(or similar)* with a `# Health check:` warning about MagicSock | Control plane sees both nodes but Mac-side Tailscale internals are wedged | **Mac side** — see "Mac-side recovery" below |
| `active` with no warnings, but `tailscale ping` still times out | Usually a VPN or firewall blocking UDP between the two | Disable VPN, check OS firewall |

Also run `/Applications/Tailscale.app/Contents/MacOS/Tailscale netcheck` to confirm UDP works at all and which DERP relays are reachable. If `UDP: false`, no Tailscale path will work — it's a local network issue, not a Tailscale issue.

### PC-side recovery (PC offline in tailscale status)

Most common cause is PC sleep + Tailscale tray icon dropping its connection.

1. Wake the PC manually (move mouse / open lid).
2. Check Windows system tray — Tailscale icon should be green. If red/missing, right-click → "Connect" (or restart the service).
3. If still offline 30s after waking, RDP/console into the PC and from admin PowerShell:
   ```powershell
   Restart-Service Tailscale
   ```
4. Verify from the Mac: `tailscale status | grep desktop` should now show `active`.

### Mac-side recovery (PC active but no traffic)

This is the harder one. Symptoms: PC shows `active` in `tailscale status`, but `tailscale ping` to the PC times out, SSH times out, and there's a Mac-side health warning like:

```
# Health check:
#     - The MagicSock function ReceiveIPv4 is not running.
```

**Why the menu-bar "Quit" doesn't fix it:** the macOS Tailscale app is two processes — the GUI app *and* a separately-launched system extension (`IPNExtension`). Quitting from the menu only kills the GUI. The extension keeps running with its broken state, and when you re-open the app it inherits that state. You have to kill **both**.

```bash
# Kill the GUI and the system extension
killall Tailscale IPNExtension

# Wait a moment, confirm gone
sleep 2
ps aux | grep -iE "tailscale|ipnextension" | grep -v grep
# Should be empty

# Relaunch
open /Applications/Tailscale.app
```

When the app comes back up it has no in-memory auth token, so a fresh login flow opens in the default browser. Click **Connect** to re-authorise — within ~5s `tailscale status` should show `active` with no health warning. Confirm with:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale ping -c 2 100.109.47.123
```

If you see `pong from desktop-5nut9e9` (with or without "via DERP"), the tunnel is back. SSH should work immediately.

**If even `killall` doesn't fix it** (rare): the system extension can occasionally hang in a state only `sudo killall IPNExtension` can clear, or only a reboot can clear. Reboot is the universal fix when nothing else works.

### A note on VPNs

Several Tailscale connectivity issues we've hit have traced back to a corporate / always-on VPN being silently re-enabled after a network change or schedule. If `tailscale netcheck` shows `UDP: false` or the DERP relays all have high latency / fail, check whether a VPN client decided to re-engage. Tailscale and most corporate VPNs fight over the routing table.

### What the symptoms looked like the first time we hit this

(Documenting for future-us so the next person doesn't have to re-derive the diagnosis.)

- Mac tailscale status showed PC `active; relay "lhr"`, no `offline` flag.
- `tailscale ping` to PC: `ping ... timed out` (no reply).
- `tailscale status` had health warning: `The MagicSock function ReceiveIPv4 is not running. You might experience connectivity issues.`
- `curl https://controlplane.tailscale.com/health` returned a clean HTTP 404 in 60 ms (TLS verified OK) — proves Tailscale's infrastructure was reachable, so the issue was local-Mac, not network.
- `scutil --proxy` showed default config, no env proxy variables — ruling out a transparent proxy.
- The Mac had two Tailscale processes: main app + `IPNExtension`. Quitting via the menu killed the main app but left the extension stuck.
- `killall Tailscale IPNExtension && open /Applications/Tailscale.app` triggered a fresh browser login. Clicking Connect restored everything within seconds.

Total time spent debugging this the first time: ~30 minutes. With this section, hopefully under 2 next time.

---

## Killing a stuck or runaway run

```bash
# Find the process
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'

# Kill it cleanly (lets Coach._learn_loop's finally close W&B)
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pkill -INT -f main.py"'

# If that doesn't work after 30s
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pkill -9 -f main.py"'

# Confirm gone
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'
```

If WSL itself is wedged: `ssh <gpu-host> 'wsl --shutdown'` from PowerShell, then re-launch. This wipes `/tmp` but leaves `~/` intact (training results in `~/AlphaBlokus/temp/` survive).

---

## Updating code on the PC mid-session

```bash
# From dev machine: push your changes to the feature branch
git push

# On PC: pull and verify
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && git pull && git log --oneline -3"'
```

If there's an in-flight training run when you pull, the on-disk code change won't affect it (Python has already imported its modules) — the next launch picks up the new code.

---

## Sanity-check commands to keep handy

```bash
# "Is the PC alive at all?"
ssh <gpu-host> 'echo ok'

# "Is WSL alive?"
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "uptime; whoami"'

# "Is CUDA working?"
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"'

# "Is there a training run going?"
ssh <gpu-host> 'wsl -d Ubuntu -- bash -lc "pgrep -af main.py"'

# "What's the latest W&B run?"
ssh <gpu-host> "wsl -d Ubuntu -- bash -lc 'cd /home/<wsl-user>/AlphaBlokus && uv run python -c \"import wandb; print(list(wandb.Api().runs(\\\"<wandb-entity>/<project>\\\", per_page=1))[0].url)\"'"
```

The last one's quote-nesting is painful — prefer the base64 pattern from the "PowerShell mangling" section above for anything more complex.

---

## When something this guide doesn't cover comes up

Update this guide. Operational knowledge erodes fast — if you find yourself debugging the same WSL/PowerShell/W&B oddity twice, add a row to one of the tables above. The point of having this file is so the next session doesn't have to relearn it.
