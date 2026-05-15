# Remote Training Setup

> **Archived 2026-05-15.** Overview doc for the original remote setup. Network/SSH layer is functionally complete (Tailscale, WSL2, OpenSSH, key auth all working). Remaining operational items — Windows default-shell to WSL2, unattended boot, repo install on the PC, GPU smoke test — are tracked in `docs/plans/gpu-training-poc.md`. The companion mac- and windows-side step lists are also archived alongside this doc.

---


SSH from MacBook (anywhere in the world) to personal Windows PC (RTX 3060 Ti, at home) for AlphaBlokus training runs. Uses Tailscale (encrypted mesh VPN) + WSL2 (Linux on Windows) + tmux (persistent sessions).

**Scenario:** Henry is travelling (Sri Lanka/Bangalore), connecting from hotel WiFi, phone hotspots, airport lounges. Needs rock-solid security and persistence.

---

## Checklist

| # | Step | Where | Effort | Done |
|---|------|-------|--------|------|
| T1 | Install WSL2 with Ubuntu on PC | PC | 15 min | |
| T2 | Verify CUDA works inside WSL2 | PC | 10 min | |
| T3 | Install Tailscale on PC | PC | 5 min | |
| T4 | Install Tailscale on MacBook | Mac | 5 min | |
| T5 | Enable Windows OpenSSH Server, set default shell to WSL2 | PC | 15 min | |
| T6 | Set up SSH key auth and disable passwords | Both | 10 min | |
| T7 | Clone repo, install deps, install tmux in WSL2 | PC | 10 min | |
| T8 | Configure PC for unattended operation | PC | 10 min | |
| T9 | Set up SSH config on MacBook | Mac | 5 min | |
| T10 | End-to-end test | Mac | 10 min | |

**Do all of this at home before travelling.**

---

## T1. Install WSL2 with Ubuntu

On the PC, open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu
```

Reboot when prompted. On first launch, create a Unix username/password.

Verify:
```powershell
wsl --list --verbose
```
Should show `Ubuntu` with `VERSION 2`.

---

## T2. Verify CUDA works inside WSL2

Open WSL2 (type `wsl` in PowerShell):
```bash
nvidia-smi
```

Should show the RTX 3060 Ti. No separate Linux NVIDIA driver needed — WSL2 uses the Windows driver automatically.

Install build tools:
```bash
sudo apt update && sudo apt install -y build-essential
```

---

## T3. Install Tailscale on PC

Download from https://tailscale.com/download/windows. Install and sign in (use Google or GitHub account).

In Tailscale settings:
- Enable **"Run unattended"** — keeps Tailscale connected even when no user is logged into Windows
- Note the Tailscale IP assigned (e.g. `100.64.x.y`) — this is permanent

---

## T4. Install Tailscale on MacBook

```bash
brew install tailscale
```

Or download from https://tailscale.com/download/mac.

Sign in with the **same account** as the PC.

Verify both machines are connected:
```bash
tailscale status
```

Should list both machines with their `100.x.x.x` IPs.

---

## T5. Enable Windows OpenSSH Server

On the PC, PowerShell as Administrator:
```powershell
# Install OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and enable on boot
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# Set default shell to WSL2 so SSH drops straight into Ubuntu
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\wsl.exe" -PropertyType String -Force
```

---

## T6. SSH key auth and disable passwords

**On the MacBook** — generate a key if you don't have one:
```bash
ssh-keygen -t ed25519 -C "henry-macbook"
```

Copy it to the PC (password auth still enabled at this point):
```bash
ssh-copy-id <windows-username>@<pc-tailscale-ip>
```

Note: use the **Windows** username, not the WSL2 one.

Verify key auth works:
```bash
ssh <windows-username>@<pc-tailscale-ip>
```

Should log in without a password prompt.

**Now disable password auth.** On the PC, edit `C:\ProgramData\ssh\sshd_config`:
```
PasswordAuthentication no
PubkeyAuthentication yes
```

Restart SSH:
```powershell
Restart-Service sshd
```

**Security note:** With Tailscale + key-only SSH, the PC has zero attack surface. No ports are open to the internet. Only devices on your Tailscale account can even reach the SSH port. And they need your private key to authenticate.

---

## T7. Clone repo and install dependencies

```bash
ssh <windows-username>@<pc-tailscale-ip>
cd ~
git clone <repo-url> AlphaBlokus
cd AlphaBlokus
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync

# Verify
uv run pytest -m "not slow"

# Install tmux
sudo apt install -y tmux
```

---

## T8. Configure PC for unattended operation

The PC needs to stay on and connected while you're away.

**Power settings** (Windows):
- Settings → System → Power → set "When plugged in, turn off screen after: Never"
- Set "When plugged in, put my device to sleep after: Never"

**Start WSL2 on boot** (PowerShell as Admin):
```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu -- echo started"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Start WSL on Boot" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
```

**Test:** Reboot the PC, don't log in, wait 2 minutes, then from your MacBook:
```bash
ssh <windows-username>@<pc-tailscale-ip>
nvidia-smi
```

If this works, you're set for unattended remote access.

---

## T9. SSH config on MacBook

Edit `~/.ssh/config`:
```
Host gpu
    HostName <pc-tailscale-ip>
    User <windows-username>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 5
    TCPKeepAlive yes
```

`ServerAliveInterval 30` sends a keepalive every 30 seconds — critical for flaky hotel WiFi. Prevents the connection from silently dying.

Now you can just type: `ssh gpu`

---

## T10. End-to-end test

```bash
# Verify GPU
ssh gpu "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"

# Run smoke test
ssh gpu "cd ~/AlphaBlokus && uv run python main.py --config run_configurations/smoke_test.json"
```

---

## Daily Workflow (from anywhere)

### Start a training run
```bash
ssh gpu
cd ~/AlphaBlokus
git pull
tmux new -s train
uv run python main.py --config run_configurations/blokus_minimal.json
# Ctrl+B, D to detach
```

### Reconnect later
```bash
ssh gpu
tmux attach -t train
```

### Quick status check (without attaching)
```bash
ssh gpu "tmux capture-pane -t train -p | tail -10"
```

### Pull results back to MacBook
```bash
rsync -avz gpu:~/AlphaBlokus/temp/<run_name>/Reporting/ ./temp/<run_name>/Reporting/
rsync -avz gpu:~/AlphaBlokus/temp/<run_name>/Nets/best.pth.tar ./temp/<run_name>/Nets/
```

### If tmux session disappeared (PC rebooted, etc.)
```bash
ssh gpu
tmux ls                    # check if any sessions exist
cd ~/AlphaBlokus
tmux new -s train          # start fresh
# training has checkpoints, so resume from where it left off
```

---

## Troubleshooting

**Can't connect at all:**
- Check Tailscale app on both machines — both should show "Connected"
- PC might have rebooted and WSL2 not started — the scheduled task should handle this but verify
- Try `tailscale ping <pc-tailscale-ip>` from MacBook

**Connection drops frequently:**
- Decrease `ServerAliveInterval` to 15 in SSH config
- Consider installing `mosh` (mobile shell) — handles roaming/sleep much better than SSH:
  - PC: `sudo apt install mosh`
  - Mac: `brew install mosh`
  - Connect: `mosh gpu` instead of `ssh gpu`

**GPU not visible after reconnect:**
- `nvidia-smi` fails → restart WSL2: from Windows, `wsl --shutdown`, then reconnect

**Training crashed while disconnected:**
- Check `tmux ls` — session should still exist with the error visible
- Check logs: `cat ~/AlphaBlokus/temp/<run_name>/Logs/alpha.log`
