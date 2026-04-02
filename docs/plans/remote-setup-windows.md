# Remote Training Setup — Windows PC Steps

You are setting up a Windows PC (with RTX 3080 Ti GPU) to be remotely accessible via SSH from a MacBook anywhere in the world. The architecture is:

- **Tailscale** — encrypted mesh VPN that gives both machines stable private IPs (`100.x.x.x`). No port forwarding, no exposure to the public internet.
- **WSL2** — runs Ubuntu inside Windows, provides a Linux environment with native CUDA GPU access.
- **OpenSSH** — Windows SSH server, configured to drop straight into the WSL2 Ubuntu shell.
- **tmux** — terminal multiplexer inside WSL2 so training processes survive SSH disconnections.

A separate MacBook plan handles the other end. Some steps require coordination — these are marked with **SYNC POINT**.

---

## Checklist

| # | Step | Effort | Done |
|---|------|--------|------|
| W1 | Install WSL2 with Ubuntu | 15 min | |
| W2 | Verify CUDA works inside WSL2 | 10 min | |
| W3 | Install Tailscale | 5 min | |
| W4 | Enable Windows OpenSSH Server, set default shell to WSL2 | 15 min | |
| W5 | **SYNC: receive SSH public key from Mac, install it** | 5 min | |
| W6 | Disable password auth | 5 min | |
| W7 | Clone repo, install dependencies, install tmux | 10 min | |
| W8 | Configure for unattended operation (no sleep, WSL on boot) | 10 min | |
| W9 | **SYNC: reboot test — Mac verifies it can connect** | 5 min | |

---

## W1. Install WSL2 with Ubuntu

Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu
```

Reboot when prompted. On first launch, Ubuntu will ask you to create a username and password. Remember these.

Verify:
```powershell
wsl --list --verbose
```
Should show `Ubuntu` with `VERSION 2`.

---

## W2. Verify CUDA works inside WSL2

Open WSL2 (type `wsl` in PowerShell):
```bash
nvidia-smi
```

Should show the RTX 3080 Ti. No separate Linux NVIDIA driver needed — WSL2 uses the Windows driver automatically.

Install build tools:
```bash
sudo apt update && sudo apt install -y build-essential
```

---

## W3. Install Tailscale

1. Download from https://tailscale.com/download/windows
2. Install and sign in (use Google or GitHub — **same account the Mac will use**)
3. In Tailscale settings, enable **"Run unattended"** — this keeps Tailscale connected even when no one is logged into Windows
4. Note the Tailscale IP assigned (looks like `100.64.x.y`) — write this down, the Mac needs it

**Give the Mac person this information:**
- Your Tailscale IP: `100.___.___.___`
- Your Windows username: `___________`

---

## W4. Enable Windows OpenSSH Server

PowerShell as Administrator:
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

At this point, SSH is running with password auth enabled. The Mac side will connect and copy over a key in the next step.

---

## W5. SYNC POINT: Receive SSH key from Mac

**Wait for the Mac side to run `ssh-copy-id`.** They will SSH in using your Windows username and the Tailscale IP, using your Windows password.

Once they confirm key auth works (they can log in without a password prompt), proceed to W6.

---

## W6. Disable password auth

Edit `C:\ProgramData\ssh\sshd_config` (use Notepad as Administrator):

Find and change these lines:
```
PasswordAuthentication no
PubkeyAuthentication yes
```

Restart SSH in PowerShell:
```powershell
Restart-Service sshd
```

**Security:** With Tailscale + key-only SSH, the PC has zero attack surface. No ports are exposed to the internet. Only devices on your Tailscale account can reach the SSH port, and they need the private key.

---

## W7. Clone repo and install dependencies

SSH should now be working. You can do this step from the WSL2 terminal directly, or let the Mac side do it over SSH. From inside WSL2:

```bash
cd ~
git clone <repo-url> AlphaBlokus
cd AlphaBlokus

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install project dependencies
uv sync

# Verify tests pass
uv run pytest -m "not slow"

# Verify GPU works from Python
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install tmux for persistent sessions
sudo apt install -y tmux
```

---

## W8. Configure for unattended operation

The PC needs to stay on and accessible while no one is physically present.

**Power settings:**
- Settings → System → Power → "When plugged in, turn off screen after": **Never**
- "When plugged in, put my device to sleep after": **Never**

**Start WSL2 automatically on boot** (PowerShell as Admin):
```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu -- echo started"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Start WSL on Boot" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
```

---

## W9. SYNC POINT: Reboot test

1. Reboot the PC
2. **Do NOT log in** — leave it at the Windows lock screen
3. Wait 2 minutes for WSL2 and Tailscale to start
4. Tell the Mac side to try connecting: `ssh gpu`

If they can connect and `nvidia-smi` works, you're done. The PC is ready for remote training.
