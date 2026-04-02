# Remote Training Setup — MacBook Steps

You are setting up a MacBook to remotely SSH into a Windows PC (with RTX 3080 Ti GPU) for running ML training jobs. The architecture is:

- **Tailscale** — encrypted mesh VPN that gives both machines stable private IPs (`100.x.x.x`). Works through any WiFi/hotspot/firewall without port forwarding.
- **SSH with key auth** — the Mac connects to the PC's Tailscale IP. Password auth is disabled; only your ed25519 key works.
- **tmux** — on the PC side, training runs inside tmux so they survive disconnections. You detach, close the laptop, reconnect later.

A separate Windows PC plan handles the other end. Some steps require coordination — these are marked with **SYNC POINT**.

---

## Checklist

| # | Step | Effort | Done |
|---|------|--------|------|
| M1 | Install Tailscale | 5 min | |
| M2 | **SYNC: get Tailscale IP and Windows username from PC** | 1 min | |
| M3 | Generate SSH key (if needed) | 2 min | |
| M4 | Copy SSH key to PC | 5 min | |
| M5 | **SYNC: confirm key auth works, tell PC to disable passwords** | 2 min | |
| M6 | Set up SSH config for convenience | 5 min | |
| M7 | **SYNC: reboot test — verify connection after PC restart** | 5 min | |
| M8 | End-to-end test: run training smoke test | 10 min | |

---

## M1. Install Tailscale

```bash
brew install tailscale
```

Or download from https://tailscale.com/download/mac.

Open the Tailscale app and sign in. **Use the same account as the PC** (Google, GitHub, etc.).

---

## M2. SYNC POINT: Get info from the PC

The PC side needs to give you:
- **Tailscale IP**: `100.___.___.___`
- **Windows username**: `___________`

Verify you can see the PC:
```bash
tailscale status
```

Should list both machines. Try pinging:
```bash
tailscale ping <pc-tailscale-ip>
```

---

## M3. Generate SSH key

If you already have `~/.ssh/id_ed25519`, skip this.

```bash
ssh-keygen -t ed25519 -C "henry-macbook"
```

Accept the default path. Set a passphrase if you want (optional but recommended).

---

## M4. Copy SSH key to PC

The PC should have SSH running with password auth enabled at this point.

```bash
ssh-copy-id <windows-username>@<pc-tailscale-ip>
```

Enter the **Windows** password when prompted.

Verify it works without a password:
```bash
ssh <windows-username>@<pc-tailscale-ip>
```

You should land in a Ubuntu (WSL2) shell with no password prompt. Type `exit` to disconnect.

---

## M5. SYNC POINT: Confirm and lock down

Tell the PC side: "Key auth works, go ahead and disable password auth."

After they disable it, verify you can still connect:
```bash
ssh <windows-username>@<pc-tailscale-ip>
```

Should still work (using your key). If it doesn't, something went wrong with the key copy — tell the PC side to re-enable password auth and retry M4.

---

## M6. Set up SSH config

Edit `~/.ssh/config` (create it if it doesn't exist):

```
Host gpu
    HostName <pc-tailscale-ip>
    User <windows-username>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 5
    TCPKeepAlive yes
```

**Replace** `<pc-tailscale-ip>` and `<windows-username>` with the real values from M2.

`ServerAliveInterval 30` sends a keepalive every 30 seconds — critical for hotel WiFi and phone hotspots. Prevents silent disconnections.

Now you can connect with just:
```bash
ssh gpu
```

---

## M7. SYNC POINT: Reboot test

The PC side will reboot and NOT log in. Wait 2 minutes, then:

```bash
ssh gpu
```

If it connects, run:
```bash
nvidia-smi
```

Should show the RTX 3080 Ti. If this works, the PC is fully set up for unattended remote access.

---

## M8. End-to-end test

```bash
# Quick GPU check
ssh gpu "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"

# Run the TicTacToe smoke test on GPU
ssh gpu
cd ~/AlphaBlokus
tmux new -s test
uv run python main.py --config run_configurations/smoke_test.json
# Wait for it to complete (~10 seconds)
# Ctrl+B, D to detach
exit
```

---

## Daily Workflow (from Sri Lanka, Bangalore, anywhere)

### Start a training run
```bash
ssh gpu
cd ~/AlphaBlokus
git pull                  # get latest code
tmux new -s train         # create a named session
uv run python main.py --config run_configurations/blokus_minimal.json
# Ctrl+B, D to detach     # training keeps running
exit                       # close SSH
```

### Reconnect and check progress
```bash
ssh gpu
tmux attach -t train      # reattach to the session
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

### If connection drops
No problem. tmux keeps running on the PC. Just reconnect:
```bash
ssh gpu
tmux attach -t train
```

### If the PC rebooted (power cut, Windows update, etc.)
```bash
ssh gpu
tmux ls                    # probably no sessions
cd ~/AlphaBlokus
tmux new -s train
# Resume training — Coach saves checkpoints every generation
uv run python main.py --config run_configurations/blokus_minimal.json
```

---

## Optional: Install mosh for flaky connections

If SSH keeps dropping on bad WiFi, `mosh` handles roaming and sleep much better:

**On the PC** (via SSH):
```bash
sudo apt install mosh
```

**On the Mac:**
```bash
brew install mosh
```

**Connect:**
```bash
mosh gpu
```

mosh uses UDP and handles IP changes seamlessly — perfect for switching between WiFi networks or going through tunnels.
