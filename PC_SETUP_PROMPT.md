# Prompt to paste into PC-side Claude Code

Hand this whole thing (everything between the `---` markers) verbatim to the Claude Code running on the home PC. It's self-contained — sets up a persistent WSL keep-alive then launches the Blokus training in a way that survives Claude / shell sessions ending.

Delete this file once the run is launched (it's a one-time ops note, not source).

---

I need to set up a persistent WSL keep-alive so detached training runs don't die when WSL2 power-cycles the VM, then launch a training run that survives across Claude / shell session shutdowns.

**Step 1 — Verify environment.** Confirm:

- WSL Ubuntu is installed (`wsl --list`)
- The `~/AlphaBlokus` repo exists inside WSL, on branch `feat/blokus-symmetries`, latest commit pulled
- `loginctl show-user henry` shows `Linger=yes`
- `nvidia-smi` from inside WSL shows the RTX 3060 Ti

**Step 2 — Install a WSL keep-alive Windows scheduled task.** From an elevated PowerShell on Windows:

```powershell
# Force WSL to reload .wslconfig (in case vmIdleTimeout=-1 hasn't taken effect)
wsl --shutdown
Start-Sleep -Seconds 4

# Register a Windows scheduled task that fires at every login and runs
# `wsl sleep` inside Ubuntu indefinitely. This is what keeps the VM up —
# systemd-user --linger only keeps the user manager alive *inside* WSL,
# not WSL itself.
$action = New-ScheduledTaskAction -Execute "wsl.exe" `
  -Argument "-d Ubuntu -- sleep 99999999"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 365) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

Register-ScheduledTask -TaskName "WSL Keepalive" -Action $action `
  -Trigger $trigger -Settings $settings -Principal $principal -Force

# Fire it once now so we don't have to log out / in
Start-ScheduledTask -TaskName "WSL Keepalive"
Start-Sleep -Seconds 5

# Verify the keepalive sleep is running inside WSL
wsl -d Ubuntu -- pgrep -af "sleep 99999999"
```

Expected output: one line showing `sleep 99999999` running with a PID.

**Step 3 — Verify the keepalive holds across a fresh `wsl` invocation.** Wait 30 seconds, then run `wsl -d Ubuntu -- uptime` from PowerShell. The uptime should be at least 30 seconds (proving WSL didn't reboot in between).

**Step 4 — Launch the Blokus training.** Inside WSL (`wsl -d Ubuntu`), from `~/AlphaBlokus`:

```bash
# Make sure the runner script is there and executable
ls -la scripts/run_blokus_pc_first.sh

# Clean any prior service + temp data for this run
systemctl --user stop alphablokus_blokus_first.service 2>/dev/null || true
systemctl --user reset-failed alphablokus_blokus_first.service 2>/dev/null || true
rm -rf temp/blokus_pc_first temp/blokus_pc_first.log

# Launch detached. systemd-user + linger + the Windows keepalive task
# together make this survive everything except a Windows reboot.
systemd-run --user --unit=alphablokus_blokus_first \
  --working-directory=/home/henry/AlphaBlokus \
  --setenv=PYTHONUNBUFFERED=1 \
  bash -lc "/home/henry/AlphaBlokus/scripts/run_blokus_pc_first.sh > /home/henry/AlphaBlokus/temp/blokus_pc_first.log 2>&1 < /dev/null"

sleep 5
systemctl --user is-active alphablokus_blokus_first.service
systemctl --user status alphablokus_blokus_first.service --no-pager | head -10
```

**Step 5 — Confirm it's still running after a session-ending check.** Wait 90 seconds (this is when previous attempts died — WSL's idle reaper would have fired by now). Then:

```bash
wsl -d Ubuntu -- systemctl --user is-active alphablokus_blokus_first.service
wsl -d Ubuntu -- pgrep -af "python.*blokus_pc_first" | head -3
wsl -d Ubuntu -- tail -c 800 /home/henry/AlphaBlokus/temp/blokus_pc_first.log | tr '\r' '\n' | tail -5
```

Service should still be `active`, python should be running, log should show self-play progress past the first game.

**Step 6 — Report back.** Tell me: (a) which steps passed/failed, (b) the W&B run URL (grep for it in the log), (c) what the log's last line looks like 90 seconds in. If anything fails at Step 5, dump `journalctl --user -u alphablokus_blokus_first.service --no-pager | tail -30` and `dmesg | tail -20`.

---

## Notes for Henry

- **Why this needs to happen on the PC and not over SSH:** WSL2 power-cycles the VM ~30 seconds after the last active session disconnects, regardless of linger or `vmIdleTimeout=-1`. The scheduled task spawns a process via Windows itself, which keeps WSL alive permanently. SSH-launched processes can't establish that anchor reliably.
- **After this setup is done once,** every future PC run uses the existing `systemd-run --user` pattern directly — no extra ceremony needed.
- **If anything goes sideways,** screenshot the PC-Claude's response and paste it back to me here. The most likely failure is the scheduled task creation needing admin privileges — open PowerShell with "Run as administrator."
- **W&B is the easy way to watch the run from your laptop** once it's launched — open https://wandb.ai/henrycassidy/alphablokus-poc and find the `blokus_pc_first_*` run.
