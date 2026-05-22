# Launching the Blokus PC run — step-by-step

Read top to bottom. Each step says what to do, where to do it, and what success looks like. Total time: ~5 minutes setup, then training runs for ~1.5 hours on its own.

You're doing two things:

1. **One-time setup**: register a Windows scheduled task that keeps WSL alive forever (so detached training doesn't die when WSL2 reaps the idle VM). Future PC runs won't need this part again.
2. **Launch the run**: spawn the training as a `systemd-run --user` service inside WSL.

---

## Step 1 — Open PowerShell as Administrator

Press `Windows`, type `PowerShell`, right-click → **Run as administrator**. Click Yes on the UAC prompt.

You should see a blue window with `PS C:\WINDOWS\system32>`.

> **Why admin:** Registering a scheduled task that runs at logon needs elevated rights.

---

## Step 2a — Stop Windows from sleeping

If the PC sleeps mid-training the GPU loses power and the run dies. Disable sleep/hibernate when plugged in:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

`0` = never. Nothing prints — silent success. (Re-enable later with non-zero minute values if you want.)

> **Heads-up:** lock-screen is fine, training keeps running. *Signing out* will kill it (don't do that). The display also turns off when idle but the machine stays awake.

---

## Step 2b — Force WSL to reload its config

Paste this and press Enter:

```powershell
wsl --shutdown
Start-Sleep -Seconds 4
```

Nothing prints. That's fine — it just shuts down all running WSL distros so the next `wsl` command starts fresh with the latest `.wslconfig`.

---

## Step 3 — Register the WSL keep-alive task

Paste this whole block into PowerShell and press Enter:

```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" `
  -Argument "-d Ubuntu -- sleep 99999999"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Days 365) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

Register-ScheduledTask -TaskName "WSL Keepalive" -Action $action `
  -Trigger $trigger -Settings $settings -Principal $principal -Force
```

**Expected output:** a table showing the task with `State: Ready`. If you see a red error mentioning `Access is denied`, you didn't open PowerShell as admin — go back to Step 1.

---

## Step 4 — Fire the task once now

Still in PowerShell:

```powershell
Start-ScheduledTask -TaskName "WSL Keepalive"
Start-Sleep -Seconds 5
wsl -d Ubuntu -- pgrep -af "sleep 99999999"
```

**Expected output:** one line like `1234 sleep 99999999`. The number is the process ID — anything works. If you see no output, the task didn't fire. Open `taskschd.msc`, find "WSL Keepalive" in the task list, right-click → Run, then retry the `pgrep` check.

---

## Step 5 — Sanity check: WSL really is staying up

Wait 30 seconds (count to 30, sip coffee), then paste:

```powershell
wsl -d Ubuntu -- uptime
```

**Expected output:** something like `09:42:11 up 0:01, 0 users, load average: 0.05, 0.03, 0.00`. The `up 0:01` (one minute or more) confirms WSL didn't reboot. If `uptime` says `up 0:00` or some single-digit seconds value, the keepalive isn't holding — paste the output back to me.

---

## Step 6 — Pull the latest code inside WSL

In PowerShell (still):

```powershell
wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && git fetch origin && git checkout feat/blokus-symmetries && git pull --ff-only && git log --oneline -1"
```

**Expected output:** the last line should be `f4f9da5` or newer commit on `feat/blokus-symmetries`. If git complains, paste back what it said.

---

## Step 7 — Launch the training

Still in PowerShell. This is one long command — paste exactly:

```powershell
wsl -d Ubuntu -- bash -lc "cd ~/AlphaBlokus && systemctl --user stop alphablokus_blokus_first.service 2>/dev/null; systemctl --user reset-failed alphablokus_blokus_first.service 2>/dev/null; rm -rf temp/blokus_pc_first temp/blokus_pc_first.log; systemd-run --user --unit=alphablokus_blokus_first --working-directory=/home/henry/AlphaBlokus --setenv=PYTHONUNBUFFERED=1 bash -lc './scripts/run_blokus_pc_first.sh > temp/blokus_pc_first.log 2>&1 < /dev/null'; sleep 5; systemctl --user is-active alphablokus_blokus_first.service"
```

**Expected output:** the last line should be `active`. If it says `inactive` or `failed`, something broke at startup — paste back the full output.

---

## Step 8 — The crucial check: still running after WSL would've timed out

The previous attempts died at 30 seconds because WSL was reaping the VM. The keepalive should now prevent that. Wait **120 seconds** (give it plenty), then paste:

```powershell
wsl -d Ubuntu -- bash -lc "systemctl --user is-active alphablokus_blokus_first.service; pgrep -af 'python.*blokus_pc_first' | head -3; tail -c 400 temp/blokus_pc_first.log | tr '\r' '\n' | tail -5"
```

**Expected output:** roughly

```
active
12345 uv run python main.py --config run_configurations/blokus_pc_first.json
12348 /home/henry/AlphaBlokus/.venv/bin/python3 main.py --config run_configurations/blokus_pc_first.json
Self Play:   4%|▍         | 1/25 [00:36<14:00, 35.00s/it]
Self Play:   8%|▊         | 2/25 [01:11<...]
```

If you see `active` plus python processes plus a non-zero `Self Play: N/25` line, **you've won**. The run is detached and going. You can close PowerShell, close your laptop, whatever — it keeps going until done.

If you see `inactive` or no python processes, that's the same failure as before. Paste back the output and I'll dig.

---

## Step 9 — Find the W&B URL

```powershell
wsl -d Ubuntu -- grep "View run" temp/blokus_pc_first.log | head -1
```

You'll get a line like `wandb: 🚀 View run at https://wandb.ai/henrycassidy/alphablokus-poc/runs/...`. Click that URL — that's your live dashboard.

---

## What to expect from here

- The run takes about 1–1.5 hours on the 3060 Ti.
- W&B updates live, refresh whenever.
- When done, the service goes to `inactive (dead)` cleanly.
- Pull the data back to your Mac with `rsync`, then re-render the HTML report locally (commands at the end of `docs/guides/REMOTE-TRAINING.md`).

Once everything's running, **paste me back the W&B URL** and I'll pick it up from there.
