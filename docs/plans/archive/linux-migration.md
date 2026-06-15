# Linux Migration — dual-boot Ubuntu on the PC

> **Status: COMPLETED 2026-06-12 — archived 2026-06-15.** All steps LM0–LM10 done;
> the box now dual-boots Ubuntu 24.04 (`gpu-linux`) and a full training run has
> since completed on it. Retained as the migration record.

Subplan of [self-play-throughput.md](../self-play-throughput.md) **P1**. Move the training
box from Windows to a native **Ubuntu dual-boot**, for: clean warmup-free
benchmarking, `forkserver` (cheap workers), stable unattended multi-hour runs over
SSH+tmux, and PyTorch's first-class platform. Windows stays installed and bootable;
you pick the OS at power-on.

---

## PC facts (these drive the plan + the risks)

Measured 2026-06-07, re-verified 2026-06-10; key items re-verified again 2026-06-12 (migration day):
- **Windows 11 Home**, **UEFI + GPT**, **Secure Boot ON** (`Confirm-SecureBootUEFI` → True, re-verified 2026-06-12).
- **Single disk** — Samsung 970 EVO Plus **1 TB NVMe**, **~630 GB free** on C: (no spare drive). `Get-PartitionSupportedSize` says C: can shrink by up to ~657 GB — a 300 GB shrink is comfortably in range (verified 2026-06-12).
- **BitLocker: OFF** — `manage-bde -status` shows *Fully Decrypted, Protection Off, no key protectors* (verified 2026-06-10, re-verified 2026-06-12). Risk 2 below is retired; LM2 is a no-op.
- **The Mac's SSH session on the box is full admin** (verified 2026-06-12) — so Claude can run `Resize-Partition` remotely; LM4 is no longer a hands-on step.
- 32 GB RAM, i5-13600KF (6 P + 8 E = 14C/20T), RTX 3060 Ti 8 GB.

The single-disk + Secure-Boot + Win11 combination sets **three real risks** — handle them in order or the migration can go badly:

1. **Single disk → we must *shrink* the one Windows partition** (no second drive to install onto). Repartitioning a live disk carries a small but real data-loss risk → **back up first**.
2. ~~**Win11 likely has device encryption (BitLocker) on.**~~ **Retired 2026-06-10:** verified fully decrypted with protection off — repartitioning and BIOS changes carry no lockout risk on this box.
3. **Secure Boot ON complicates the NVIDIA proprietary driver.** Either disable Secure Boot in BIOS (simplest) or complete Ubuntu's **MOK enrollment** prompt at first reboot.

Use **Ubuntu 24.04.4 LTS** (point release Feb 2026, HWE kernel 6.17) — it schedules the 13th-gen Intel P/E hybrid cores correctly and has current NVIDIA driver support. *Re-evaluated 2026-06-10:* Ubuntu 26.04 LTS exists (Apr 2026) but is pre-first-point-release; for an NVIDIA/CUDA box the mature 24.04.x series is the right pick. Last 24.04 point release (24.04.5) lands Aug 2026, with standard support to 2029 — no urgency to chase 26.04.

**Pre-flight blocker found 2026-06-10:** the Mac's Tailscale client is **logged out** (failed re-registration; restart didn't fix it) — the GPU box is unreachable from the Mac. Henry must re-auth (menu-bar Tailscale → Log in, browser flow) *before migration day*: it's needed to re-verify the PC facts below and it gates the LM7 handoff.

---

## Time estimate: **~half a day hands-on (3–6 h)** — ~3 h if nothing snags

Honest breakdown (NVMe makes the disk steps fast; the variance is backup size + driver/BitLocker snags):

| Step | Time | Notes |
|------|------|-------|
| Back up irreplaceable data | 0–60 min | Code is in git; only personal/local data matters |
| Check + suspend BitLocker | ~10 min | + locate the recovery key |
| Make Ubuntu 24.04 USB (on the Mac) | ~20–30 min | ~5 GB ISO download + flash |
| Shrink C: ~200–300 GB | ~10–20 min | NVMe is fast |
| Disable Secure Boot in BIOS | ~5 min | or plan MOK enrollment |
| Install Ubuntu alongside Windows | ~30–45 min | UEFI, into the freed space; enable OpenSSH |
| **Make Ubuntu reachable** (Tailscale + SSH key) | ~10–15 min | the handoff — without it I can't reach the box |
| NVIDIA + CUDA drivers + verify | ~20–40 min | reboot; possible MOK step |
| Repo + `uv sync` + verify `torch.cuda` | ~15–30 min | CUDA torch is the *default* wheel on Linux — no `--torch-backend` wrinkle |
| Validation run + unattended (tmux) check | ~15–30 min | short training cycle on GPU |

**Realistic: a half-day block (4–6 h with buffer); ~3 h if the driver + BitLocker behave.** The riskiest minutes are the shrink (LM4) — that's why LM1–LM2 gate it.

**Division of labour (revised 2026-06-12):** Claude drives **LM1 (backup pull), LM3 (flash, modulo one sudo password), and LM4 (remote shrink)** from the Mac — the admin SSH session makes the shrink remote-able. Henry's hands-on steps reduce to **LM5–LM7** (BIOS toggle, USB boot + installer, and the Tailscale/SSH bootstrap) — **LM7 is the handoff**: until Ubuntu has its own SSH + Tailscale node, the Windows `gpu-cmd` route is useless (different OS). Once LM7 is done, **Claude drives LM8–LM10** (drivers, repo, uv, validation) over SSH exactly as we do now.

---

## Checklist

| # | Item | Who | Effort | Done |
|---|------|-----|--------|------|
| LM0 | **Re-login Tailscale on the Mac** (menu-bar → Log in, browser auth; client found logged out 2026-06-10) — gates re-verifying PC facts and the LM7 handoff | Henry | ~5 min | ✔ 2026-06-10 |
| LM1 | **Back up** anything irreplaceable on C: — file-level pull of Videos/Downloads/PycharmProjects/Projects (~38 GB) to the Mac over SSH; Desktop+Pictures are OneDrive-synced already | Claude (SSH) | ~30–60 min | ✔ 2026-06-12 — 42 GB at `~/pc-backup-2026-06-12/` |
| LM2 | **Check + suspend BitLocker / device encryption**; save the recovery key off-machine | Henry | ~10 min | ✔ 2026-06-10 — verified OFF (fully decrypted); nothing to suspend |
| LM3 | **Make the Ubuntu 24.04 LTS USB installer** (ISO downloaded+verified on the PC; flashed over SSH with header-written-last raw write) | Claude (SSH) | ~20–30 min | ✔ 2026-06-12 — flashed + readback-verified, stick in the PC |
| LM4 | **Shrink C: by 300 GB** via remote `Resize-Partition` (admin SSH verified) → leaves unallocated space for Ubuntu | Claude (SSH) | ~5 min | ✔ 2026-06-12 — C: now 677 GB Healthy, 300 GiB unallocated |
| LM5 | **Disable Secure Boot** in BIOS/UEFI (or commit to MOK enrollment in LM7) | Henry | ~5 min | ✔ 2026-06-12 — ASUS BIOS "OS Type: Other OS"; `mokutil` confirms disabled |
| LM6 | **Install Ubuntu alongside Windows** (boot the USB, "Install alongside", into the freed space; sets up the UEFI/GRUB boot menu) | Henry | ~30–45 min | ✔ 2026-06-12 — needed "safe graphics" boot (nouveau corruption on the 3060 Ti) |
| LM7 | **Make Ubuntu reachable (the handoff)** — bootstrap script (`linux-bootstrap` branch) set up SSH + key + Tailscale; Mac reaches the box via **LAN IP** (Mac's own Tailscale blocked by Netskope — see LM7 snag note) | Henry | ~10–15 min | ✔ 2026-06-12 — `gpu-linux` alias → 192.168.0.13 |
| LM8 | **NVIDIA + CUDA drivers** — installer's "third-party software" tick pre-installed driver 595.71.05 / CUDA 13.2 | Claude (SSH) | ~20–40 min | ✔ 2026-06-12 — `nvidia-smi` shows the 3060 Ti, no extra install needed |
| LM9 | **Clone repo + `uv sync`**, verify `torch.cuda.is_available()` (CUDA is the default Linux wheel — no override needed) | Claude (SSH) | ~15–30 min | ✔ 2026-06-12 — torch 2.10.0+cu128, CUDA True, 3060 Ti |
| LM10 | **Validation** — run a short training cycle on the GPU; confirm an SSH+tmux session survives detach (unattended-run check) | Claude (SSH) | ~15–30 min | ✔ 2026-06-12 — blokus_validation (2 gen × 20 eps) in 128 s, ran detached in tmux start to finish |

---

## LM1. Back up irreplaceable data *(Claude, over SSH)*
The AlphaBlokus repo + results are in git/GitHub, so the *project* is safe. Surveyed 2026-06-12: Desktop + Pictures are OneDrive-redirected (cloud-backed already); the local-only data is **Videos (~24 GB), PycharmProjects (~8 GB), Downloads (~6 GB), Projects (~0.3 GB)**. Claude pulls all four to `~/pc-backup-2026-06-12/` on the Mac via `scp` over the Tailscale link. This is a file-level safety copy, not a disk image — sufficient because the shrink only risks the filesystem, and Windows itself is reinstallable.

## LM2. Check + suspend BitLocker
Win11 Home often has "device encryption" on. Repartitioning or changing Secure Boot on an encrypted volume can force a recovery-key prompt at next boot. Check `manage-bde -status` (or Settings → Device encryption), **save the recovery key somewhere off the machine**, and **suspend** encryption before LM4/LM5. Re-enable after if you want.

## LM3. Make the Ubuntu 24.04 USB installer *(Claude, over SSH to the PC)*
**Status 2026-06-10: ISO downloaded + verified** — `~/Downloads/ubuntu-24.04.4-desktop-amd64.iso` (SHA256 `3a4c9877…d1e` matches the official SHA256SUMS).

**Route revised 2026-06-12:** Henry has no sudo on the Mac, so the Mac `dd` route is out. Instead the stick goes **straight into the PC** and Claude flashes it over the admin SSH session: download the ISO on the PC (re-verify SHA256), identify the stick in `Get-Disk` by **BusType = USB** + size, `diskpart clean` it, then raw-write the ISO to `\\.\PhysicalDriveN` with a PowerShell stream. Bonus: the stick is already in the PC for LM6 — no carrying it back.

## LM4. Shrink C: *(Claude, over SSH)*
The Mac's SSH session is admin, so Claude runs the shrink remotely — no Disk Management clicking needed:

```powershell
Resize-Partition -DriveLetter C -Size (677115871232)   # current 999.2 GB − 300 GiB
```

`Get-PartitionSupportedSize` (2026-06-12) confirms SizeMin ≈ 341.5 GB, so a 300 GiB shrink is well within Windows' unmovable-file limit. Leaves "unallocated" space the Ubuntu installer will use. NVMe makes this quick. *(Do LM1 first; LM2 verified moot.)*

Layout verified 2026-06-10 (disk 0): EFI System 100 MB → MSR 16 MB → **C: ~930 GB** → Recovery ~800 MB. Standard and clean — the shrink opens a gap between C: and the small Recovery partition, which is exactly where Ubuntu installs; GRUB reuses the existing EFI partition. Nothing awkward.

## LM5. Disable Secure Boot
No F2-mashing needed: when Henry is at the PC, Claude runs `shutdown /r /fw /t 5` over SSH — Windows reboots **straight into the UEFI firmware setup**. There, turn Secure Boot **off** (usually under Boot or Security). Alternative: leave it on and handle the **MOK enrollment** blue screen Ubuntu shows on first reboot after installing the NVIDIA driver — either works; disabling is simpler.

## LM6. Install Ubuntu alongside Windows
Boot the USB (one-time boot menu, usually F12). Choose **"Install Ubuntu alongside Windows Manager"** (it targets the unallocated space from LM4). Keep it **UEFI** (matches Windows). The installer sets up GRUB so you pick Windows/Ubuntu at power-on. **Tick "Install OpenSSH server"** in the installer (or `sudo apt install openssh-server` after) so the box is reachable in LM7.

## LM7. Make Ubuntu reachable — the handoff *(Henry, at the PC)*
The fresh Ubuntu has **no Tailscale and no SSH access set up** — the Windows node and the `gpu-cmd` route don't carry over (separate OS). So I can't reach it until you bootstrap it, at the PC:
1. **SSH:** confirm `openssh-server` is running (`systemctl status ssh`); if not, `sudo apt install -y openssh-server`.
2. **Tailscale:** `curl -fsSL https://tailscale.com/install.sh | sh` then `sudo tailscale up` → authenticate to the **same tailnet** in the browser. Ubuntu joins as a **new node** (its own hostname + `100.x` IP, separate from `desktop-5nut9e9`).
3. **Let the Mac in:** add my Mac's SSH public key to `~/.ssh/authorized_keys` (or set a user password). Tell me the new node's Tailscale name/IP + the Linux username.

Then from the Mac I confirm `tailscale status` shows the new node, add a `gpu-linux` SSH alias, and take over. **This step gates everything after it.**

**Snag found 2026-06-12:** the Mac's Tailscale client cannot re-register — `tailscale debug ts2021` shows the control-plane request intercepted and killed by the **on-device Netskope client** (destination rewritten to a non-Tailscale IP, `unexpected EOF` after the Noise handshake; reproduces on home Wi-Fi, work VPN, and phone hotspot, so it's the device agent, not the network). Unfixable without admin rights — raise with Transak IT. **Workaround:** while at home, SSH to the box over the **LAN IP** directly (`gpu-linux` alias points at the LAN address); the box's own Tailscale node is healthy, so phone/other devices can still reach it remotely.

## LM8. NVIDIA + CUDA drivers *(Claude, over SSH)*
`sudo ubuntu-drivers install` (or the recommended `nvidia-driver-5xx`), reboot, `nvidia-smi` should list the 3060 Ti. If Secure Boot was left on, complete the MOK enrollment at reboot. CUDA toolkit comes via the driver + the torch wheel (no separate CUDA install needed for PyTorch). **Nothing to pre-stage offline:** the NVIDIA driver installs from the Ubuntu archive over the PC's own network, and the 24.04.4 installer's kernel/firmware covers the motherboard NIC — pre-downloading `.deb` chains would be brittle and is unnecessary.

## LM9. Repo + uv + verify GPU *(Claude, over SSH)*
Clone the repo, `uv sync` — on **Linux the default torch wheel bundles CUDA**, so unlike native Windows there's no `--torch-backend` dance. Verify `torch.cuda.is_available()` is `True`. The `forkserver` start method (already in the code, auto-selected on Linux) activates here.

## LM10. Validation *(Claude, over SSH)*
Run a short training cycle (a couple of generations, modest games) to confirm GPU training + self-play work end-to-end. Start it under `tmux`, detach, reconnect — confirm it keeps running (the unattended-run capability WSL never gave us). Then the roadmap's P2–P4 tuning happens here, on the clean substrate.

---

## Notes
- **Windows stays fully intact** — this is dual-boot, not a wipe. You boot Windows whenever you want; you just can't run Windows and train simultaneously.
- **Rollback** if you abandon Ubuntu later: delete the Ubuntu partition + restore the Windows bootloader. Non-trivial but documented; the shrink (LM4) is the only thing touching the Windows partition.
- **Once on Linux, the whole remote-training story simplifies** — direct SSH to Linux, `tmux`, no `wsl.exe` hop, no "Catastrophic failure". The [REMOTE-TRAINING.md](../../guides/REMOTE-TRAINING.md) runbook will need a Linux section.
