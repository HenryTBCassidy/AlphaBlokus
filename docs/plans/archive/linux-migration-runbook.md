# Linux migration — today's runbook (2026-06-12)

> **Status: COMPLETED 2026-06-12 — archived 2026-06-15.** This was the day-of
> execution sequence; the migration finished and was validated. Historical record.

Companion to [linux-migration.md](linux-migration.md) — this is the exact, ordered
sequence for migration day, with every command spelled out. Claude-driven steps are
marked **[Claude]**; your hands-on steps are marked **[Henry]**.

**Answering the "can the drivers go on the USB?" question:** no — and they don't
need to. The flashed stick is a read-only live image; you can't pre-install software
into it. The NVIDIA driver is installed *after* Ubuntu is on disk, over the PC's own
network (LM8, Claude does it over SSH). Two steps by necessity, but step two is
hands-off for you.

---

## Phase 0 — prep (done before you touch the PC)

| ✓ | Step | Who |
|---|------|-----|
| ✔ | ISO verified (`ubuntu-24.04.4-desktop-amd64.iso`, SHA256 matches official) | Claude |
| ✔ | BitLocker re-verified OFF; Secure Boot re-verified ON; shrink limits verified | Claude |
| | Flash the USB — stick goes in the **PC**; Claude flashes it over admin SSH (no Mac sudo available) | Claude |
| | Backup pull: PC's Videos/Downloads/PycharmProjects/Projects → `~/pc-backup-2026-06-12` on the Mac (~38 GB; Desktop+Pictures already on OneDrive) | Claude |
| | Shrink C: by 300 GiB remotely (`Resize-Partition`), gated on the backup finishing | Claude |

**The flash** (revised — no Mac sudo): Henry plugs the stick into the **PC**; Claude
downloads the ISO on the PC, verifies SHA256 against the known-good value, identifies
the stick in `Get-Disk` (BusType USB, ~115 GB), `diskpart clean`s it, and raw-writes
the ISO to `\\.\PhysicalDriveN` via PowerShell. The stick then stays in the PC,
ready for Phase 1.

---

## Phase 1 — at the PC [Henry, ~60–90 min]

Bring: the flashed USB stick, your phone (for the Tailscale browser auth).

### 1. Reboot into BIOS (no key-mashing needed)
Tell Claude you're at the PC; Claude runs `shutdown /r /fw /t 5` over SSH and the
machine reboots **straight into UEFI setup**. (Fallback if the SSH route is down:
mash **Del** during power-on.)

### 2. In the BIOS
- Find **Secure Boot** (usually under Boot, Security, or Windows OS Configuration)
  → set **Disabled**.
- Don't change anything else. **Save & Exit** — but plug the USB stick in first,
  and on the way out use the BIOS's **boot-override / boot menu** to boot the
  **UEFI: <SanDisk USB>** entry directly. (If you miss it: reboot and press **F11**
  — MSI — or **F12/F8** depending on board for the one-time boot menu.)

### 3. Ubuntu installer
1. Choose **"Try or Install Ubuntu"** at the GRUB menu.
2. Pick language → **Install Ubuntu** → Interactive installation.
3. **Default selection** of apps is fine. On the "Optimise your computer" screen,
   **tick "Install third-party software for graphics and Wi-Fi hardware"** — this
   can pull the NVIDIA driver during install (saves time in LM8; harmless if it
   doesn't).
4. At "How do you want to install Ubuntu?" choose **"Install Ubuntu alongside
   Windows Boot Manager"**. It will offer the unallocated 300 GB — confirm the
   slider/summary shows it installing into **free space**, *not* resizing or
   erasing anything. If it shows "Erase disk", STOP and back out — wrong option.
5. Username: `henry` (lowercase, keeps paths simple). Hostname: `gpu-linux`.
   Pick a password you can type at the machine; we'll be on keys for SSH.
6. Install (~15–25 min), then **Restart now**, pull the stick when prompted.

### 4. First boot + handoff (LM7) — paste-block
GRUB menu appears → **Ubuntu** (Windows Boot Manager is right there too — that's
your dual boot working). Log in, open a Terminal (Ctrl+Alt+T), and paste this whole
block:

```bash
sudo apt update && sudo apt install -y openssh-server curl
sudo systemctl enable --now ssh
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMV6huhIWMWscm7giZbzbSZiplucPLQBp8+Xe3N6gug5 henry-macbook' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

`tailscale up` prints a login URL — open it (phone or the Ubuntu browser itself)
and authenticate to the **same tailnet** as always. Then run `tailscale ip -4` and
tell Claude the IP. **That's the handoff — you're done with hands-on work.**

---

## Phase 2 — over SSH [Claude, ~60–90 min, Henry can walk away]

1. Add `gpu-linux` alias to the Mac's `~/.ssh/config`; verify key SSH works.
2. **LM8:** `sudo ubuntu-drivers install` → reboot → `nvidia-smi` shows the 3060 Ti.
3. **LM9:** clone AlphaBlokus, `uv sync`, verify `torch.cuda.is_available()`.
4. **LM10:** short GPU training cycle under `tmux`; detach/reattach to confirm
   unattended-run survival.
5. Update REMOTE-TRAINING.md with the Linux route; tick off the plan.

*(Claude will need Henry's Ubuntu password once for passwordless-sudo setup —
or Henry runs `echo "henry ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/henry`
in the Phase-1 paste session to pre-authorize it.)*

---

## If something goes wrong

| Symptom | What it means | Fix |
|---|---|---|
| Windows won't boot after the shrink | Filesystem issue from resize (very rare; backup exists) | Boot Windows recovery (auto after 2 failed boots) → Startup Repair; worst case `chkdsk /f` from recovery console |
| No "alongside Windows" option in installer | Installer can't see the free space or disk | Back out, check `Something else` shows the 300 GB free space; do NOT proceed with "Erase disk" |
| PC boots straight to Windows after install, no GRUB | Boot order put Windows Boot Manager first | BIOS → Boot order → put **ubuntu** first (GRUB chain-loads Windows) |
| Blue "Perform MOK management" screen on a reboot | Secure Boot got left on + a signed module needs enrolling | Enroll MOK → Continue → enter the password set during driver install; or just disable Secure Boot in BIOS |
| `nvidia-smi` says "No devices found" | Driver/kernel mismatch or Secure Boot blocking the module | Claude debugs over SSH (`ubuntu-drivers`, dkms status); Secure Boot off clears the common case |
| Ubuntu node not appearing in tailnet | Auth flow incomplete | Re-run `sudo tailscale up`, finish the browser login |
| Tailscale fine but SSH refused | sshd not running | At the PC: `sudo systemctl enable --now ssh` |
| Want Windows back exactly as before | Abandoning the experiment | Boot Windows → delete Ubuntu partitions in Disk Management → extend C: → `bcdedit` / `bootrec` to drop GRUB; backup at `~/pc-backup-2026-06-12` regardless |

**Rollback note:** Windows is untouched except for being 300 GB smaller; every step
before the installer writes partitions (Phase 1 step 4) is fully reversible by
just rebooting.
