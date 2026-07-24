# `local/` — machine-local, gitignored

Everything in this directory is gitignored **except** `README.md` and `secrets.env.example`.
It holds machine-specific and secret files that must never enter version control:

- **`secrets.env`** — the single source of truth for all API credentials (R2/AWS, W&B, RunPod).
  Copy it from `secrets.env.example` and fill in real values. See below.
- Ad-hoc launch scripts, notes, and other machine-local scratch.

## Secrets convention

All API keys live in **one** file, `local/secrets.env`, in `KEY=VALUE` form. To use them,
**source** the file into the environment for the command that needs it — never `cat`/print it:

```bash
set -a; source local/secrets.env; set +a
# ...then run your command in the SAME shell invocation (shell state does not persist between calls).
```

- Non-secret config (R2 endpoint/bucket, W&B project) belongs in the **run config JSON**, not here.
- On the Mac, S3/R2 calls also need the corporate CA bundle: `export AWS_CA_BUNDLE="$HOME/.corp-ca-bundle.pem"`.
- SSH identities are **not** kept here — they live in `~/.ssh/` (OS-managed), outside the repo.

Full write-up: `docs/guides/CLOUD-TRAINING.md` → "Secrets".
