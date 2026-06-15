#!/usr/bin/env bash
#
# Fetch a finished training run's HTML reports from the GPU box down to the
# local repo mirror, so they land at
#
#     temp/runs/<group>/<run>/Reporting/{report.html,arena_replays.html}
#
# i.e. exactly where they'd sit if the run had executed locally. The run
# actually executes on the box, and the box cannot push back to the laptop
# (the devtunnel only goes laptop -> box), so retrieval is always laptop-pulled.
#
# arena_replays.html is huge (~750 MB of embedded games) but is almost all
# text, so it's gzip'd on the box and gunzip'd locally to keep the transfer
# quick rather than streaming the raw file over the devtunnel relay.
#
# Usage:
#   scripts/fetch_run_reports.sh [run_name] [group]
#
# Defaults: newest run under temp/runs/<group>/ on the box; group = blokus.
# Override the SSH host with REMOTE_HOST (default: gpu-anywhere).
set -euo pipefail

REMOTE="${REMOTE_HOST:-gpu-anywhere}"
REPO_REMOTE='AlphaBlokus'          # relative to the box's $HOME (ssh lands there)
GROUP="${2:-blokus}"
RUN="${1:-}"

remote_runs="$REPO_REMOTE/temp/runs/$GROUP"

if [[ -z "$RUN" ]]; then
  RUN="$(ssh "$REMOTE" "ls -1dt $remote_runs/*/ 2>/dev/null | head -1 | xargs -rn1 basename")"
  [[ -z "$RUN" ]] && { echo "No runs found under $remote_runs on $REMOTE" >&2; exit 1; }
  echo "Latest run: $RUN"
fi

remote_rep="$remote_runs/$RUN/Reporting"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_rep="$here/temp/runs/$GROUP/$RUN/Reporting"
mkdir -p "$local_rep"

# report.html — small, copy directly (scp -C compresses in transit).
if ssh "$REMOTE" "test -f $remote_rep/report.html"; then
  echo "-> report.html"
  scp -C "$REMOTE:$remote_rep/report.html" "$local_rep/report.html"
else
  echo "!! no report.html on box for $RUN" >&2
fi

# arena_replays.html — large; gzip on the box, ship the .gz, gunzip locally.
if ssh "$REMOTE" "test -f $remote_rep/arena_replays.html"; then
  echo "-> arena_replays.html (gzip transfer)"
  tmp="/tmp/${RUN}_arena_replays.html.gz"
  ssh "$REMOTE" "gzip -c $remote_rep/arena_replays.html > $tmp"
  scp "$REMOTE:$tmp" "$local_rep/arena_replays.html.gz"
  ssh "$REMOTE" "rm -f $tmp"
  gunzip -f "$local_rep/arena_replays.html.gz"
else
  echo "!! no arena_replays.html on box for $RUN" >&2
fi

echo "Done. Reports at: $local_rep"
ls -lh "$local_rep"
