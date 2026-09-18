#!/usr/bin/env bash
# Copy one cell's results off the pod the moment it finishes. Called by _lib.sh after every cell.
#   bash sync.sh <cell-name>
# Two independent destinations; either alone is enough to survive a dead pod.
set -uo pipefail
W=/workspace
NAME=${1:?usage: sync.sh <cell-name>}
SRC=$W/results/$NAME

# 1. Google Drive via rclone (configure once: rclone config, remote named "gdrive")
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
  rclone copy "$SRC" "gdrive:research/taboo-rerun/$NAME" -q && echo "  synced $NAME -> gdrive"
  rclone copy "$W/logs/$NAME" "gdrive:research/taboo-rerun/$NAME/logs" -q
else
  echo "  (no rclone gdrive remote — skipping Drive copy)"
fi

# 2. Small artifacts into the git repo, so provenance travels with the code.
#    Metrics, manifests and a transcript sample only — never the full rollouts.
REPO=$W/mats-12-neel
if [ -d "$REPO/.git" ]; then
  DST=$REPO/project/results/$NAME
  mkdir -p "$DST"
  find "$SRC" -maxdepth 2 -name '*.json' -size -2M -exec cp {} "$DST/" \; 2>/dev/null
  cp "$SRC/manifest.json" "$DST/" 2>/dev/null
  cd "$REPO" && git add -A project/results && \
    git commit -q -m "results: $NAME ($(date -Is))" && echo "  committed $NAME" || true
else
  echo "  (no repo at $REPO — clone it if you want results committed)"
fi

echo "  keep rollouts: $W/rollouts/$NAME  (needed for the bootstrap and transcript sampling)"
