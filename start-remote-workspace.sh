#!/usr/bin/env bash
# Start the standard dev workspace in two tmux sessions.
# Safe to re-run — skips sessions that already exist.

set -e

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=0.9
ACTIVATE="source $WORKDIR/venv/bin/activate"

# ── Session: left ────────────────────────────────────────────────────────────
if tmux has-session -t left 2>/dev/null; then
    echo "Session 'left' already exists, skipping."
else
    tmux new-session -d -s left -c "$WORKDIR"

    # Window 0: claude
    tmux send-keys -t left:0 "$ACTIVATE" Enter
    tmux send-keys -t left:0 "claude --continue" Enter

    # Window 1: mlflow
    tmux new-window -t left -c "$WORKDIR"
    tmux send-keys -t left:1 "$ACTIVATE" Enter
    tmux send-keys -t left:1 "mlflow server" Enter
fi

# ── Session: right ───────────────────────────────────────────────────────────
if tmux has-session -t right 2>/dev/null; then
    echo "Session 'right' already exists, skipping."
else
    tmux new-session -d -s right -c "$WORKDIR"

    # Window 0: annotate
    tmux send-keys -t right:0 "$ACTIVATE" Enter
    tmux send-keys -t right:0 "python annotate.py --music-path /mnt/datengrab/archive/music/ --out-path chunks/manual/v$VERSION --chunks-per-song 9" Enter

    # Window 1: inspector
    tmux new-window -t right -c "$WORKDIR"
    tmux send-keys -t right:1 "$ACTIVATE" Enter
    tmux send-keys -t right:1 "python inspector.py --chunks-path chunks/manual/v$VERSION --music-path /mnt/datengrab/archive/music/ --checkpoints-path checkpoints --port 8051" Enter

    # Window 2: ui
    tmux new-window -t right -c "$WORKDIR/ui"
    tmux send-keys -t right:2 "npm run dev" Enter
fi

# ── Focus session: left ──────────────────────────────────────────────────────
if [ -n "$TMUX" ]; then
    tmux switch-client -t left
else
    tmux attach-session -t left
fi
