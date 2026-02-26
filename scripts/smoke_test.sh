#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the new pipeline:
# 1) render.py  — one random MIDI → WAV + bar_starts.json
# 2) chunk.py   — WAV + JSON → chunk.ogg + chunk.bars
# 3) train.py   — one epoch with phase_gru_mel
# 4) apply.py   — inference on one chunk

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"

MIDI_PATH="${MIDI_PATH:-midi_files}"
SOUNDFONT_PATH="${SOUNDFONT_PATH:-soundfonts}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RENDERED_PATH="${RENDERED_PATH:-smoketests/rendered/${RUN_ID}}"
CHUNKS_PATH="${CHUNKS_PATH:-smoketests/chunks/${RUN_ID}}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-smoketests/checkpoints}"
TAG="${TAG:-smoke-${RUN_ID}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found/executable: $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$MIDI_PATH" ]]; then
  echo "midi path not found: $MIDI_PATH"
  exit 1
fi

if [[ ! -d "$SOUNDFONT_PATH" ]]; then
  echo "soundfont path not found: $SOUNDFONT_PATH"
  exit 1
fi

if [[ -z "$(find "$MIDI_PATH" -type f -name "*.mid" -print -quit)" ]]; then
  echo "no .mid files found under: $MIDI_PATH"
  exit 1
fi

if [[ -z "$(find "$SOUNDFONT_PATH" -type f -name "*.sf2" -print -quit)" ]]; then
  echo "no .sf2 files found under: $SOUNDFONT_PATH"
  exit 1
fi

echo "== Smoke Test =="
echo "python:          $PYTHON_BIN"
echo "device:          $DEVICE_TYPE"
echo "midi path:       $MIDI_PATH"
echo "soundfont path:  $SOUNDFONT_PATH"
echo "run id:          $RUN_ID"
echo "rendered:        $RENDERED_PATH"
echo "chunks:          $CHUNKS_PATH"
echo "checkpoints:     $CHECKPOINTS_PATH"
echo "tag:             $TAG"
echo

mkdir -p "$RENDERED_PATH" "$CHUNKS_PATH" "$CHECKPOINTS_PATH"

# ── Step 1: render ─────────────────────────────────────────────────────────────
echo "== Step 1/4: render (1 song) =="
"$PYTHON_BIN" render.py \
  --midi-path "$MIDI_PATH" \
  --soundfont-path "$SOUNDFONT_PATH" \
  --out-path "$RENDERED_PATH" \
  --target-count 1 \
  --max-song-length 300 \
  --render-timeout-seconds 60 \
  --overwrite

if ! find "$RENDERED_PATH" -maxdepth 1 -type f -name "*.json" | grep -q .; then
  echo "render output missing .json files in $RENDERED_PATH"
  exit 1
fi

# ── Step 2: chunk ──────────────────────────────────────────────────────────────
echo
echo "== Step 2/4: chunk =="
"$PYTHON_BIN" chunk.py \
  --in-path "$RENDERED_PATH" \
  --out-path "$CHUNKS_PATH" \
  --chunks-per-song 3 \
  --overwrite

if ! find "$CHUNKS_PATH" -maxdepth 1 -type f -name "*.bars" | grep -q .; then
  echo "chunk output missing .bars files in $CHUNKS_PATH"
  exit 1
fi

CHUNK_FILE="$(find "$CHUNKS_PATH" -maxdepth 1 -type f -name '*.ogg' | head -n 1 || true)"
if [[ -z "$CHUNK_FILE" ]]; then
  echo "no .ogg chunks generated in $CHUNKS_PATH"
  exit 1
fi

# ── Step 3: train ──────────────────────────────────────────────────────────────
echo
echo "== Step 3/4: train (1 epoch) =="
"$PYTHON_BIN" train.py phase_gru_mel \
  --chunks-path "$CHUNKS_PATH" \
  --checkpoints-path "$CHECKPOINTS_PATH" \
  --tag "$TAG" \
  --num-epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --warmup-seconds 4.0 \
  --num-workers 2

CHECKPOINT_FILE="$CHECKPOINTS_PATH/$TAG/1.pt"
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "checkpoint not found: $CHECKPOINT_FILE"
  exit 1
fi

# ── Step 4: apply ──────────────────────────────────────────────────────────────
echo
echo "== Step 4/4: apply =="
"$PYTHON_BIN" apply.py \
  "$CHECKPOINT_FILE" \
  "$CHUNK_FILE" \
  --device-type "$DEVICE_TYPE"

PRED_FILE="${CHUNK_FILE%.ogg}.phase_prediction"
if [[ ! -f "$PRED_FILE" ]]; then
  echo "phase prediction file missing: $PRED_FILE"
  exit 1
fi

echo
echo "Smoke test completed successfully."
echo "rendered:    $RENDERED_PATH"
echo "chunks:      $CHUNKS_PATH"
echo "checkpoint:  $CHECKPOINT_FILE"
echo "chunk:       $CHUNK_FILE"
echo "prediction:  $PRED_FILE"
