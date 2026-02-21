#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the phase pipeline:
# 1) render_and_chop one random MIDI with one random soundfont
# 2) train one epoch
# 3) run inference on one chunk

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"

MIDI_PATH="${MIDI_PATH:-midi_files/lakh-midi/Goo_Goo_Dolls}"
SOUNDFONT_PATH="${SOUNDFONT_PATH:-soundfonts}"
SOUNDFONT_FILE="${SOUNDFONT_FILE:-soundfonts/FluidR3_GM.sf2}"

RUN_ID="${RUN_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"
CHUNKS_PATH="${CHUNKS_PATH:-chunks/${RUN_ID}}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-checkpoints}"
TAG="${TAG:-${RUN_ID}}"
SMOKE_DATASET_SIZE="${SMOKE_DATASET_SIZE:-}"

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

if [[ ! -f "$SOUNDFONT_FILE" ]]; then
  echo "soundfont file not found: $SOUNDFONT_FILE"
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
echo "python:         $PYTHON_BIN"
echo "device:         $DEVICE_TYPE"
echo "midi path:      $MIDI_PATH"
echo "soundfont path: $SOUNDFONT_PATH"
echo "soundfont file: $SOUNDFONT_FILE"
echo "run id:         $RUN_ID"
echo "chunks path:    $CHUNKS_PATH"
echo "tag:            $TAG"
echo

mkdir -p "$CHUNKS_PATH" "$CHECKPOINTS_PATH"

echo "== Step 1/3: render_and_chop (1 random song) =="
"$PYTHON_BIN" render_and_chop.py \
  --midi-path "$MIDI_PATH" \
  --soundfont-path "$SOUNDFONT_PATH" \
  --single-soundfont "$SOUNDFONT_FILE" \
  --out-path "$CHUNKS_PATH" \
  --target-count 1 \
  --max-song-length 300 \
  --render-timeout-seconds 60 \
  --max-processes 1 \
  --overwrite

if ! find "$CHUNKS_PATH" -maxdepth 1 -type f -name "*.phase" | grep -q .; then
  echo "render_and_chop output missing .phase labels in $CHUNKS_PATH"
  exit 1
fi

CHUNK_FILE="$(find "$CHUNKS_PATH" -maxdepth 1 -type f -name '*.ogg' | head -n 1 || true)"
if [[ -z "$CHUNK_FILE" ]]; then
  echo "no chunks generated in $CHUNKS_PATH"
  exit 1
fi

echo "== Step 2/3: train =="
TRAIN_DATASET_ARGS=()
if [[ -n "$SMOKE_DATASET_SIZE" ]]; then
  TRAIN_DATASET_ARGS+=(--dataset-size "$SMOKE_DATASET_SIZE")
fi

"$PYTHON_BIN" train.py phase_tcn_mel \
  --chunks-path "$CHUNKS_PATH" \
  --checkpoints-path "$CHECKPOINTS_PATH" \
  --tag "$TAG" \
  --num-epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --learning-rate-decay 0.95 \
  --anticipation-min 0.0 \
  --anticipation-max 1.0 \
  --warmup-seconds 8.0 \
  "${TRAIN_DATASET_ARGS[@]}"

CHECKPOINT_FILE="$CHECKPOINTS_PATH/$TAG/1.pt"
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "checkpoint not found: $CHECKPOINT_FILE"
  exit 1
fi

echo "== Step 3/3: apply =="
"$PYTHON_BIN" apply.py \
  "$CHECKPOINT_FILE" \
  "$CHUNK_FILE" \
  --anticipation 0.25 \
  --device-type "$DEVICE_TYPE"

PRED_FILE="${CHUNK_FILE%.ogg}.phase_prediction"
if [[ ! -f "$PRED_FILE" ]]; then
  echo "phase prediction file missing: $PRED_FILE"
  exit 1
fi

echo
echo "Smoke test completed successfully."
echo "checkpoint:  $CHECKPOINT_FILE"
echo "chunk:       $CHUNK_FILE"
echo "prediction:  $PRED_FILE"
