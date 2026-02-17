#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the phase pipeline:
# 1) render one MIDI with one soundfont
# 2) chop into phase chunks
# 3) train one epoch
# 4) run inference on one chunk

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"

MIDI_FILE="${MIDI_FILE:-data/midi/lakh-midi/Goo_Goo_Dolls/Name.mid}"
SOUNDFONT_FILE="${SOUNDFONT_FILE:-soundfonts/FluidR3_GM.sf2}"

RUN_ID="${RUN_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"
RENDER_ROOT="${RENDER_ROOT:-data/rendered_midi/${RUN_ID}}"
CHUNKS_PATH="${CHUNKS_PATH:-data/chunks/${RUN_ID}}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-checkpoints}"
TAG="${TAG:-${RUN_ID}}"
SMOKE_DATASET_SIZE="${SMOKE_DATASET_SIZE:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found/executable: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$MIDI_FILE" ]]; then
  echo "midi file not found: $MIDI_FILE"
  exit 1
fi

if [[ ! -f "$SOUNDFONT_FILE" ]]; then
  echo "soundfont file not found: $SOUNDFONT_FILE"
  exit 1
fi

echo "== Smoke Test =="
echo "python:        $PYTHON_BIN"
echo "device:        $DEVICE_TYPE"
echo "midi:          $MIDI_FILE"
echo "soundfont:     $SOUNDFONT_FILE"
echo "run id:        $RUN_ID"
echo "render root:   $RENDER_ROOT"
echo "chunks path:   $CHUNKS_PATH"
echo "tag:           $TAG"
if [[ -n "$SMOKE_DATASET_SIZE" ]]; then
  echo "dataset size:  $SMOKE_DATASET_SIZE"
else
  echo "dataset size:  full smoke set"
fi
echo

mkdir -p "$RENDER_ROOT" "$CHUNKS_PATH" "$CHECKPOINTS_PATH"

echo "== Step 1/4: render_midi =="
"$PYTHON_BIN" render_midi.py \
  --single-file "$MIDI_FILE" \
  --single-soundfont "$SOUNDFONT_FILE" \
  --out-path "$RENDER_ROOT/single/song" \
  --midi-jitter-max-seconds 0.05 \
  --max-processes 1 \
  --overwrite

if [[ ! -f "$RENDER_ROOT/single/song/events.csv" || ! -f "$RENDER_ROOT/single/song/bars.csv" ]]; then
  echo "render output missing events.csv or bars.csv in $RENDER_ROOT/single/song"
  exit 1
fi

echo "== Step 2/4: chop =="
"$PYTHON_BIN" chop.py \
  --in-path "$RENDER_ROOT" \
  --out-path "$CHUNKS_PATH" \
  --max-processes 1

CHUNK_FILE="$(find "$CHUNKS_PATH" -maxdepth 1 -type f -name '*.ogg' | head -n 1 || true)"
if [[ -z "$CHUNK_FILE" ]]; then
  echo "no chunks generated in $CHUNKS_PATH"
  exit 1
fi

echo "== Step 3/4: train =="
TRAIN_DATASET_ARGS=()
if [[ -n "$SMOKE_DATASET_SIZE" ]]; then
  TRAIN_DATASET_ARGS+=(--dataset-size "$SMOKE_DATASET_SIZE")
fi

"$PYTHON_BIN" train.py CRS \
  --chunks-path "$CHUNKS_PATH" \
  --checkpoints-path "$CHECKPOINTS_PATH" \
  --tag "$TAG" \
  --num-epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --learning-rate-decay 0.95 \
  --anticipation-min 0.0 \
  --anticipation-max 0.5 \
  "${TRAIN_DATASET_ARGS[@]}"

CHECKPOINT_FILE="$CHECKPOINTS_PATH/$TAG/1.pt"
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "checkpoint not found: $CHECKPOINT_FILE"
  exit 1
fi

echo "== Step 4/4: apply =="
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
echo "checkpoint:     $CHECKPOINT_FILE"
echo "chunk:          $CHUNK_FILE"
echo "prediction:     $PRED_FILE"
