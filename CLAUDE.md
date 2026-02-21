# Dance — Project Overview for Claude

## Goal

Train a neural network to estimate musical **bar phase** in real time from audio.
- Output: `phase ∈ [0, 1)` where 0.0 = bar start
- The model predicts phase at a future time `t + a`, where `a` is an explicit **anticipation** input in `[0.0, 0.5]` seconds
- Supports varying time signatures (4/4, 3/4, 7/8, mid-song changes)
- Designed to run in the browser via ONNX export

See `meta/phase_pivot_spec.md` for the full agreed spec.
See `meta/NEW_IDEAS.md` for the anticipation training strategy.

---

## Data Pipeline

### Step 1 — Render: MIDI → OGG  (`render_midi.py`)
```
fluidsynth → .wav → ffmpeg(loudnorm) → .ogg
```
- Input: `midi_files/lmd_matched/` (Lakh MIDI Dataset matched subset)
- Soundfonts: `soundfonts/*.sf2`
- Output: `data/rendered/<version>/` — one `.ogg` + `bars.csv` per (song, soundfont)
- `bars.csv` contains bar-start times in seconds, computed from MIDI tempo/time-signature events
- Skips songs with no usable time signature, or songs already rendered

### Step 2 — Chop: OGG → chunks  (`chop.py`)
```
ffmpeg(asetrate pitch-shift + alimiter) → chunk.ogg  +  chunk.phase
```
- Input: `data/rendered/<version>/`
- Output: `data/chunks/<version>/` — pairs of `<name>.ogg` + `<name>.phase`
- Each chunk is `chunk_duration` seconds (32s, from `config.py`)
- `.phase` file: one float per audio frame (50 FPS), `phase_label_extra_seconds` (1s) beyond chunk end for anticipation headroom
- Random pitch augmentation: `pitch ∈ [0.71, 1.29]`

### Step 1+2 combined — `render_and_chop.py`  (**preferred for new data**)
```
fluidsynth → tmp.wav → ffmpeg(pitch+trim+limit) → chunk.ogg  +  chunk.phase
```
- Eliminates the intermediate `.ogg` and one full encode/decode cycle
- Same chunk format and phase label logic as the two-step pipeline
- WAV is written to a temp file and deleted immediately after chopping

Current data: `data/rendered/v1/` (279 songs), `data/chunks/v1/` (~770 chunks)
Older chunks (pre-pivot): `chunks/v1/` (350 chunks, kick/snare era — may be stale)

---

## Training  (`train.py`)

```bash
python train.py <model_name> \
  --chunks-path data/chunks/v1 \
  -t <tag> -e <epochs> -b <batch_size>
```

- Dataset: `dancer_data.py` — loads `.ogg` + `.phase` pairs
- Loss: circular MSE on sin/cos encoding of phase (`CustomLoss` in `train.py`)
  - Warmup: first N seconds of each sequence masked from loss
- Anticipation: sampled once per batch from `U(anticipation_min, anticipation_max)`
  - Default range: `[0.0, 0.5]` seconds
- Target phase computed by `create_future_phase_labels()` — interpolates in the `.phase` file
- Checkpoints: `checkpoints/<tag>/<epoch>.pt`
- Loss CSV: `checkpoints/<tag>/loss.csv`

### Overfit sanity check
```bash
python train.py phase_tcn_mel --chunks-path data/chunks/v1 \
  -t overfit_test -e 200 -b 4 --dataset-size 4
```
Training loss should fall clearly within ~50 epochs. If not, there is a bug.

---

## Models  (`models/`)

Active model: **`phase_tcn_mel`** (`models/phase_tcn_mel.py`)
- TCN (Temporal Convolutional Network) operating on mel spectrograms
- Takes anticipation as an explicit input
- Output: `[sin, cos]` per frame (2D unit-vector encoding of phase angle)

Selector: `models/selector.py` — `getModelClass(name)`, `loadModel()`, `saveModel()`

Older models (kick/snare era, not used for phase training):
`cnn_and_rnn.py`, `v2*.py`, `rnn_only.py`, etc.

---

## Config  (`config.py`)

| Variable | Value | Meaning |
|---|---|---|
| `samplerate` | 16000 | Hz |
| `frame_size` | 320 | samples (50 FPS) |
| `chunk_duration` | 32 | seconds of audio per chunk |
| `channels` | 1 | mono |

`render_and_chop.py` also defines `phase_label_extra_seconds = 1` (extra phase label coverage beyond chunk audio for anticipation).

---

## Inference / Viewer

- `apply.py` — run a model checkpoint on audio, produce phase predictions
- `try_model.py` — quick inference test
- `view_chunk.py`, `view_audio_and_presence.py` — visualise chunks and labels

---

## Key Design Decisions

- Phase encoded as `[sin(2π·phase), cos(2π·phase)]` — circular, wrap-safe
- Loss normalises predictions to unit circle before MSE — direction-only loss
- Anticipation is an explicit model input, not baked into labels at data time
- Phase labels written `chunk_duration + phase_label_extra_seconds` long so every frame has a valid target regardless of anticipation value
- Pitch augmentation via `asetrate` trick (changes playback speed = pitch shift without quality loss)
