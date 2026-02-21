# Training Guide (Phase Model)

This project now trains for **bar phase estimation** (single scalar in `[0, 1)`) with an **anticipation input**.

## Scope

- Target: `phase(t + a)`
- `a` (anticipation) is sampled uniformly in `[anticipation_min, anticipation_max]` during training
- Loss: circular (wrap-safe)
- Time signatures are read from MIDI and used to generate bar-phase labels
- Songs without usable time-signature data are skipped
- Default audio config for phase mode:
  - samplerate: `16000`
  - chunk duration: `32s`
  - frame size: `320` (50 FPS)

## Environment (GPU PC)

Use a fresh venv.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install non-PyTorch deps:

```bash
pip install mido soundfile numpy matplotlib onnx
```

Install PyTorch + torchaudio for your CUDA version from the official index:

```bash
# Example for CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

System tools needed:

```bash
timidity --version
ffmpeg -version
```

## 0) Tiny Smoke Run (recommended first)

Use one MIDI file and one soundfont:

Fast path (script):

```bash
./scripts/smoke_test.sh
```

Equivalent manual commands:

```bash
python render_midi.py \
  --single-file "data/midi/lakh-midi/Goo_Goo_Dolls/Name.mid" \
  --single-soundfont "soundfonts/FluidR3_GM.sf2" \
  --out-path "data/rendered_midi/smoke/single/song" \
  --midi-jitter-max-seconds 0.05 \
  --max-processes 1 \
  --overwrite
```

Check outputs exist:
- `data/rendered_midi/smoke/single/song/events.csv`
- `data/rendered_midi/smoke/single/song/bars.csv`
- one or more `.ogg` files

Create chunks + phase labels:

```bash
python chop.py \
  --in-path "data/rendered_midi/smoke" \
  --out-path "data/chunks/smoke_phase" \
  --max-processes 1
```

Train quick sanity model:

```bash
python train.py phase_tcn_mel \
  --chunks-path "data/chunks/smoke_phase" \
  --checkpoints-path "checkpoints" \
  --tag "smoke-phase" \
  --num-epochs 1 \
  --batch-size 2 \
  --anticipation-min 0.0 \
  --anticipation-max 1.0 \
  --warmup-seconds 8.0
```

Run inference on an `.ogg` chunk:

```bash
python apply.py \
  checkpoints/smoke-phase/1.pt \
  data/chunks/smoke_phase/<some_chunk>.ogg \
  --anticipation 0.25
```

Output file:
- `<chunk>.phase_prediction`

## 1) Full Data Generation

Render full MIDI corpus:

```bash
python render_midi.py \
  --midi-path "data/midi/lakh-midi" \
  --soundfont-path "soundfonts" \
  --out-path "data/rendered_midi/lakh_phase" \
  --midi-jitter-max-seconds 0.05 \
  --max-processes 8
```

Create chunks:

```bash
python chop.py \
  --in-path "data/rendered_midi/lakh_phase" \
  --out-path "data/chunks/lakh_phase" \
  --max-processes 8
```

## 2) Full Training (GPU)

Baseline:

```bash
python train.py phase_tcn_mel \
  --chunks-path "data/chunks/lakh_phase" \
  --checkpoints-path "checkpoints" \
  --tag "phase-tcn-mel-a0-500ms" \
  --num-epochs 30 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --learning-rate-decay 0.95 \
  --anticipation-min 0.0 \
  --anticipation-max 1.0 \
  --warmup-seconds 8.0 \
  --onnx
```

Continue training:

```bash
python train.py phase_tcn_mel \
  --chunks-path "data/chunks/lakh_phase" \
  --checkpoints-path "checkpoints" \
  --tag "phase-tcn-mel-a0-500ms" \
  --continue-from 30 \
  --num-epochs 20 \
  --anticipation-min 0.0 \
  --anticipation-max 1.0 \
  --warmup-seconds 8.0
```

## 3) Quick Eval/Inspection

Visual inspection:

```bash
python try_model.py checkpoints \
  phase-tcn-mel-a0-500ms/30.pt \
  --dataset-path "data/chunks/lakh_phase" \
  --anticipation 0.25 \
  --device-type cuda
```

Batch inference on one file:

```bash
python apply.py \
  checkpoints/phase-tcn-mel-a0-500ms/30.pt \
  <audio.ogg> \
  --anticipation 0.25 \
  --device-type cuda
```

## Notes

- Current label files are `.phase` (one scalar per frame).
- Existing kick/snare checkpoints are not directly comparable to new phase checkpoints.
- Keep an eye on skipped-song ratio during rendering; if too many files are filtered due to time-signature constraints, we can relax the filter logic.
