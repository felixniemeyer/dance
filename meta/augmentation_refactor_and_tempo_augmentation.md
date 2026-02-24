# Data Pipeline & Augmentation — Agreed Design

## Audio specs

| | value |
|---|---|
| Render samplerate | 44100 Hz (CD quality, MIDI → WAV) |
| Chunk samplerate | 24000 Hz |
| FPS | 60 |
| Frame size | 400 samples (24000 / 60) |
| Chunk duration | 32 s → 1920 frames |
| FFT window default | fft_frames=2 → 800 samples → 33ms → 30 Hz/bin |

---

## Pipeline

```
render.py
  MIDI + soundfont  →  fluidsynth  →  44.1kHz WAV  +  bar_starts.json
  · one output per (MIDI, soundfont) pair — no augmentation variants
  · >40k combinations already provides enough diversity

chunk.py   (shared for MIDI and future real-world input)
  input : WAV at any samplerate  +  bar_starts.json (absolute seconds)
  · resample to 24kHz
  · extract random window
  · apply ONE random tempo augmentation (see below)
  · write  chunk.ogg  +  chunk.bars

label_audio.py   (future — real-world audio)
  audio file  →  beat tracker  →  manual downbeat confirmation
  →  bar_starts.json  →  chunk.py (same path from here)
```

---

## Label format: `.bars`

Bar start times in seconds **relative to chunk start**, one per line:

```
0.000000
1.954321
3.908642
5.862963
...
```

Phase and phase_rate are derived on the fly in DanceDataset — never pre-sampled:

```python
# phase at frame f:
t = f * frame_size / samplerate
i = bisect(bar_starts, t) - 1
phase(f) = (t - bar_starts[i]) / (bar_starts[i+1] - bar_starts[i])

# phase_rate at frame f (radians per frame):
bar_duration_frames = (bar_starts[i+1] - bar_starts[i]) * fps
phase_rate(f) = 2π / bar_duration_frames
```

No `.phase` files in the new pipeline. Frame-rate independent — recompute at any fps
from the same `.bars` file.

---

## Tempo augmentation modes (offline, in chunk.py)

Pitch-preserving time stretch via `pyrubberband` / `librosa.effects.time_stretch`.

### 1. None (identity)
No stretch. bar_starts unchanged.

### 2. Uniform stretch
Stretch entire chunk by factor `f ~ U(0.75, 1.33)`.

```
bar_starts_new[i] = bar_starts[i] * f
```

Audio: time-stretch whole chunk by f.

### 3. Sudden jump
At a random split point `t_split` (not necessarily on a bar boundary),
tempo changes by factor `f ~ U(0.5, 2.0)` for the remainder.

```
source material needed after split: (chunk_duration - t_split) * f
output: first_half + time_stretch(second_half, 1/f)
total output duration: chunk_duration  (exact)

bar_starts_new[i] = bar_starts[i]                            if bar_starts[i] <= t_split
                  = t_split + (bar_starts[i] - t_split) / f  if bar_starts[i] >  t_split
```

This is the most important mode — trains the model to recover quickly after
an abrupt tempo change (song transition, double-time, half-time).

### 4. Gradual drift
Tempo interpolates linearly from `f_start ~ U(0.8, 1.0)` to
`f_end ~ U(1.0, 1.25)` (or reversed) across N segments.
Each segment is a small uniform stretch; bar_starts updated per segment.

---

## Phase_rate supervision

The model outputs `[sin(phase), cos(phase), phase_rate]`.
With `.bars` providing ground-truth bar durations, phase_rate now gets a
direct loss term:

```python
gt_phase_rate = 2π / bar_duration_frames          # per frame
loss = loss_phase + α * MSE(pred_phase_rate, gt_phase_rate)
```

α to be tuned; start with 0.1 (phase_rate scale is ~0.01–0.1 rad/frame
vs phase loss scale ~1.0).

---

## DanceDataset changes

- Loads `.ogg` + `.bars` (replaces `.phase`)
- Computes `phase_labels` and `rate_labels` tensors in `__getitem__`
- Returns `(frames, phase_labels, rate_labels, path)`

---

## Open questions / TODO

- Pitch augmentation: keep asetrate in chunk.py (one random pitch per chunk)
  or drop entirely given soundfont diversity?
- Gradual drift: implement as N=8 segments or use rubberband's variable-rate API?
- Noise / reverb augmentation: future work, note alongside chunk as `.meta.json`
