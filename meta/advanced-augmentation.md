# Advanced Augmentation Plan

## Goal

Improve long-run robustness and recovery after corruption events (noise/dropout/silence),
while keeping label alignment correct and preserving the existing offline tempo/pitch flow.

---

## Architecture Split

### 1) Offline augmentation in `chopper.py` (label-affecting)

Keep these exactly in chopper:

- Tempo modes (`none`, `uniform`, `sudden_jump`, `gradual`)
- Pitch shift

Reason:

- They are expensive and already implemented offline.
- They can change label timing and therefore must stay where `.bars` remapping is done.

### 2) Online augmentation in dataset loading (label-preserving)

Apply these in `DanceDataset.__getitem__` before framing:

- Silence masking (time mask / dropout-like)
- Noise injection (multiple types)
- Room simulation and similar acoustics

Reason:

- No `.bars` remapping needed if timing is unchanged.
- Infinite variation across epochs without increasing stored chunk count.

Chopper config and train-time config remain independent.

---

## Library Choice

Use `audiomentations` in training data loading.

Planned transforms:

- `TimeMask` (silence insertion)
- `AddBackgroundNoise` (if corpus provided)
- `AddColorNoise`
- `AddGaussianNoise`
- `AddGaussianSNR`
- `AddShortNoises` (if corpus provided)
- `RoomSimulator` (optional, lower probability due to cost)

Pitch shift is not used online (already in chopper).

---

## Probability Model

Each transform rolls independently with its own probability.

- Probabilities are unrelated.
- Overlap is allowed (multiple transforms may apply to one sample).
- This is true both within online transforms and relative to offline chopper transforms.

---

## Hard Constraints

- Maximum single corruption duration (silence/noise interval): **8 seconds**
- Actual duration sampled randomly in `[0, 8s]`
- Validation stays clean (no online augmentation on val loader)
- Reproducibility is not required for now

---

## Train-Time Config Surface (`train.py`)

Add dedicated online augmentation args (examples):

- `--augment-online` (bool, default on for train split)
- `--augment-ramp-epochs` (int, default e.g. 5)
- `--noise-corpus-path` (str, recursive scan root, optional)

Per-transform probabilities:

- `--p-time-mask`
- `--p-add-gaussian-snr`
- `--p-add-gaussian-noise`
- `--p-add-color-noise`
- `--p-add-background-noise`
- `--p-add-short-noises`
- `--p-room-simulator`

Key bounds:

- `--max-mask-seconds` (default `8.0`)
- `--max-noise-seconds` (default `8.0`)

Implementation note:

- Keep defaults conservative so baseline training behavior remains stable.

---

## Augmentation Ramp (Curriculum)

Use less augmentation early for faster representation learning.

Simple schedule:

- `strength_scale(epoch) = min(1.0, (epoch + 1) / augment_ramp_epochs)`
- Effective probability = `base_probability * strength_scale`
- Optionally scale transform intensity ranges similarly

If `augment_ramp_epochs <= 0`, apply full strength from epoch 1.

---

## Dataset Integration Details

Apply online augmentation in `DanceDataset` flow:

1. load audio (`float32`)
2. optional normalize (existing behavior)
3. apply online augmentation on waveform (numpy array)
4. convert to tensor
5. frame reshape + label compute as today

Important:

- Ensure output length is unchanged after augmentation.
- If a transform can alter length, crop/pad back to expected sample count.
- Keep channel mono.

---

## Noise Corpus Handling

`--noise-corpus-path`:

- recursively scan for audio files
- pass file list to `audiomentations` transforms that need assets
- if path missing/empty, automatically disable corpus-dependent transforms
  (`AddBackgroundNoise`, `AddShortNoises`) with a clear warning

---

## Performance Notes

- Online augmentation runs in DataLoader workers (CPU).
- `RoomSimulator` can be expensive: keep low default `p`.
- Background/short-noise mixing is lightweight relative to convolution/reverb.

---

## Validation Policy

- Train split: online augmentation enabled
- Val split: clean only (no online augmentation)

This keeps reported validation loss comparable and stable.

---

## Rollout Plan

1. Add train args and augmentation config object.
2. Extend `DanceDataset` to accept an optional augmenter callable/pipeline.
3. Build online augmenter from args (with independent per-transform probabilities).
4. Wire train/val datasets separately (`train=augmented`, `val=clean`).
5. Add startup logging: active transforms, probs, ramp settings, corpus stats.
6. Run smoke training and measure loader throughput.

---

## Non-Goals (for now)

- Unifying chopper and online configs into one shared schema
- Deterministic/replayable augmentation
- Online pitch/tempo transforms
- Augmentation metadata per-sample persisted to disk

