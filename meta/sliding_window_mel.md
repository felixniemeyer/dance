# Idea: Sliding-window mel spectrogram at higher frame rate

## Current setup
- Frame size: 320 samples at 16 kHz → 50 Hz (20 ms per frame)
- Mel spectrogram computed once per frame (non-overlapping)

## Idea
Use a **sliding (overlapping) FFT window** to produce mel frames at a higher rate,
e.g. 80 Hz (12.5 ms hop), while keeping a larger analysis window (e.g. 512 or 1024 samples)
for frequency resolution.

This separates:
- **Hop size** (temporal resolution of the model output, currently locked to frame_size)
- **Window size** (FFT resolution, needs to be larger for good low-freq content)

At 80 Hz the model would output phase at a finer grid, which could improve
accuracy of beat-onset detection and anticipation alignment.

## Trade-offs
- More frames per chunk → larger tensors, more compute per batch
- Would require changing config.py (frame_size), data pipeline (chop.py, render_and_chop.py),
  dancer_data.py, and the model architecture
- Existing v1/v2 chunks would be incompatible

## Decision (2026-02-21)
Keep 50 Hz / 320 samples for now to avoid a full data+model rewrite.
Revisit if phase-boundary precision turns out to be a bottleneck.
