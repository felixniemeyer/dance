# Phase Pivot Spec (Feb 16, 2026)

This document captures the agreed project pivot from kick/snare detection to bar-phase estimation.

## Goal

Estimate musical bar phase as a single scalar:
- `phase in [0, 1)` (bar phase only)
- `0.0` = start of bar
- supports varying and changing time signatures (`4/4`, `3/4`, `7/8`, mid-song changes)

## Model Objective

For each frame at time `t`, predict phase at future time:
- `target = phase(t + a)`
- `a` is an explicit model input: anticipation in seconds
- anticipation range: `a in [0.0, 0.5]`

## Anticipation Training Strategy

- Sample anticipation from a uniform distribution.
- Current preference: sample once per chunk/sequence (not per frame), but keep design open to revisit.
- Long-term direction: reduce dependence on pre-generated chunks and move toward on-the-fly augmented sampling.

## Loss

- Use circular loss (phase wrap-safe), not plain MSE.
- Reason: `phase=0.99` and `phase=0.01` must be treated as close.

## Labels and Meter Handling

- Use bar phase only; no separate beat-class output.
- Follow MIDI time-signature events strictly, including mid-song changes.
- Add filtering to ignore songs without proper/usable time-signature information.
- Handle pickup bars (auftakt): strip pickup bars where feasible so phase origin aligns to first full bar.

## Data Augmentation: MIDI Jitter

Apply note-time jitter to improve robustness on real-world timing variation:
- jitter magnitude sampled per note: `j ~ U(0, 0.05s)`
- direction multiplier: `r ~ U(-1, 1)`
- applied offset: `delta_t = j * r`
- jitter note timings only (tempo/time-signature map stays unchanged)

## Agreed Decisions Summary

- Output: single bar phase scalar
- Loss: circular
- Anticipation: explicit input, predicts future phase `t + a`
- Time signatures: follow MIDI events, filter bad songs
- Pickup bars: strip if possible
- Jitter: per-note random in `[-0.05s, 0.05s]` via `j * r`

