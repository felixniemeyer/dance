# Real-World Audio Training Data

## Motivation

MIDI-rendered data has perfect labels but limited sonic diversity and mostly constant tempo.
Real-world recordings add:
- Natural timbres, room acoustics, mixing styles
- Humanised timing, micro-tempo variation
- Genre diversity not well covered by Lakh MIDI

## Pipeline Overview

```
audio file
    │
    ▼
offline beat tracker (madmom / librosa)
    │  beat timestamps
    ▼
automatic downbeat estimation
    │  proposed downbeat offset within beat grid
    ▼
manual confirmation TUI   ←── 5% human effort
    │  confirmed downbeat offset  (or reject whole song)
    ▼
metadata file  (audio path + bar_starts in seconds)
    │
    ▼
shared chunker (same as MIDI pipeline from this point)
    │  pitch / tempo / noise augmentation
    ▼
.ogg chunks + .phase labels
```

## Manual Confirmation TUI

A terminal tool (`label_audio.py`) that makes confirmation fast.

### What the user does

1. Tool runs beat tracker automatically, proposes a downbeat offset
2. User hears a looping playback of a few bars with the proposed bar grid
3. User shifts the downbeat offset with `h`/`l` until bar boundaries sound right
4. User accepts the entire song (`Enter`) or rejects it (`d`)

That's it — one decision per song.

### Keybindings

| Key | Action |
|-----|--------|
| `Space` | Play / pause loop |
| `l` | Shift downbeat offset +1 beat |
| `h` | Shift downbeat offset −1 beat |
| `Enter` | Accept song with current downbeat offset |
| `d` | Reject entire song (skip, don't save) |
| `n` | Next song without deciding (revisit later) |
| `q` | Save and quit |

### Display

```
Song: Radiohead - Karma Police.mp3
Beat tracker: 130 BPM, 4/4  (confidence 0.84)

  Downbeat offset: beat 2 of 4   [h ◄  ► l]

  |  bar  |  bar  |  bar  |  bar  |
  ↑       ↑       ↑       ↑
  playing...

[Space] play  [h/l] shift downbeat  [Enter] accept  [d] reject
```

## Output Format

One `.meta.json` per accepted song:

```json
{
  "audio": "songs/karma_police.mp3",
  "bar_starts": [0.412, 2.254, 4.091, 5.938, ...],
  "time_signature": "4/4",
  "bpm": 130.2
}
```

Bar starts are in seconds (continuous resolution — no frame-rate dependency).
Computed from beat timestamps + confirmed downbeat offset.

## Throughput Estimate

- ~2 songs/minute (mostly just listen + maybe one h/l press + Enter)
- 1 hour → ~120 songs → ~360 raw chunks (32s chunks, ~3 per song)
- With augmentation (pitch × 3, tempo × 3, noise × 2) → **~2000+ effective chunks/hour of labelling**

## Notes

- Store original audio untouched; all augmentation happens at chunk time
- Songs where beat tracker confidence is high can be pre-sorted to the front (easy wins)
- Madmom's `DBNDownBeatTrackingProcessor` is the best starting point for downbeat estimation
- Bar starts stored in seconds so the format is independent of frame rate / chunk size
