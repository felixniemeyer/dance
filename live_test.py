#!/usr/bin/env python3
"""
live_test.py — Real-time bar phase estimation from desktop audio.

Usage:
    python live_test.py checkpoints/<tag>/<epoch>.pt
    python live_test.py checkpoints/<tag>/<epoch>.pt --device 12
    python live_test.py --list-devices
"""

import argparse
import queue
import threading
from math import gcd

import numpy as np
import scipy.signal
import sounddevice as sd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import config
from models.selector import loadModel

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', nargs='?', help='Path to .pt checkpoint')
parser.add_argument('--device', default=None,
                    help='Audio input device name or index (see --list-devices)')
parser.add_argument('--list-devices', action='store_true',
                    help='Print available input devices and exit')
parser.add_argument('--window', type=float, default=8.0,
                    help='Scrolling display window in seconds (default: 8)')
args = parser.parse_args()

if args.list_devices:
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0:
            tag = '  ← monitor' if 'monitor' in d['name'].lower() else ''
            print(f"  [{i:2d}]  {d['name']}  ({int(d['default_samplerate'])} Hz){tag}")
    raise SystemExit

if not args.checkpoint:
    parser.error('checkpoint is required (or use --list-devices)')

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR     = config.samplerate        # 24 000 Hz
FRAME_SIZE    = config.frame_size        # 400 samples
FPS           = TARGET_SR / FRAME_SIZE   # 60.0
WINDOW_FRAMES = int(args.window * FPS)   # 480 at 8 s
PROC_FRAMES   = 6                        # frames per GRU batch (~100 ms)
PROC_SAMPLES  = PROC_FRAMES * FRAME_SIZE # 2400 resampled samples per batch

# ── Model ─────────────────────────────────────────────────────────────────────

print(f'Loading {args.checkpoint} …')
model, _ = loadModel(args.checkpoint)
model.eval()
n_mels    = model.hparams['n_mels']
ckpt_name = '/'.join(args.checkpoint.replace('\\', '/').split('/')[-2:])

# ── Device selection ──────────────────────────────────────────────────────────

device = args.device
if device is None:
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and 'monitor' in d['name'].lower():
            device = i
            break
    if device is not None:
        print(f"Auto-detected monitor: {sd.query_devices(device)['name']}")
    else:
        print('No monitor device found — using default input. Try --list-devices.')

dev_info  = sd.query_devices(device, 'input')
native_sr = int(dev_info['default_samplerate'])
_gcd      = gcd(TARGET_SR, native_sr)
resamp_up, resamp_down = TARGET_SR // _gcd, native_sr // _gcd
print(f'Device: {dev_info["name"]}  |  {native_sr} Hz → {TARGET_SR} Hz')

# ── Ring buffers (proc thread writes, display reads) ──────────────────────────

mel_ring   = np.full((n_mels, WINDOW_FRAMES), -10.0, dtype=np.float32)
phase_ring = np.full(WINDOW_FRAMES, np.nan,          dtype=np.float32)
_lock      = threading.Lock()

# ── Processing thread ─────────────────────────────────────────────────────────

_queue     = queue.Queue()
_tail      = np.zeros(0, dtype=np.float32)  # resampled samples not yet batched
_gru_state = None


def _proc_loop():
    global _tail, _gru_state

    while True:
        try:
            raw = _queue.get(timeout=1.0)
        except queue.Empty:
            continue

        mono = (raw.mean(axis=1) if raw.ndim == 2 else raw.ravel()).astype(np.float32)

        if resamp_up == resamp_down:
            resampled = mono
        else:
            resampled = scipy.signal.resample_poly(mono, resamp_up, resamp_down).astype(np.float32)

        _tail = np.concatenate([_tail, resampled])

        while len(_tail) >= PROC_SAMPLES:
            chunk   = _tail[:PROC_SAMPLES]
            _tail   = _tail[PROC_SAMPLES:]
            batch_t = torch.from_numpy(chunk).reshape(1, PROC_FRAMES, FRAME_SIZE)

            with torch.no_grad():
                mel_t           = model.frontend(batch_t)        # [1, F, n_mels]
                out, _gru_state = model(batch_t, _gru_state)     # [1, F, 3]
                _gru_state      = _gru_state.detach()

            mel_np = mel_t[0].numpy()                            # [F, n_mels]
            sc     = out[0].numpy()                              # [F, 3]
            phases = np.arctan2(sc[:, 0], sc[:, 1]) / (2 * np.pi) % 1.0

            with _lock:
                mel_ring[:]   = np.roll(mel_ring,   -PROC_FRAMES, axis=1)
                phase_ring[:] = np.roll(phase_ring, -PROC_FRAMES)
                mel_ring[:, -PROC_FRAMES:] = mel_np.T
                phase_ring[-PROC_FRAMES:]  = phases


threading.Thread(target=_proc_loop, daemon=True).start()

# ── Audio callback ────────────────────────────────────────────────────────────

def _on_audio(indata, frames, time_info, status):
    if status:
        print(f'[sd] {status}')
    _queue.put(indata.copy())

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor('#08081a')
ax.set_facecolor('#08081a')
fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.12)

img = ax.imshow(
    mel_ring,
    aspect='auto', origin='lower', interpolation='nearest',
    extent=[0, args.window, 0, 1],
    cmap='inferno', vmin=-8, vmax=2,
)

xs = np.linspace(0, args.window, WINDOW_FRAMES)
phase_line, = ax.plot(xs, phase_ring, color='#00e5ff', lw=1.5, alpha=0.9)
phase_dot,  = ax.plot([], [], 'o', color='#00e5ff', ms=7, zorder=5)

ax.set_xlim(0, args.window)
ax.set_ylim(0, 1)
ax.set_xlabel('seconds', color='#555', fontsize=9)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '¼', '½', '¾', '1'], color='#666', fontsize=8)
ax.tick_params(colors='#444', length=3)
for sp in ax.spines.values():
    sp.set_edgecolor('#1a1a2e')
for y in [0.25, 0.5, 0.75]:
    ax.axhline(y, color='white', alpha=0.06, lw=0.7)

title_txt = ax.set_title(
    f'{ckpt_name}   phase: —', color='#888', fontsize=9, loc='left')


def _update(_):
    with _lock:
        mel_snap   = mel_ring.copy()
        phase_snap = phase_ring.copy()

    img.set_data(mel_snap)
    phase_line.set_ydata(phase_snap)

    cur = phase_snap[-1]
    if not np.isnan(cur):
        phase_dot.set_data([xs[-1]], [cur])
        title_txt.set_text(f'{ckpt_name}   phase: {cur:.3f}')
    else:
        phase_dot.set_data([], [])

    return img, phase_line, phase_dot, title_txt


ani = animation.FuncAnimation(fig, _update, interval=80, blit=True)  # ~12 fps

# ── Start ─────────────────────────────────────────────────────────────────────

print(f'mel bins: {n_mels}  |  window: {args.window} s  |  close window to stop')

with sd.InputStream(device=device, channels=1, samplerate=native_sr,
                    blocksize=2048, dtype='float32', callback=_on_audio):
    plt.show()
