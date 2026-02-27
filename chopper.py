"""
Chop rendered WAV files into training chunks with tempo/pitch augmentation.

Usage:
    python chopper.py \
        --in-path data/rendered/v2 \
        --out-path data/chunks/v2 \
        --chunks-per-song 3

Augmentation modes (weighted random per chunk):
    none        25%  — raw slice
    uniform     25%  — time-stretch full chunk by f ~ U(0.75, 1.33)
    sudden_jump 35%  — no stretch for first t_split seconds, then stretch by f ~ U(0.5, 2.0)
    gradual     15%  — N segments with linearly interpolated stretch factors

Pitch shift ~ U(-pitch_range, pitch_range) semitones applied after tempo augmentation.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile

from config import chunk_duration, samplerate as CHUNK_SR

parser = argparse.ArgumentParser(description='Chop rendered WAV files into training chunks.')
parser.add_argument('--in-path', type=str, required=True,
    help='path to folder containing .wav + .json pairs')
parser.add_argument('--out-path', type=str, required=True,
    help='path to write chunk .ogg + .bars files')
parser.add_argument('--chunks-per-song', type=int, default=3)
parser.add_argument('--pitch-range', type=float, default=2.0,
    help='max pitch shift in semitones (applied after tempo augmentation)')
parser.add_argument('--overwrite', default=False, action='store_true')

args = None  # set by main() when run as script

TEMPO_MODES   = ['none', 'uniform', 'sudden_jump', 'gradual']
TEMPO_WEIGHTS = [0.25,   0.25,     0.35,          0.15]
N_GRADUAL_SEGS = 8


# ── audio helpers ──────────────────────────────────────────────────────────────

def _atempo_chain(rate: float) -> str:
    """
    Build an ffmpeg atempo filter string for any rate.
    atempo is limited to [0.5, 2.0] per filter node, so chain them for
    values outside that range (e.g. rate=4.0 → 'atempo=2.0,atempo=2.0').
    """
    filters = []
    r = rate
    while r > 2.0 + 1e-9:
        filters.append('atempo=2.0')
        r /= 2.0
    while r < 0.5 - 1e-9:
        filters.append('atempo=0.5')
        r *= 2.0
    if abs(r - 1.0) > 1e-4:
        filters.append(f'atempo={r:.8f}')
    return ','.join(filters) if filters else 'anull'


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    WSOLA time-stretch via ffmpeg atempo (SoundTouch engine) — same algorithm
    as Mixxx's time-scaling.  Pure time-domain: no phase vocoder, no spectral
    smearing, transients stay sharp.

    rate > 1 → faster (shorter output); rate < 1 → slower (longer output).
    Falls back to the input unchanged on ffmpeg error.
    """
    if abs(rate - 1.0) < 1e-4:
        return audio

    fd_in,  tmp_in  = tempfile.mkstemp(suffix='.wav')
    fd_out, tmp_out = tempfile.mkstemp(suffix='.wav')
    os.close(fd_in); os.close(fd_out)
    try:
        soundfile.write(tmp_in, audio.astype(np.float32), CHUNK_SR)
        cmd = [
            'ffmpeg', '-v', 'warning', '-y',
            '-i', tmp_in,
            '-af', _atempo_chain(rate),
            '-ar', str(CHUNK_SR),
            tmp_out,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return audio   # graceful fallback
        out, _ = soundfile.read(tmp_out, dtype='float32')
        return out.astype(np.float32)
    finally:
        for p in (tmp_in, tmp_out):
            if os.path.exists(p):
                os.remove(p)


def _ensure_len(audio, n_samples):
    """Trim or zero-pad audio to exactly n_samples."""
    if len(audio) > n_samples:
        return audio[:n_samples]
    if len(audio) < n_samples:
        return np.pad(audio, (0, n_samples - len(audio)))
    return audio


def _write_ogg(audio, out_path):
    """Encode float32 audio array to OGG Vorbis q6 via ffmpeg."""
    fd, tmpwav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        soundfile.write(tmpwav, audio, CHUNK_SR)
        cmd = [
            'ffmpeg', '-v', 'warning', '-y',
            '-i', tmpwav,
            '-ac', '1',
            '-c:a', 'libvorbis', '-qscale:a', '6',
            '-ar', str(CHUNK_SR),
            out_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    finally:
        if os.path.exists(tmpwav):
            os.remove(tmpwav)


# ── augmentation modes ─────────────────────────────────────────────────────────

def _aug_none(audio, bar_starts, start_s):
    """Slice chunk_duration seconds with no tempo change."""
    s = int(start_s * CHUNK_SR)
    e = s + int(chunk_duration * CHUNK_SR)
    if e > len(audio):
        return None, None
    chunk = audio[s:e]
    rel = [b - start_s for b in bar_starts if start_s <= b < start_s + chunk_duration]
    if len(rel) < 2:
        return None, None
    return chunk, rel


def _aug_uniform(audio, bar_starts, start_s, f):
    """Uniform stretch: consume chunk_duration*f source seconds, output chunk_duration."""
    src_dur = chunk_duration * f
    s = int(start_s * CHUNK_SR)
    e = s + int(src_dur * CHUNK_SR)
    if e > len(audio):
        return None, None
    chunk = _ensure_len(_time_stretch(audio[s:e], rate=f), int(chunk_duration * CHUNK_SR))
    rel = []
    for b in bar_starts:
        src_t = b - start_s
        if 0 <= src_t < src_dur:
            out_t = src_t / f
            if out_t < chunk_duration:
                rel.append(out_t)
    if len(rel) < 2:
        return None, None
    return chunk, rel


def _aug_sudden_jump(audio, bar_starts, start_s, t_split, f):
    """First t_split s at 1x; remaining (chunk_duration - t_split) s output from f-stretched source."""
    abs_split = start_s + t_split
    needed_src = (chunk_duration - t_split) * f

    first_s = int(start_s * CHUNK_SR)
    first_e = int(abs_split * CHUNK_SR)
    second_s = first_e
    second_e = second_s + int(needed_src * CHUNK_SR)

    if first_e > len(audio) or second_e > len(audio):
        return None, None

    first_audio = audio[first_s:first_e]
    second_src = audio[second_s:second_e]
    if len(second_src) == 0:
        return None, None

    target_second = int((chunk_duration - t_split) * CHUNK_SR)
    second_audio = _ensure_len(_time_stretch(second_src, rate=f), target_second)
    chunk = np.concatenate([first_audio, second_audio])

    rel = []
    for b in bar_starts:
        src_t = b - start_s
        if src_t < 0:
            continue
        if src_t <= t_split:
            out_t = src_t
        else:
            out_t = t_split + (b - abs_split) / f
        if 0 <= out_t < chunk_duration:
            rel.append(out_t)
    if len(rel) < 2:
        return None, None
    return chunk, rel


def _aug_gradual(audio, bar_starts, start_s, f_start, f_end):
    """N segments with linearly interpolated stretch factors f_start → f_end."""
    n = N_GRADUAL_SEGS
    seg_out_dur = chunk_duration / n
    seg_out_samples = int(seg_out_dur * CHUNK_SR)

    # Compute source time boundaries per segment
    seg_src_starts = []
    seg_src_ends = []
    src_cur = start_s
    fs = []
    for k in range(n):
        fk = f_start + (f_end - f_start) * k / max(n - 1, 1)
        fs.append(fk)
        seg_src_dur = seg_out_dur * fk
        seg_src_starts.append(src_cur)
        seg_src_ends.append(src_cur + seg_src_dur)
        src_cur += seg_src_dur

    if int(src_cur * CHUNK_SR) > len(audio):
        return None, None

    segments = []
    for k in range(n):
        s = int(seg_src_starts[k] * CHUNK_SR)
        e = int(seg_src_ends[k] * CHUNK_SR)
        src_seg = audio[s:e]
        if len(src_seg) == 0:
            return None, None
        segments.append(_ensure_len(_time_stretch(src_seg, rate=fs[k]), seg_out_samples))

    chunk = np.concatenate(segments)

    rel = []
    for b in bar_starts:
        for k in range(n):
            if seg_src_starts[k] <= b < seg_src_ends[k]:
                out_t = k * seg_out_dur + (b - seg_src_starts[k]) / fs[k]
                if 0 <= out_t < chunk_duration:
                    rel.append(out_t)
                break
    if len(rel) < 2:
        return None, None
    return chunk, rel


# ── chunk generation ───────────────────────────────────────────────────────────

def _make_chunk(audio, bar_starts, start_s):
    """Pick a random augmentation mode and apply it. Returns (audio, rel_bars) or (None, None)."""
    mode = random.choices(TEMPO_MODES, weights=TEMPO_WEIGHTS, k=1)[0]

    if mode == 'none':
        return _aug_none(audio, bar_starts, start_s)

    if mode == 'uniform':
        f = random.uniform(0.75, 1.33)
        return _aug_uniform(audio, bar_starts, start_s, f)

    if mode == 'sudden_jump':
        t_split = random.uniform(6, 26)
        f = random.uniform(0.5, 2.0)
        return _aug_sudden_jump(audio, bar_starts, start_s, t_split, f)

    if mode == 'gradual':
        f_start = random.uniform(0.8, 1.2)
        f_end = random.uniform(0.8, 1.2)
        return _aug_gradual(audio, bar_starts, start_s, f_start, f_end)

    return None, None


def process_audio(audio, bar_starts, stem, out_path, n_chunks, pitch_range=2.0, overwrite=False):
    """
    Core chunking logic on already-loaded, already-resampled audio (float32 at CHUNK_SR).

    stem:     base name for output files (no extension, no directory)
    out_path: directory to write .ogg + .bars pairs
    Returns:  number of chunks written
    """
    song_dur = len(audio) / CHUNK_SR
    # Need at least chunk_duration plus headroom for worst-case sudden_jump (f=2, t_split=26)
    if song_dur < chunk_duration + 2:
        print(f'  Song too short ({song_dur:.1f}s), skipping')
        return 0

    written = 0

    for i in range(n_chunks):
        # Leave enough source headroom; worst-case sudden_jump f=2 needs ~58s of source
        max_needed_src = chunk_duration * 2.5
        max_start = max(0.0, song_dur - max_needed_src)
        start_s = random.uniform(0.0, max_start)

        chunk_audio, rel_bars = _make_chunk(audio, bar_starts, start_s)
        if chunk_audio is None:
            continue

        # Pitch shift (applied after tempo, before writing)
        n_steps = random.uniform(-pitch_range, pitch_range)
        if abs(n_steps) >= 0.01:
            chunk_audio = librosa.effects.pitch_shift(
                chunk_audio.astype(np.float32), sr=CHUNK_SR, n_steps=n_steps)

        # Normalize
        peak = np.abs(chunk_audio).max()
        if peak > 0:
            chunk_audio = chunk_audio / peak

        out_stem = os.path.join(out_path, f'{stem}__{i:03d}')
        if not overwrite and os.path.exists(out_stem + '.ogg'):
            written += 1  # count already-done as success
            continue

        # Write .bars
        with open(out_stem + '.bars', 'w', encoding='utf8') as f:
            for b in rel_bars:
                f.write(f'{b:.6f}\n')

        # Write .ogg
        ok = _write_ogg(chunk_audio, out_stem + '.ogg')
        if not ok:
            print(f'  ffmpeg failed for chunk {i}')
            if os.path.exists(out_stem + '.bars'):
                os.remove(out_stem + '.bars')
            continue

        written += 1

    return written


def process_song(wav_path, json_path):
    """Generate args.chunks_per_song chunks from one song. Returns number written."""
    try:
        audio, sr = soundfile.read(wav_path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != CHUNK_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CHUNK_SR)
        with open(json_path, 'r', encoding='utf8') as f:
            meta = json.load(f)
        bar_starts = meta['bar_starts']
    except Exception as e:
        print(f'  Load error: {e}')
        return 0

    stem = Path(wav_path).stem
    return process_audio(audio, bar_starts, stem,
                         args.out_path, args.chunks_per_song, args.pitch_range, args.overwrite)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global args
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    in_path = Path(args.in_path)
    pairs = []
    for wav_file in sorted(in_path.glob('*.wav')):
        json_file = wav_file.with_suffix('.json')
        if json_file.exists():
            pairs.append((str(wav_file), str(json_file)))

    if not pairs:
        print(f'No WAV+JSON pairs found in {args.in_path}')
        sys.exit(1)

    random.shuffle(pairs)
    print(f'Found {len(pairs)} songs, generating {args.chunks_per_song} chunks each\n')

    total = 0
    for idx, (wav_path, json_path) in enumerate(pairs):
        print(f'[{idx+1}/{len(pairs)}] {Path(wav_path).stem}')
        n = process_song(wav_path, json_path)
        total += n
        print(f'  → {n} chunks written')

    print(f'\nDone: {total} chunks written to {args.out_path}')


if __name__ == '__main__':
    main()
