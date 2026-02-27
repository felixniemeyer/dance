"""
Chop rendered WAV files into training chunks with tempo/pitch augmentation.

Usage:
    python chopper.py \
        --in-path data/rendered/v2 \
        --out-path data/chunks/v2 \
        --chunks-per-song 3

Augmentation modes (weighted random per chunk):
    none        25%  — raw slice, pitch shift only
    uniform     25%  — time-stretch full chunk by f ~ U(0.75, 1.33)
    sudden_jump 35%  — rate=1 up to a bar boundary, then f ~ U(0.5, 2.0)
    gradual     15%  — N segments with linearly interpolated stretch factors

Time-stretch and pitch-shift are always applied together in a single rubberband
pass (no double vocoding).  Pitch shift ~ U(-pitch_range, +pitch_range) semitones
is sampled once per chunk and applied uniformly.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile
import subprocess
import tempfile

from config import chunk_duration, samplerate as CHUNK_SR

parser = argparse.ArgumentParser(description='Chop rendered WAV files into training chunks.')
parser.add_argument('--in-path', type=str, required=True,
    help='path to folder containing .wav + .json pairs')
parser.add_argument('--out-path', type=str, required=True,
    help='path to write chunk .ogg + .bars files')
parser.add_argument('--chunks-per-song', type=int, default=3)
parser.add_argument('--pitch-range', type=float, default=2.0,
    help='max pitch shift in semitones')
parser.add_argument('--overwrite', default=False, action='store_true')

args = None  # set by main() when run as script

TEMPO_MODES    = ['none', 'uniform', 'sudden_jump', 'gradual']
TEMPO_WEIGHTS  = [0.25,   0.25,      0.35,          0.15]
N_GRADUAL_SEGS = 8


# ── audio helpers ──────────────────────────────────────────────────────────────

def _stretch_and_pitch(audio: np.ndarray, rate: float, n_semitones: float) -> np.ndarray:
    """
    Time-stretch + pitch-shift in a single rubberband pass.

    rate       : >1 = faster (shorter output), <1 = slower, 1.0 = unchanged
    n_semitones: positive = pitch up, negative = pitch down, 0 = unchanged

    Using a single rubberband invocation avoids the double-artefact problem of
    running a phase vocoder after WSOLA (or vice versa).
    """
    a = audio.astype(np.float32)
    no_stretch = abs(rate - 1.0) < 1e-4
    no_pitch   = abs(n_semitones) < 0.01

    if no_stretch and no_pitch:
        return a
    if no_stretch:
        return pyrb.pitch_shift(a, CHUNK_SR, n_steps=n_semitones)
    rbargs = {'--pitch': str(n_semitones)} if not no_pitch else {}
    return pyrb.time_stretch(a, CHUNK_SR, rate=rate, rbargs=rbargs)


def _ensure_len(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly n_samples."""
    if len(audio) > n_samples:
        return audio[:n_samples]
    if len(audio) < n_samples:
        return np.pad(audio, (0, n_samples - len(audio)))
    return audio


def _write_ogg(audio: np.ndarray, out_path: str) -> bool:
    """Encode float32 array to OGG Vorbis q6 via ffmpeg."""
    fd, tmpwav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        soundfile.write(tmpwav, audio, CHUNK_SR)
        cmd = [
            'ffmpeg', '-v', 'warning', '-y',
            '-i', tmpwav,
            '-ac', '1', '-c:a', 'libvorbis', '-qscale:a', '6',
            '-ar', str(CHUNK_SR), out_path,
        ]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0
    finally:
        if os.path.exists(tmpwav):
            os.remove(tmpwav)


# ── augmentation modes ─────────────────────────────────────────────────────────

def _aug_none(audio, bar_starts, start_s, n_semitones):
    """No tempo change; pitch shift only."""
    s = int(start_s * CHUNK_SR)
    e = s + int(chunk_duration * CHUNK_SR)
    if e > len(audio):
        return None, None
    rel = [b - start_s for b in bar_starts if start_s <= b < start_s + chunk_duration]
    if len(rel) < 2:
        return None, None
    chunk = _stretch_and_pitch(audio[s:e], 1.0, n_semitones)
    return chunk, rel


def _aug_uniform(audio, bar_starts, start_s, f, n_semitones):
    """Uniform stretch: consume chunk_duration×f source seconds, output chunk_duration."""
    src_dur = chunk_duration * f
    s = int(start_s * CHUNK_SR)
    e = s + int(src_dur * CHUNK_SR)
    if e > len(audio):
        return None, None
    chunk = _ensure_len(_stretch_and_pitch(audio[s:e], f, n_semitones),
                        int(chunk_duration * CHUNK_SR))
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


def _aug_sudden_jump(audio, bar_starts, start_s, f, n_semitones):
    """
    Rate=1 up to a bar boundary chosen from those in [6, 26] s of the chunk,
    then rate=f for the rest.  Snapping to a bar boundary ensures the tempo
    change lands exactly on a musical phrase boundary.
    """
    # Candidate bar boundaries in [6, 26] s relative to chunk start
    candidates = [b - start_s for b in bar_starts if 6.0 <= b - start_s <= 26.0]
    if not candidates:
        return None, None
    t_split = random.choice(candidates)

    abs_split    = start_s + t_split
    needed_src   = (chunk_duration - t_split) * f
    first_s  = int(start_s  * CHUNK_SR)
    first_e  = int(abs_split * CHUNK_SR)
    second_s = first_e
    second_e = second_s + int(needed_src * CHUNK_SR)

    if first_e > len(audio) or second_e > len(audio):
        return None, None

    first_audio  = _stretch_and_pitch(audio[first_s:first_e],  1.0, n_semitones)
    second_src   = audio[second_s:second_e]
    if len(second_src) == 0:
        return None, None
    target_second = int((chunk_duration - t_split) * CHUNK_SR)
    second_audio  = _ensure_len(_stretch_and_pitch(second_src, f, n_semitones), target_second)
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


def _aug_gradual(audio, bar_starts, start_s, f_start, f_end, n_semitones):
    """N segments with linearly interpolated stretch factors f_start → f_end."""
    n = N_GRADUAL_SEGS
    seg_out_dur     = chunk_duration / n
    seg_out_samples = int(seg_out_dur * CHUNK_SR)

    seg_src_starts, seg_src_ends, fs = [], [], []
    src_cur = start_s
    for k in range(n):
        fk = f_start + (f_end - f_start) * k / max(n - 1, 1)
        fs.append(fk)
        seg_src_starts.append(src_cur)
        seg_src_ends.append(src_cur + seg_out_dur * fk)
        src_cur += seg_out_dur * fk

    if int(src_cur * CHUNK_SR) > len(audio):
        return None, None

    segments = []
    for k in range(n):
        s = int(seg_src_starts[k] * CHUNK_SR)
        e = int(seg_src_ends[k]   * CHUNK_SR)
        src_seg = audio[s:e]
        if len(src_seg) == 0:
            return None, None
        segments.append(
            _ensure_len(_stretch_and_pitch(src_seg, fs[k], n_semitones), seg_out_samples)
        )

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

def _make_chunk(audio, bar_starts, start_s, n_semitones):
    """Pick a random augmentation mode and apply it. Returns (audio, rel_bars) or (None, None)."""
    mode = random.choices(TEMPO_MODES, weights=TEMPO_WEIGHTS, k=1)[0]

    if mode == 'none':
        return _aug_none(audio, bar_starts, start_s, n_semitones)

    if mode == 'uniform':
        f = random.uniform(0.75, 1.33)
        return _aug_uniform(audio, bar_starts, start_s, f, n_semitones)

    if mode == 'sudden_jump':
        f = random.uniform(0.5, 2.0)
        return _aug_sudden_jump(audio, bar_starts, start_s, f, n_semitones)

    if mode == 'gradual':
        f_start = random.uniform(0.8, 1.2)
        f_end   = random.uniform(0.8, 1.2)
        return _aug_gradual(audio, bar_starts, start_s, f_start, f_end, n_semitones)

    return None, None


def process_audio(audio, bar_starts, stem, out_path, n_chunks, pitch_range=2.0, overwrite=False):
    """
    Core chunking + augmentation on already-loaded float32 audio at CHUNK_SR.

    stem:     base name for output files (no extension, no directory)
    out_path: directory to write .ogg + .bars pairs
    Returns:  number of chunks written
    """
    song_dur = len(audio) / CHUNK_SR
    if song_dur < chunk_duration + 2:
        print(f'  Song too short ({song_dur:.1f}s), skipping')
        return 0

    os.makedirs(out_path, exist_ok=True)
    written = 0

    for i in range(n_chunks):
        # Leave enough source headroom for worst-case sudden_jump (f=2, t_split=26 → ~58s)
        max_start = max(0.0, song_dur - chunk_duration * 2.5)
        start_s   = random.uniform(0.0, max_start)

        # Single pitch shift value for the whole chunk
        n_semitones = random.uniform(-pitch_range, pitch_range)

        chunk_audio, rel_bars = _make_chunk(audio, bar_starts, start_s, n_semitones)
        if chunk_audio is None:
            continue

        # Normalize
        peak = np.abs(chunk_audio).max()
        if peak > 0:
            chunk_audio = chunk_audio / peak

        out_stem = os.path.join(out_path, f'{stem}__{i:03d}')
        if not overwrite and os.path.exists(out_stem + '.ogg'):
            written += 1
            continue

        with open(out_stem + '.bars', 'w', encoding='utf8') as f:
            for b in rel_bars:
                f.write(f'{b:.6f}\n')

        if not _write_ogg(chunk_audio, out_stem + '.ogg'):
            print(f'  ffmpeg failed for chunk {i}')
            if os.path.exists(out_stem + '.bars'):
                os.remove(out_stem + '.bars')
            continue

        written += 1

    return written


def process_audio_segmented(audio, segments, stem, out_path, n_chunks,
                            pitch_range=2.0, overwrite=False):
    """
    Generate chunks from a list of annotated segments.

    segments : list of {start, end, bar_starts, status}
               as sent by the frontend save payload.
               Only 'accepted' segments contribute audio; a 'skipped' segment
               breaks contiguity between accepted ones.

    Chunks are drawn only from contiguous accepted runs.  n_chunks is
    distributed across runs proportionally to their duration (≥1 attempt per
    run if budget allows).
    """
    song_dur = len(audio) / CHUNK_SR

    # Sort by start time so we can detect contiguous runs in order
    sorted_segs = sorted(segments, key=lambda s: s['start'])

    # Build contiguous accepted runs: a 'skipped' segment acts as a barrier
    runs: list[list[dict]] = []
    current_run: list[dict] = []
    for seg in sorted_segs:
        status = seg.get('status', 'pending')
        if status == 'accepted':
            current_run.append(seg)
        elif status == 'skipped':
            if current_run:
                runs.append(current_run)
            current_run = []
        # 'pending' segments are treated as accepted gaps — keep current run open
        # (shouldn't happen at save time, but be defensive)
    if current_run:
        runs.append(current_run)

    if not runs:
        print('  No accepted segments — nothing written')
        return 0

    os.makedirs(out_path, exist_ok=True)

    # Proportional chunk budget (at least 1 attempt per run)
    run_durs   = [sum(s['end'] - s['start'] for s in run) for run in runs]
    total_dur  = sum(run_durs) or 1.0
    run_budget = []
    remaining  = n_chunks
    for i, dur in enumerate(run_durs):
        if i == len(run_durs) - 1:
            run_budget.append(max(1, remaining))
        else:
            alloc = max(1, round(n_chunks * dur / total_dur))
            run_budget.append(alloc)
            remaining -= alloc

    written   = 0
    chunk_ctr = 0

    for run, n_run in zip(runs, run_budget):
        run_start = run[0]['start']
        run_end   = run[-1]['end']
        run_dur   = run_end - run_start

        if run_dur < chunk_duration + 2:
            print(f'  Run [{run_start:.1f}–{run_end:.1f}s] too short, skipping')
            continue

        # Collect all bar_starts for this run (augmentation modes may look
        # slightly beyond the chunk window, so pass the full run's bars)
        run_bar_starts = sorted(
            b for seg in run for b in seg.get('bar_starts', [])
        )

        for _ in range(n_run):
            max_start = max(run_start, run_end - chunk_duration * 2.5)
            start_s   = random.uniform(run_start, max_start)

            n_semitones = random.uniform(-pitch_range, pitch_range)
            chunk_audio, rel_bars = _make_chunk(
                audio, run_bar_starts, start_s, n_semitones)
            if chunk_audio is None:
                continue

            peak = np.abs(chunk_audio).max()
            if peak > 0:
                chunk_audio = chunk_audio / peak

            out_stem = os.path.join(out_path, f'{stem}__{chunk_ctr:03d}')
            chunk_ctr += 1

            if not overwrite and os.path.exists(out_stem + '.ogg'):
                written += 1
                continue

            with open(out_stem + '.bars', 'w', encoding='utf8') as f:
                for b in rel_bars:
                    f.write(f'{b:.6f}\n')

            if not _write_ogg(chunk_audio, out_stem + '.ogg'):
                print(f'  ffmpeg failed for chunk {chunk_ctr}')
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
