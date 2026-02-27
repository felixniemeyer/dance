#!/usr/bin/env python3
"""
annotate.py — Flask backend for the manual bar annotation tool.

Usage:
    python annotate.py --music-path ~/music --out-path data/chunks/v2

Development (Vite hot-reload):
    python annotate.py ...          # backend on :8050
    cd annotate_ui && npm run dev   # Vite dev server on :5173 (proxies /api + /audio)

Production (serve built UI from Flask):
    cd annotate_ui && npm run build
    python annotate.py ...          # serves UI + API on :8050
"""

import argparse
import mimetypes
import random
import secrets
import threading
from pathlib import Path

import subprocess

import librosa
import librosa.feature.rhythm   # explicit import needed for lazy-loader in librosa 0.10+
import numpy as np
from flask import Flask, abort, jsonify, request, send_file
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

# Madmom processors are expensive to instantiate (loads model weights from disk).
# Cache them at module level — one instance serves all songs.
_madmom_rnn: RNNBeatProcessor         | None = None
_madmom_dbn: DBNBeatTrackingProcessor | None = None

def _get_madmom_processors() -> tuple[RNNBeatProcessor, DBNBeatTrackingProcessor]:
    global _madmom_rnn, _madmom_dbn
    if _madmom_rnn is None:
        _madmom_rnn = RNNBeatProcessor()
        _madmom_dbn = DBNBeatTrackingProcessor(fps=100)
    return _madmom_rnn, _madmom_dbn

from chopper import process_audio_segmented
from config import samplerate as CHUNK_SR

# ── CLI ───────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Annotation tool backend')
    p.add_argument('--music-path', required=True,
                   help='Root folder to scan recursively (mp3/wav/flac/ogg/m4a)')
    p.add_argument('--out-path', required=True,
                   help='Chunk output directory')
    p.add_argument('--chunks-per-song', type=int, default=3)
    p.add_argument('--pitch-range', type=float, default=2.0)
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--librosa', action='store_true',
                   help='Use librosa beat tracker instead of madmom (slower, less accurate)')
    return p


# ── constants ─────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
HOP_LENGTH = 512


# ── module-level state ────────────────────────────────────────────────────────

_current_audio: np.ndarray | None = None   # decoded audio for the song currently being annotated
_token_to_file: dict[str, str]    = {}
_files:         list[str]         = []     # remaining queue (mutated in-place)
_state:         dict              = {'done': True}
_args  = None
_out_path: Path = None

# ── Background preload ────────────────────────────────────────────────────────

_preload_lock   = threading.Lock()
_preload_for:   str | None           = None   # filepath currently being preloaded
_preload_data:  dict | None          = None   # result dict when done, None while loading or if failed
_preload_thread: threading.Thread | None = None


# ── audio + beat detection ────────────────────────────────────────────────────

def _make_token(filepath: str) -> str:
    tok = secrets.token_urlsafe(8)
    _token_to_file.clear()
    _token_to_file[tok] = filepath
    return tok


def _load_audio_ffmpeg(filepath: str, sr: int) -> np.ndarray:
    """Decode any audio format via ffmpeg → mono float32 PCM at sr Hz."""
    cmd = [
        'ffmpeg', '-v', 'error',
        '-i', filepath,
        '-f', 'f32le', '-ac', '1', '-ar', str(sr),
        'pipe:1',
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {result.stderr.decode()[:300]}')
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _regularize_beats(beat_times: list[float], smooth_beats: int = 8) -> list[float]:
    """
    Smooth inter-beat intervals with a running median, then rebuild positions.

    Kills high-frequency jitter (from imperfect onset snapping or DP errors)
    while preserving slow tempo drift (live music without metronome).

    smooth_beats : window width in beats.  8 ≈ 2 bars at 4/4 → ~4 s at 120 BPM.
                   At each beat the local tempo is estimated as the median IBI
                   of its ±(smooth_beats//2) neighbours, so the effective
                   smoothing time is smooth_beats × beat_period.
    """
    arr = np.array(beat_times, dtype=np.float64)
    if len(arr) < 3:
        return beat_times
    ibis = np.diff(arr)
    half = smooth_beats // 2
    smooth = np.array([
        np.median(ibis[max(0, i - half) : i + half + 1])
        for i in range(len(ibis))
    ])
    out    = np.empty(len(arr))
    out[0] = arr[0]
    out[1:] = out[0] + np.cumsum(smooth)
    return out.tolist()


def _split_into_segments(beat_times: np.ndarray) -> list[tuple[int, int]]:
    """
    Split beat_times into tempo-coherent segments, returning (start_idx, end_idx) pairs.

    TODO: replace with madmom + proper top-down segmentation (moving-average MSD).
    Currently returns a single segment covering the whole track.
    """
    return [(0, len(beat_times))]


def _detect_beats_librosa(audio: np.ndarray) -> list[dict]:
    """
    Fallback beat detector using librosa (HPSS → DP beat tracker → IBI smoothing).
    Enabled with --librosa flag.
    """
    _, y_perc = librosa.effects.hpss(audio, margin=3.0)
    onset_env = librosa.onset.onset_strength(
        y=y_perc, sr=CHUNK_SR, hop_length=HOP_LENGTH,
        aggregate=np.median,
    )
    tempo_global = float(np.atleast_1d(
        librosa.feature.rhythm.tempo(onset_envelope=onset_env,
                                     sr=CHUNK_SR, hop_length=HOP_LENGTH)
    )[0])
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=CHUNK_SR,
        hop_length=HOP_LENGTH,
        start_bpm=tempo_global,
        tightness=100,
        trim=False,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=CHUNK_SR, hop_length=HOP_LENGTH)
    if len(beat_times) < 4:
        return []

    seg_slices = _split_into_segments(beat_times)
    segments: list[dict] = []
    for si, ei in seg_slices:
        seg_beats = beat_times[si:ei]
        if len(seg_beats) < 4:
            continue
        smoothed  = _regularize_beats(seg_beats.tolist())
        ibis      = np.diff(smoothed)
        seg_tempo = float(60.0 / np.median(ibis)) if len(ibis) else tempo_global
        segments.append({
            'start_time': float(seg_beats[0]),
            'end_time':   float(seg_beats[-1]),
            'beat_times': smoothed,
            'tempo':      seg_tempo,
        })
    return segments


def _detect_beats_madmom(filepath: str) -> list[dict]:
    """
    Primary beat detector using madmom's RNN + DBN pipeline.

    RNNBeatProcessor runs a bi-directional RNN over the audio to produce a
    beat activation function; DBNBeatTrackingProcessor decodes it with a
    dynamic Bayesian network, giving stable, accurately-timed beat positions.
    No post-hoc smoothing is applied — the DBN output is already clean.
    """
    rnn, dbn   = _get_madmom_processors()
    act        = rnn(filepath)
    beat_times = np.array(dbn(act))

    if len(beat_times) < 4:
        return []

    seg_slices = _split_into_segments(beat_times)
    segments: list[dict] = []
    for si, ei in seg_slices:
        seg_beats = beat_times[si:ei]
        if len(seg_beats) < 4:
            continue
        ibis      = np.diff(seg_beats)
        seg_tempo = float(60.0 / np.median(ibis)) if len(ibis) else 120.0
        segments.append({
            'start_time': float(seg_beats[0]),
            'end_time':   float(seg_beats[-1]),
            'beat_times': seg_beats.tolist(),
            'tempo':      seg_tempo,
        })
    return segments


def _load_song_data(filepath: str) -> dict | None:
    """
    Load audio and run beat detection for one file.
    Pure function — no global side-effects.
    Returns a data dict on success or None on failure/skip.
    """
    name = Path(filepath).name
    try:
        audio = _load_audio_ffmpeg(filepath, CHUNK_SR)
    except Exception as exc:
        print(f'  Load error ({name}): {exc}')
        return None
    try:
        if _args is not None and _args.librosa:
            segments = _detect_beats_librosa(audio)
        else:
            segments = _detect_beats_madmom(filepath)
    except Exception as exc:
        print(f'  Beat detection error ({name}): {exc}')
        return None
    if not segments:
        print(f'  No beats detected ({name}), skipping')
        return None
    return {
        '_filepath': filepath,
        '_audio':    audio,
        'filename':  name,
        'segments':  segments,
    }


# ── preload helpers ───────────────────────────────────────────────────────────

def _preload_worker(filepath: str) -> None:
    """Thread target: load + analyse one file, store result under lock."""
    data = _load_song_data(filepath)
    with _preload_lock:
        global _preload_data
        if _preload_for == filepath:   # still relevant (not superseded)
            _preload_data = data       # None means failed


def _start_preload(filepath: str) -> None:
    """Start background preloading of filepath (no-op if already loading it)."""
    global _preload_for, _preload_data, _preload_thread
    with _preload_lock:
        if _preload_for == filepath:
            return   # already loading this file
        _preload_for  = filepath
        _preload_data = None
    print(f'\n[preload] {Path(filepath).name}')
    _preload_thread = threading.Thread(
        target=_preload_worker, args=(filepath,), daemon=True)
    _preload_thread.start()


def _consume_preload(filepath: str) -> dict | None:
    """
    Return preloaded data for filepath if available.
    Blocks (joins thread) if the preload is still in progress.
    Returns None if the file wasn't preloaded or preload failed.
    """
    global _preload_data, _preload_for
    # If the thread is still running for this exact file, wait for it
    if _preload_thread is not None and _preload_thread.is_alive():
        with _preload_lock:
            target = _preload_for
        if target == filepath:
            print('  [waiting for preload]')
            _preload_thread.join()
    # Collect result
    with _preload_lock:
        if _preload_for == filepath and _preload_data is not None:
            data = _preload_data
            _preload_data = None
            _preload_for  = None
            return data
    return None


# ── file queue ────────────────────────────────────────────────────────────────

def _advance(max_skip: int = 20) -> None:
    """Pop next file from _files, use preloaded data if available, update _state."""
    global _state, _current_audio
    for _ in range(max_skip + 1):
        if not _files:
            _state = {'done': True, 'remaining': 0}
            print('\n[annotate] All files done!')
            return
        filepath = _files.pop(0)
        if not Path(filepath).exists():
            continue
        name = Path(filepath).name
        print(f'\n[annotate] {name}  ({len(_files)} remaining)')

        # ── Use preloaded data if available ──────────────────────────────────
        data = _consume_preload(filepath)
        if data is None:
            print('  [preload miss — loading synchronously]')
            data = _load_song_data(filepath)
        else:
            print('  [preload hit]')

        if data is None:
            continue   # file failed analysis — try the next one

        n_segs      = len(data['segments'])
        total_beats = sum(len(s['beat_times']) for s in data['segments'])
        print(f'  {n_segs} segment(s)  {total_beats} beats')

        # ── Activate current song ─────────────────────────────────────────────
        _current_audio = data['_audio']
        token = _make_token(filepath)
        _state = {
            'done':      False,
            '_filepath': filepath,
            'token':     token,
            'filename':  name,
            'segments':  [{'id': i, **seg} for i, seg in enumerate(data['segments'])],
            'remaining': len(_files),
        }

        # ── Kick off preload for the next queued file ─────────────────────────
        if _files:
            _start_preload(_files[0])
        return

    _state = {'done': True, 'remaining': 0}


def _public_state() -> dict:
    """Strip internal keys before sending to client."""
    if _state.get('done'):
        return {'done': True, 'remaining': 0}
    with _preload_lock:
        preloading = _preload_thread is not None and _preload_thread.is_alive()
    return {
        'done':       False,
        'token':      _state['token'],
        'filename':   _state['filename'],
        'segments':   _state['segments'],
        'remaining':  _state['remaining'],
        'preloading': preloading,
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.after_request
def _cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/api/state')
def api_state():
    return jsonify(_public_state())


@app.route('/api/action', methods=['POST', 'OPTIONS'])
def api_action():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json(force=True)
    act  = data.get('type', '')

    if act == 'save' and not _state.get('done'):
        if _current_audio is not None:
            payload_segs = data.get('segments', [])
            if not payload_segs:
                return jsonify({'error': 'segments missing'}), 400
            stem = Path(_state['_filepath']).stem
            n = process_audio_segmented(
                _current_audio, payload_segs, stem,
                str(_out_path), _args.chunks_per_song, _args.pitch_range,
                overwrite=False,
            )
            print(f'  Saved {n} chunks for {stem}')

    elif act == 'skip':
        print(f'  Skipped {_state.get("filename", "")}')

    _advance()
    return jsonify(_public_state())


@app.route('/audio/<token>')
def serve_audio(token):
    fp = _token_to_file.get(token)
    if not fp:
        abort(404)
    mime = mimetypes.guess_type(fp)[0] or 'application/octet-stream'
    return send_file(fp, mimetype=mime, conditional=True)


# Serve the built Vite UI (production mode)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_ui(path):
    dist = Path(__file__).parent / 'annotate_ui' / 'dist'
    if not dist.exists():
        return (
            'UI not built. Run:  cd annotate_ui && npm run build\n'
            'Or use the Vite dev server: cd annotate_ui && npm run dev',
            503,
        )
    target = dist / path
    if path and target.is_file():
        return send_file(str(target))
    return send_file(str(dist / 'index.html'))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global _args, _out_path
    _args     = _make_parser().parse_args()
    _out_path = Path(_args.out_path)
    _out_path.mkdir(parents=True, exist_ok=True)

    # Find already-processed stems
    done_stems: set[str] = set()
    for f in _out_path.glob('*.ogg'):
        parts = f.stem.rsplit('__', 1)
        if len(parts) == 2:
            done_stems.add(parts[0])

    # Collect audio files
    music_path = Path(_args.music_path)
    all_files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        all_files.extend(music_path.rglob(f'*{ext}'))
        all_files.extend(music_path.rglob(f'*{ext.upper()}'))
    all_files = list({f.resolve() for f in all_files})
    random.shuffle(all_files)
    _files[:] = [str(f) for f in all_files if f.stem not in done_stems]

    print(f'Found {len(all_files)} audio files; {len(_files)} left to annotate')
    if not _files:
        print('Nothing left to annotate.')
        return

    _advance()   # load first file + start preloading second
    print(f'Starting backend → http://{_args.host}:{_args.port}')
    app.run(host=_args.host, port=_args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
