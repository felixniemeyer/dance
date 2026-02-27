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
from pathlib import Path

import librosa
import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from chopper import process_audio
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
    return p


# ── constants ─────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
HOP_LENGTH = 512

# ── module-level state ────────────────────────────────────────────────────────

_audio_cache:   dict[str, np.ndarray] = {}
_token_to_file: dict[str, str]        = {}
_files:         list[str]             = []   # remaining queue (mutated in-place)
_state:         dict                  = {'done': True}
_args  = None
_out_path: Path = None


# ── audio + beat detection ────────────────────────────────────────────────────

def _load_audio(filepath: str) -> np.ndarray:
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    audio, _ = librosa.load(filepath, sr=CHUNK_SR, mono=True)
    _audio_cache.clear()
    _audio_cache[filepath] = audio
    return audio


def _make_token(filepath: str) -> str:
    tok = secrets.token_urlsafe(8)
    _token_to_file.clear()
    _token_to_file[tok] = filepath
    return tok


def _detect_beats(audio: np.ndarray) -> tuple[float, list[float]]:
    """
    Constant-tempo beat grid aligned to onset peaks (Mixxx-style):
      1. Estimate a single global BPM.
      2. Grid-search phase offset to maximise onset energy at beat positions.
      3. Return an evenly-spaced beat-time array.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=CHUNK_SR, hop_length=HOP_LENGTH)

    tempo = float(np.atleast_1d(
        librosa.beat.tempo(onset_envelope=onset_env,
                           sr=CHUNK_SR, hop_length=HOP_LENGTH)
    )[0])

    fps          = CHUNK_SR / HOP_LENGTH
    beat_period  = fps * 60.0 / tempo
    n            = len(onset_env)
    n_phase      = max(1, int(round(beat_period)))

    best_score, best_phase = -1.0, 0
    for ph in range(n_phase):
        idxs  = np.round(np.arange(ph, n, beat_period)).astype(int)
        idxs  = idxs[idxs < n]
        score = float(onset_env[idxs].sum())
        if score > best_score:
            best_score, best_phase = score, ph

    frames     = np.round(np.arange(best_phase, n, beat_period)).astype(int)
    frames     = frames[frames < n]
    beat_times = librosa.frames_to_time(frames, sr=CHUNK_SR, hop_length=HOP_LENGTH)
    return tempo, beat_times.tolist()


def _compute_bar_starts(beat_times: list, beat_idx: int, bpb: int) -> list[float]:
    return list(beat_times[beat_idx::bpb])


# ── file queue ────────────────────────────────────────────────────────────────

def _advance(max_skip: int = 20) -> None:
    """Pop next file from _files, load it, update _state."""
    global _state
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
        try:
            audio = _load_audio(filepath)
        except Exception as exc:
            print(f'  Load error: {exc}, skipping…')
            continue
        try:
            tempo, beat_times = _detect_beats(audio)
        except Exception as exc:
            print(f'  Beat detection error: {exc}, skipping…')
            continue
        if not beat_times:
            print('  No beats detected, skipping…')
            continue
        print(f'  {tempo:.1f} BPM  {len(beat_times)} beats')
        token  = _make_token(filepath)
        _state = {
            'done':       False,
            '_filepath':  filepath,   # internal only
            'token':      token,
            'filename':   name,
            'beat_times': beat_times,
            'tempo':      tempo,
            'remaining':  len(_files),
        }
        return
    _state = {'done': True, 'remaining': 0}


def _public_state() -> dict:
    """Strip internal keys before sending to client."""
    if _state.get('done'):
        return {'done': True, 'remaining': 0}
    return {
        'done':       False,
        'token':      _state['token'],
        'filename':   _state['filename'],
        'beat_times': _state['beat_times'],
        'tempo':      _state['tempo'],
        'remaining':  _state['remaining'],
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
        fp = _state.get('_filepath')
        if fp and fp in _audio_cache:
            bt  = _state.get('beat_times', [])
            bi  = int(data.get('beat_idx', 0))
            bpb = int(data.get('bpb', 4))
            bar_starts = _compute_bar_starts(bt, bi, bpb)
            stem = Path(fp).stem
            n    = process_audio(
                _audio_cache[fp], bar_starts, stem,
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

    _advance()   # load first file
    print(f'Starting backend → http://{_args.host}:{_args.port}')
    app.run(host=_args.host, port=_args.port, debug=False, threaded=False)


if __name__ == '__main__':
    main()
