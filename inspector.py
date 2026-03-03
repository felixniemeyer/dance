"""
inspector_api.py — Flask REST API backend for the Dance inspector UI tab.

Strips all Dash/Plotly UI from inspector.py — exposes JSON endpoints
consumed by the Vue/Vite frontend at /inspect/*.

Usage:
    python inspector_api.py \\
        --chunks-path data/chunks/<run> \\
        --test-data-path realworld-test-data \\
        --checkpoints-path checkpoints \\
        --port 8051
"""

import argparse
import os
import random
import subprocess

import numpy as np
import soundfile
import torch
from pathlib import Path
from flask import Flask, abort, jsonify, request, send_file

import config
from models.selector import loadModel
from real_world_audio_utils import make_audio_response, scan_audio_files

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--chunks-path',      type=str, required=True)
parser.add_argument('--music-path',       type=str, default=None,
                    help='Root folder of real-world audio library (scanned recursively)')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints')
parser.add_argument('--port',             type=int, default=8051)
parser.add_argument('--host',             default='0.0.0.0')
args = parser.parse_args()

# ── Discovery ──────────────────────────────────────────────────────────────────

AUDIO_EXTS = {'.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac'}


def find_chunks(path):
    if not os.path.isdir(path):
        print(f'[inspector] chunks path not found: {path!r}')
        return []
    names = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.ogg'):
            stem = f[:-4]
            if os.path.exists(os.path.join(path, stem + '.bars')):
                names.append(stem)
    return names



def find_checkpoints(path):
    """Returns (tags, tag_epochs) — tags ordered by recency, epochs descending."""
    if not os.path.isdir(path):
        return [], {}
    tag_epochs = {}
    tag_mtime  = {}
    for pt in Path(path).rglob('*.pt'):
        tag = pt.parent.name
        try:
            epoch = int(pt.stem)
        except ValueError:
            continue
        tag_epochs.setdefault(tag, []).append(epoch)
        tag_mtime[tag] = max(tag_mtime.get(tag, 0), pt.stat().st_mtime)
    tags = sorted(tag_epochs, key=lambda t: tag_mtime[t], reverse=True)
    for tag in tags:
        tag_epochs[tag] = sorted(tag_epochs[tag], reverse=True)
    return tags, tag_epochs

# ── Music library (optional --music-path) ──────────────────────────────────────

# relative_path → absolute_path, populated at startup
_music_files: dict[str, str] = {}


def scan_music_library(music_path: str) -> dict[str, str]:
    """Scan music-path and return all audio files from cache.
    Returns {relative_path: absolute_path}. No duration filtering at startup."""
    abs_paths = scan_audio_files(music_path)
    root = music_path.rstrip('/')
    result = {}
    for ap in abs_paths:
        try:
            rel = os.path.relpath(ap, root)
        except ValueError:
            rel = os.path.basename(ap)
        result[rel] = ap
    return result


# ── Audio loading ──────────────────────────────────────────────────────────────

def _load_audio_ffmpeg(path):
    cmd = [
        'ffmpeg', '-v', 'warning',
        '-i', path,
        '-ac', '1', '-ar', str(config.samplerate),
        '-f', 'f32le', '-',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {result.stderr.decode()}')
    return np.frombuffer(result.stdout, dtype=np.float32).copy(), config.samplerate


def load_audio(path):
    ext = Path(path).suffix.lower()
    if ext in {'.ogg', '.wav', '.flac'}:
        audio, sr = soundfile.read(path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    return _load_audio_ffmpeg(path)

# ── Label data ─────────────────────────────────────────────────────────────────

def _phase_from_bar_times(bar_times, duration):
    """Reconstruct sawtooth phase label from bar-start times (.bars format)."""
    fps      = config.samplerate / config.frame_size
    n_frames = int(duration * fps) + 1
    times    = np.arange(n_frames) / fps
    phase    = np.full(n_frames, np.nan, dtype=np.float32)
    bars     = sorted(bar_times)
    for i, t0 in enumerate(bars):
        t1      = bars[i + 1] if i + 1 < len(bars) else t0 + (bars[-1] - bars[-2] if len(bars) >= 2 else 2.0)
        bar_dur = max(t1 - t0, 1e-6)
        mask    = (times >= t0) & (times < t1)
        phase[mask] = (times[mask] - t0) / bar_dur
    valid = ~np.isnan(phase)
    return times[valid].tolist(), phase[valid].tolist()


def load_chunk_label(stem):
    """Return label data for a chunk: bar_times, phase_times, phase, duration."""
    ogg_path  = os.path.join(args.chunks_path, stem + '.ogg')
    bars_path = os.path.join(args.chunks_path, stem + '.bars')
    audio, sr = soundfile.read(ogg_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    with open(bars_path) as f:
        bar_times = sorted(float(l) for l in f if l.strip())
    duration              = len(audio) / sr
    phase_times, phase    = _phase_from_bar_times(bar_times, duration)
    return {
        'duration':    duration,
        'bar_times':   bar_times,
        'phase_times': phase_times,
        'phase':       phase,
    }

# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(ckpt_tag, ckpt_epoch, audio_path):
    ckpt_path = os.path.join(args.checkpoints_path, ckpt_tag, f'{ckpt_epoch}.pt')
    audio, sr = load_audio(audio_path)
    audio_t  = torch.from_numpy(audio)
    length   = len(audio_t)
    frames_n = length // config.frame_size
    offset   = length % config.frame_size
    seq = audio_t[offset:offset + frames_n * config.frame_size].reshape(1, frames_n, config.frame_size)
    model, _ = loadModel(ckpt_path)
    model.eval()
    with torch.no_grad():
        out, _ = model(seq)
    angles = torch.atan2(out[0, :, 0], out[0, :, 1])
    phases = torch.remainder(angles / (2 * torch.pi), 1.0).numpy()
    frame_dur = config.frame_size / sr
    times = np.arange(frames_n) * frame_dur + frame_dur / 2
    return times.tolist(), phases.tolist()

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.after_request
def _cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/chunks')
def list_chunks():
    return jsonify({'chunks': find_chunks(args.chunks_path)})


@app.route('/checkpoints')
def list_checkpoints():
    tags, tag_epochs = find_checkpoints(args.checkpoints_path)
    return jsonify({'tags': tags, 'epochs': tag_epochs})


@app.route('/chunk-data/<path:stem>')
def chunk_data(stem):
    try:
        return jsonify(load_chunk_label(stem))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chunk-audio/<path:filename>')
def chunk_audio(filename):
    path = os.path.join(args.chunks_path, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype='audio/ogg')


@app.route('/music-files')
def list_music_files():
    return jsonify({'files': sorted(_music_files.keys())})


@app.route('/music-audio/<path:relpath>')
def music_audio(relpath):
    abs_path = _music_files.get(relpath)
    if not abs_path or not os.path.exists(abs_path):
        abort(404)
    return make_audio_response(abs_path)


@app.route('/delete', methods=['POST', 'OPTIONS'])
def delete_chunks():
    if request.method == 'OPTIONS':
        return '', 200
    data  = request.get_json(force=True)
    stem  = data.get('stem', '')
    scope = data.get('scope', 'this')   # 'this' | 'song' | 'sf'

    def parse_stem(s):
        parts = s.split('__')
        if len(parts) >= 3:
            return parts[0], parts[-2]   # song, sf
        if len(parts) == 2:
            return parts[0], None
        return s, None

    all_stems = find_chunks(args.chunks_path)

    if scope == 'this':
        targets = [stem]
    elif scope == 'song':
        song, _ = parse_stem(stem)
        targets = [s for s in all_stems if parse_stem(s)[0] == song]
    elif scope == 'sf':
        _, sf = parse_stem(stem)
        targets = [s for s in all_stems if sf and parse_stem(s)[1] == sf]
    else:
        return jsonify({'error': 'invalid scope'}), 400

    deleted = 0
    for t in targets:
        for ext in ('.ogg', '.bars'):
            path = os.path.join(args.chunks_path, t + ext)
            if os.path.exists(path):
                os.remove(path)
                if ext == '.ogg':
                    deleted += 1

    return jsonify({'deleted': deleted, 'stems': targets})


@app.route('/infer', methods=['POST', 'OPTIONS'])
def infer():
    if request.method == 'OPTIONS':
        return '', 200
    data  = request.get_json(force=True)
    source = data.get('source')   # 'chunk' | 'test'
    file   = data.get('file', '')
    tag    = data.get('tag', '')
    epoch  = data.get('epoch')
    if not tag or epoch is None:
        return jsonify({'error': 'tag and epoch required'}), 400
    if source == 'chunk':
        audio_path = os.path.join(args.chunks_path, file + '.ogg')
    elif source == 'music':
        audio_path = _music_files.get(file)
        if not audio_path:
            return jsonify({'error': f'music file not found: {file}'}), 404
    else:
        return jsonify({'error': 'source must be chunk or music'}), 400
    if not os.path.exists(audio_path):
        return jsonify({'error': f'file not found: {audio_path}'}), 404
    try:
        times, phases = run_inference(tag, epoch, audio_path)
        return jsonify({'times': times, 'phases': phases})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import socket
    ip      = socket.gethostbyname(socket.gethostname())
    chunks  = find_chunks(args.chunks_path)
    tags, _ = find_checkpoints(args.checkpoints_path)

    if args.music_path:
        print(f'Scanning music library: {args.music_path} …')
        _music_files.update(scan_music_library(args.music_path))
        print(f'  → {len(_music_files)} tracks loaded')

    print(f'\nDance Inspector API')
    print(f'  chunks:      {args.chunks_path}  ({len(chunks)} chunks)')
    if args.music_path:
        print(f'  music:       {args.music_path}  ({len(_music_files)} tracks)')
    print(f'  checkpoints: {args.checkpoints_path}  ({len(tags)} tags)')
    print(f'  http://{ip}:{args.port}\n')
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
