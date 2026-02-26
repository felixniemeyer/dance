#!/usr/bin/env python3
"""
annotate.py — interactive manual bar annotation tool for the dance project.

Opens a Dash browser tab. For each audio file in --music-path:
  - Loads audio, runs librosa beat detection
  - Shows waveform + beat markers in a Plotly figure
  - Keyboard controls adjust the downbeat position and time signature
  - Enter saves chunks via chunk.process_audio; Esc skips the file

Keyboard controls:
  h / l     Move selected downbeat left / right by one beat
  1-9       Set beats per bar
  Space     Toggle loop playback of the current bar
  Enter     Save chunks + advance to next file
  Esc       Skip file, advance to next

Usage:
    python annotate.py --music-path ~/music --out-path data/chunks/v2
"""

import argparse
import mimetypes
import random
import secrets
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import flask
import librosa
import numpy as np
import plotly.graph_objects as go

from chopper import process_audio
from config import samplerate as CHUNK_SR


# ── CLI ───────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Interactive bar annotation tool for dance training data.')
    p.add_argument('--music-path', required=True,
                   help='Root folder to scan recursively (mp3/wav/flac/ogg/m4a)')
    p.add_argument('--out-path', required=True,
                   help='Chunk output directory (same format as pipeline output)')
    p.add_argument('--chunks-per-song', type=int, default=3)
    p.add_argument('--pitch-range', type=float, default=2.0,
                   help='Max pitch shift in semitones (passed to process_audio)')
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--host', default='0.0.0.0',
                   help='Host to bind (default 0.0.0.0 = all interfaces)')
    return p


# ── constants ─────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
WAVEFORM_POINTS  = 2000


# ── module-level shared state ─────────────────────────────────────────────────
# Holds at most the current file; cleared on every file change.

_audio_cache:    dict[str, np.ndarray] = {}   # filepath → float32 mono at CHUNK_SR
_token_to_file:  dict[str, str]        = {}   # token → filepath (one entry max)


# ── audio helpers ─────────────────────────────────────────────────────────────

def _load_audio(filepath: str) -> np.ndarray:
    """Load + resample to CHUNK_SR, mono. Caches one file at a time."""
    if filepath in _audio_cache:
        return _audio_cache[filepath]
    # librosa.load handles MP3/FLAC/WAV/OGG and resampling transparently
    audio, _ = librosa.load(filepath, sr=CHUNK_SR, mono=True)
    _audio_cache.clear()
    _audio_cache[filepath] = audio
    return audio


def _make_token(filepath: str) -> str:
    """Register filepath under a fresh random token; clear the previous one."""
    tok = secrets.token_urlsafe(8)
    _token_to_file.clear()
    _token_to_file[tok] = filepath
    return tok


def _compute_waveform(audio: np.ndarray) -> tuple[list, list]:
    """Downsample abs envelope to ≈WAVEFORM_POINTS for fast Plotly display."""
    step = max(1, len(audio) // WAVEFORM_POINTS)
    n    = (len(audio) // step) * step
    env  = np.abs(audio[:n]).reshape(-1, step).max(axis=1)
    times = (np.arange(len(env)) * step / CHUNK_SR).tolist()
    return times, env.tolist()


def _detect_beats(audio: np.ndarray) -> list[float]:
    """Return beat times in seconds via librosa.beat.beat_track."""
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=CHUNK_SR)
    return librosa.frames_to_time(beat_frames, sr=CHUNK_SR).tolist()


# ── bar maths ─────────────────────────────────────────────────────────────────

def compute_bar_starts(beat_times: list, beat_idx: int, beats_per_bar: int) -> list[float]:
    """Every beats_per_bar-th beat starting from beat_idx."""
    return list(beat_times[beat_idx::beats_per_bar])


def _bar_extents(beat_times: list, beat_idx: int, beats_per_bar: int) -> tuple[float, float]:
    """(bar_start_s, bar_end_s) for the currently selected downbeat."""
    if not beat_times or beat_idx >= len(beat_times):
        return 0.0, 4.0
    bar_start = float(beat_times[beat_idx])
    end_idx   = beat_idx + beats_per_bar
    bar_end   = float(beat_times[end_idx]) if end_idx < len(beat_times) else bar_start + 4.0
    return bar_start, bar_end


# ── state helpers ─────────────────────────────────────────────────────────────

def _advance(state: dict) -> tuple[dict, str | None]:
    """
    Pop the next existing file from state['files'].
    Returns (new_state, filepath) or (new_state, None) if the queue is empty.
    """
    files = list(state['files'])
    while files:
        nf = files.pop(0)
        if Path(nf).exists():
            s = dict(state)
            s.update(
                files=files, current_file=nf,
                beat_idx=0, beats_per_bar=4, playing=False,
                beat_times=[], token='', bar_start=0.0, bar_end=4.0,
            )
            return s, nf
    s = dict(state)
    s.update(files=[], current_file=None, playing=False)
    return s, None


def _load_next(state: dict, max_skip: int = 20) -> tuple[dict, dict]:
    """
    Advance to the next file, load audio, run beat detection.
    Silently skips up to max_skip files that fail to load or yield no beats.
    Returns (new_state, waveform_data_dict).
    """
    for _ in range(max_skip + 1):
        state, filepath = _advance(state)
        if filepath is None:
            print('\n[annotate] All files done!')
            return state, {'times': [], 'values': []}

        print(f'\n[annotate] {Path(filepath).name}  ({len(state["files"])} remaining)')

        try:
            audio = _load_audio(filepath)
        except Exception as exc:
            print(f'  Load error: {exc}, skipping…')
            continue

        beat_times = _detect_beats(audio)
        if not beat_times:
            print('  No beats detected, skipping…')
            continue

        print(f'  {len(beat_times)} beats detected')
        wf_times, wf_values = _compute_waveform(audio)
        token                = _make_token(filepath)
        bar_start, bar_end   = _bar_extents(beat_times, 0, 4)

        state = dict(state)
        state.update(
            beat_times=beat_times, token=token,
            bar_start=bar_start, bar_end=bar_end,
        )
        return state, {'times': wf_times, 'values': wf_values}

    print('  Too many skips in a row — stopping.')
    state['current_file'] = None
    return state, {'times': [], 'values': []}


# ── figure builder ────────────────────────────────────────────────────────────

def _build_figure(state: dict, waveform: dict) -> go.Figure:
    """Build a Plotly waveform figure with beat markers and bar highlight."""
    times         = waveform.get('times', [])
    values        = waveform.get('values', [])
    beat_times    = state.get('beat_times', [])
    beat_idx      = state.get('beat_idx', 0)
    beats_per_bar = state.get('beats_per_bar', 4)
    current_file  = state.get('current_file', '')

    fig = go.Figure()
    if times:
        fig.add_trace(go.Scatter(
            x=times, y=values, mode='lines',
            line=dict(color='#4a9eff', width=1), name='waveform',
        ))

    shapes = []

    # Green rect highlighting the current bar (drawn below the waveform)
    bar_start, bar_end = _bar_extents(beat_times, beat_idx, beats_per_bar)
    shapes.append(dict(
        type='rect', x0=bar_start, x1=bar_end, y0=0, y1=1,
        xref='x', yref='paper',
        fillcolor='rgba(0,200,80,0.12)', line=dict(width=0), layer='below',
    ))

    # Beat markers — colour-coded by role
    for i, bt in enumerate(beat_times):
        if i == beat_idx:
            color, width = 'rgba(255,80,80,0.95)', 2      # selected downbeat: red
        elif i >= beat_idx and (i - beat_idx) % beats_per_bar == 0:
            color, width = 'rgba(255,220,80,0.70)', 1     # other bar starts: yellow
        else:
            color, width = 'rgba(255,255,255,0.18)', 1    # beat: faint white
        shapes.append(dict(
            type='line', x0=bt, x1=bt, y0=0, y1=1,
            xref='x', yref='paper',
            line=dict(color=color, width=width),
        ))

    fname = Path(current_file).name if current_file else '—'
    fig.update_layout(
        shapes=shapes,
        title=dict(
            text=(f'{fname}   beat_idx={beat_idx}   '
                  f'beats_per_bar={beats_per_bar}   {len(beat_times)} beats'),
            font=dict(color='#eee', size=13),
        ),
        paper_bgcolor='#1a1a2e', plot_bgcolor='#12122a',
        font=dict(color='#eee'),
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(title='seconds', color='#aaa', gridcolor='#2a2a4a'),
        yaxis=dict(showticklabels=False, gridcolor='#2a2a4a'),
        showlegend=False,
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────

def build_app(args):
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Discover which stems are already done so we can skip them
    done_stems: set[str] = set()
    for f in out_path.glob('*.ogg'):
        parts = f.stem.rsplit('__', 1)
        if len(parts) == 2:
            done_stems.add(parts[0])

    # Collect audio files, deduplicate, shuffle, filter already-done
    music_path = Path(args.music_path)
    all_files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        all_files.extend(music_path.rglob(f'*{ext}'))
        all_files.extend(music_path.rglob(f'*{ext.upper()}'))
    all_files = list({f.resolve() for f in all_files})
    random.shuffle(all_files)
    files = [str(f) for f in all_files if f.stem not in done_stems]

    print(f'Found {len(all_files)} audio files; {len(files)} left to annotate')
    if not files:
        print('Nothing left to annotate — all files already processed.')
        return None

    # Flask server wraps the Dash app so we can add a /audio/<token> route
    server = flask.Flask(__name__)
    app    = dash.Dash(__name__, server=server)

    @server.route('/audio/<token>')
    def serve_audio(token):
        fp = _token_to_file.get(token)
        if not fp:
            flask.abort(404)
        mime = mimetypes.guess_type(fp)[0] or 'audio/mpeg'
        return flask.send_file(fp, mimetype=mime, conditional=True)

    # ── layout ────────────────────────────────────────────────────────────────

    initial_state = dict(
        files=files, current_file=None,
        beat_times=[], beat_idx=0, beats_per_bar=4,
        playing=False, token='',
        bar_start=0.0, bar_end=4.0,
    )

    app.layout = html.Div([
        # Data stores
        dcc.Store(id='state',      data=initial_state),
        dcc.Store(id='waveform',   data={'times': [], 'values': []}),
        dcc.Store(id='keypress',   data=None),
        dcc.Store(id='_kb-ready',  data=False),   # dummy: keyboard setup output
        dcc.Store(id='_audio-ctrl', data=None),   # dummy: audio control output

        # Hidden audio player
        html.Audio(id='audio-el', style={'display': 'none'}),

        # Main display
        dcc.Graph(id='graph', config={'displayModeBar': False},
                  style={'height': '320px'}),

        html.Div(id='status', style={
            'fontFamily': 'monospace', 'padding': '8px 12px',
            'fontSize': '13px', 'color': '#ccc',
        }),

        html.Pre(
            'Keys:  h / l = move downbeat   1-9 = beats per bar'
            '   Space = toggle loop   Enter = save + next   Esc = skip',
            style={'color': '#555', 'fontSize': '11px',
                   'padding': '4px 12px', 'margin': 0},
        ),

        # 50 ms keyboard poll interval (runs indefinitely, clientside only)
        dcc.Interval(id='key-poll', interval=50),
        # One-shot interval: triggers first file load 300 ms after page renders
        dcc.Interval(id='init-iv', interval=300, max_intervals=1),
    ], style={
        'backgroundColor': '#1a1a2e', 'color': '#eee',
        'minHeight': '100vh', 'fontFamily': 'sans-serif',
    })

    # ── clientside: attach keyboard listener exactly once ─────────────────────

    app.clientside_callback(
        """
        function(n) {
            if (window._kbReady) return true;
            window._kbReady   = true;
            window._pendingKey = null;
            var H = new Set(['h','l',' ','Enter','Escape',
                             '1','2','3','4','5','6','7','8','9']);
            document.addEventListener('keydown', function(e) {
                if (H.has(e.key)) {
                    e.preventDefault();
                    window._pendingKey = {key: e.key, t: Date.now()};
                }
            });
            return true;
        }
        """,
        Output('_kb-ready', 'data'),
        Input('init-iv', 'n_intervals'),
    )

    # ── clientside: poll window._pendingKey every 50 ms → keypress store ──────

    app.clientside_callback(
        """
        function(n) {
            if (window._pendingKey) {
                var k = window._pendingKey;
                window._pendingKey = null;
                return k;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('keypress', 'data'),
        Input('key-poll', 'n_intervals'),
    )

    # ── clientside: audio control (src swap + play/pause/loop) ───────────────

    app.clientside_callback(
        """
        function(state) {
            var audio = document.getElementById('audio-el');
            if (!audio || !state) return null;

            // Swap src when file changes (token is a per-file nonce)
            var tok = state.token || '';
            if (audio.getAttribute('data-token') !== tok) {
                audio.setAttribute('data-token', tok);
                audio.src = tok ? ('/audio/' + tok) : '';
                audio.load();
                audio.pause();
                // Remove stale loop handler
                if (audio._loopH) {
                    audio.removeEventListener('timeupdate', audio._loopH);
                    audio._loopH = null;
                }
            }

            // Install loop handler (idempotent)
            if (!audio._loopH) {
                audio._loopH = function() {
                    if (audio._loopEnd != null && audio.currentTime >= audio._loopEnd) {
                        audio.currentTime = audio._loopStart || 0;
                    }
                };
                audio.addEventListener('timeupdate', audio._loopH);
            }

            audio._loopStart = state.bar_start || 0;
            audio._loopEnd   = state.bar_end   || 4;

            if (state.playing) {
                audio.currentTime = audio._loopStart;
                audio.play().catch(function(){});
            } else {
                audio.pause();
            }
            return null;
        }
        """,
        Output('_audio-ctrl', 'data'),
        Input('state', 'data'),
    )

    # ── server: load first file on startup ────────────────────────────────────

    @app.callback(
        Output('state',    'data', allow_duplicate=True),
        Output('waveform', 'data', allow_duplicate=True),
        Input('init-iv', 'n_intervals'),
        State('state', 'data'),
        prevent_initial_call=True,
    )
    def on_init(_n, state):
        return _load_next(state)

    # ── server: handle key presses ────────────────────────────────────────────

    @app.callback(
        Output('state',    'data'),
        Output('waveform', 'data'),
        Input('keypress',  'data'),
        State('state',     'data'),
        State('waveform',  'data'),
        prevent_initial_call=True,
    )
    def on_keypress(keypress, state, waveform):
        if keypress is None or not state.get('current_file'):
            return dash.no_update, dash.no_update

        key           = keypress.get('key', '')
        state         = dict(state)
        beat_times    = state.get('beat_times', [])
        beat_idx      = state.get('beat_idx', 0)
        beats_per_bar = state.get('beats_per_bar', 4)

        if key == 'h':
            beat_idx = max(0, beat_idx - 1)
            state['beat_idx'] = beat_idx

        elif key == 'l':
            beat_idx = min(len(beat_times) - 1, beat_idx + 1)
            state['beat_idx'] = beat_idx

        elif key in '123456789':
            beats_per_bar = int(key)
            state['beats_per_bar'] = beats_per_bar

        elif key == ' ':
            state['playing'] = not state.get('playing', False)

        elif key == 'Enter':
            cf = state.get('current_file')
            if cf and cf in _audio_cache:
                bar_starts = compute_bar_starts(beat_times, beat_idx, beats_per_bar)
                stem = Path(cf).stem
                n = process_audio(
                    _audio_cache[cf], bar_starts, stem,
                    str(out_path), args.chunks_per_song, args.pitch_range,
                    overwrite=False,
                )
                print(f'  Saved {n} chunks for {stem}')
            return _load_next(state)

        elif key == 'Escape':
            print(f'  Skipped {Path(state.get("current_file", "")).name}')
            return _load_next(state)

        else:
            return dash.no_update, dash.no_update

        # Update bar extents so the audio loop region stays in sync
        bar_start, bar_end = _bar_extents(beat_times, beat_idx, beats_per_bar)
        state['bar_start'] = bar_start
        state['bar_end']   = bar_end
        return state, waveform

    # ── server: redraw figure + status line ───────────────────────────────────

    @app.callback(
        Output('graph',   'figure'),
        Output('status',  'children'),
        Input('state',    'data'),
        Input('waveform', 'data'),
    )
    def update_display(state, waveform):
        if not state.get('current_file'):
            fig = go.Figure()
            fig.update_layout(
                title='All done — you can close this window.',
                paper_bgcolor='#1a1a2e', plot_bgcolor='#12122a',
                font=dict(color='#eee'),
            )
            return fig, 'No more files to annotate.'

        fig           = _build_figure(state, waveform)
        beat_times    = state.get('beat_times', [])
        beat_idx      = state.get('beat_idx', 0)
        beats_per_bar = state.get('beats_per_bar', 4)
        playing       = state.get('playing', False)
        n_remaining   = len(state.get('files', []))
        n_bars        = max(0, (len(beat_times) - beat_idx + beats_per_bar - 1) // beats_per_bar)

        status = (
            f'File: {Path(state["current_file"]).name}  |  '
            f'beats: {len(beat_times)}  ~{n_bars} bars  |  '
            f'remaining: {n_remaining}  |  '
            f'{"▶ PLAYING" if playing else "⏸ paused"}'
        )
        return fig, status

    return app


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = _make_parser().parse_args()
    app  = build_app(args)
    if app is None:
        return
    print(f'Starting annotator → http://{args.host}:{args.port}')
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
