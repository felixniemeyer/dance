#!/usr/bin/env python3
"""
annotate.py — interactive manual bar annotation tool for the dance project.

Opens a Dash browser tab. For each audio file in --music-path:
  - Loads audio, fits a constant-tempo beat grid (like Mixxx's BPM engine)
  - Shows waveform + beat markers in a Plotly figure
  - h/l update the graph INSTANTLY (pure clientside, no server round-trip)
  - Enter saves chunks; Esc skips the file

Keyboard controls:
  h / l     Shift beat onset offset ±1 beat (fine-tune the grid phase)
  j / k     Navigate to next / previous bar
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
WAVEFORM_POINTS  = 3000
HOP_LENGTH       = 512    # for onset / beat analysis


# ── module-level shared state ─────────────────────────────────────────────────

_audio_cache:   dict[str, np.ndarray] = {}
_token_to_file: dict[str, str]        = {}


# ── audio helpers ─────────────────────────────────────────────────────────────

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


def _compute_waveform(audio: np.ndarray) -> tuple[list, list]:
    step  = max(1, len(audio) // WAVEFORM_POINTS)
    n     = (len(audio) // step) * step
    env   = np.abs(audio[:n]).reshape(-1, step).max(axis=1)
    times = (np.arange(len(env)) * step / CHUNK_SR).tolist()
    return times, env.tolist()


def _detect_beats(audio: np.ndarray) -> tuple[float, list[float]]:
    """
    Fit a constant-tempo beat grid to the audio, aligned to onset peaks.

    Strategy (same idea as Mixxx's BPM engine):
      1. Estimate a single global BPM via the aggregate tempogram.
      2. Grid-search over all phase offsets (0 … beat_period) to find the
         shift that maximises the total onset energy under the beat grid.
      3. Return the resulting evenly-spaced beat times.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=CHUNK_SR, hop_length=HOP_LENGTH)

    # Single global tempo — intentionally ignores tempo changes
    tempo = float(
        np.atleast_1d(
            librosa.beat.tempo(
                onset_envelope=onset_env, sr=CHUNK_SR, hop_length=HOP_LENGTH)
        )[0]
    )

    fps                = CHUNK_SR / HOP_LENGTH          # onset frames per second
    beat_period_frames = fps * 60.0 / tempo
    n_frames           = len(onset_env)
    n_phase            = max(1, int(np.round(beat_period_frames)))

    # Find the phase offset that best aligns with the music's onsets
    best_score, best_phase = -1.0, 0
    for ph in range(n_phase):
        idxs  = np.round(np.arange(ph, n_frames, beat_period_frames)).astype(int)
        idxs  = idxs[idxs < n_frames]
        score = float(onset_env[idxs].sum())
        if score > best_score:
            best_score, best_phase = score, ph

    # Produce the regular grid
    frames     = np.round(np.arange(best_phase, n_frames, beat_period_frames)).astype(int)
    frames     = frames[frames < n_frames]
    beat_times = librosa.frames_to_time(frames, sr=CHUNK_SR, hop_length=HOP_LENGTH)
    return tempo, beat_times.tolist()


# ── state helpers ─────────────────────────────────────────────────────────────

def compute_bar_starts(beat_times: list, beat_idx: int, beats_per_bar: int) -> list[float]:
    return list(beat_times[beat_idx::beats_per_bar])


def _advance(state: dict) -> tuple[dict, str | None]:
    files = list(state['files'])
    while files:
        nf = files.pop(0)
        if Path(nf).exists():
            s = dict(state)
            s.update(files=files, current_file=nf,
                     beat_idx=0, beats_per_bar=4, playing=False,
                     beat_times=[], tempo=120.0, token='')
            return s, nf
    s = dict(state)
    s.update(files=[], current_file=None, playing=False)
    return s, None


def _load_next(state: dict, max_skip: int = 20) -> tuple[dict, dict]:
    """Advance to next file, load audio, fit beat grid. Skips bad files."""
    for _ in range(max_skip + 1):
        state, filepath = _advance(state)
        if filepath is None:
            print('\n[annotate] All files done!')
            return state, {'times': [], 'values': []}

        name = Path(filepath).name
        print(f'\n[annotate] {name}  ({len(state["files"])} remaining)')

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
        wf_times, wf_values = _compute_waveform(audio)
        token = _make_token(filepath)

        state = dict(state)
        state.update(beat_times=beat_times, tempo=tempo, token=token)
        return state, {'times': wf_times, 'values': wf_values}

    print('  Too many skips in a row — stopping.')
    state['current_file'] = None
    return state, {'times': [], 'values': []}


# ── Dash app ──────────────────────────────────────────────────────────────────

# ── Inline JavaScript ─────────────────────────────────────────────────────────
#
# All graph/audio manipulation for h/l/1-9/Space runs entirely in the browser.
# The server is only involved when loading a new file (init or Enter/Esc).
#
# window.ann holds all JS-side state and helper functions.

_JS_SETUP = """
function(n) {
    /* Run once: define window.ann namespace + attach keyboard listener. */
    if (window.ann && window.ann._ready) return window.dash_clientside.no_update;

    window.ann = window.ann || {};
    Object.assign(window.ann, {
        _ready:    false,
        beatTimes: [],
        beatIdx:   0,
        bpb:       4,
        tempo:     120,
        filename:  '',
        playing:   false,
        loopStart: 0,
        loopEnd:   4,
        pendingAction: null,

        barExtents: function() {
            var bt = window.ann.beatTimes, bi = window.ann.beatIdx, bpb = window.ann.bpb;
            if (!bt.length || bi >= bt.length) return [0, 4];
            var s  = bt[bi];
            var ei = bi + bpb;
            return [s, ei < bt.length ? bt[ei] : s + 4];
        },

        buildShapes: function() {
            var bt = window.ann.beatTimes, bi = window.ann.beatIdx, bpb = window.ann.bpb;
            var ext = window.ann.barExtents();
            var shapes = [{
                type: 'rect', x0: ext[0], x1: ext[1], y0: 0, y1: 1,
                xref: 'x', yref: 'paper',
                fillcolor: 'rgba(0,200,80,0.12)',
                line: {width: 0}, layer: 'below'
            }];
            for (var i = 0; i < bt.length; i++) {
                var color, w;
                if      (i === bi)                         { color = 'rgba(255,80,80,0.95)';   w = 2; }
                else if (i >= bi && (i - bi) % bpb === 0) { color = 'rgba(255,220,80,0.70)';  w = 1; }
                else                                       { color = 'rgba(255,255,255,0.18)'; w = 1; }
                shapes.push({
                    type: 'line', x0: bt[i], x1: bt[i], y0: 0, y1: 1,
                    xref: 'x', yref: 'paper',
                    line: {color: color, width: w}
                });
            }
            return shapes;
        },

        buildTitle: function() {
            var a = window.ann;
            return a.filename + '   ' + a.tempo.toFixed(1) + ' BPM'
                 + '   beat_idx=' + a.beatIdx + '   bpb=' + a.bpb;
        },

        getGraphDiv: function() {
            var el = document.getElementById('graph');
            if (!el) return null;
            if (el._fullLayout) return el;
            return el.querySelector('.js-plotly-plot') || el;
        },

        /* Update shapes + title, optionally scroll view to keep current bar visible. */
        update: function(scroll) {
            var a  = window.ann, bt = a.beatTimes, bi = a.beatIdx, bpb = a.bpb;
            var gd = a.getGraphDiv();
            if (!gd || !window.Plotly) return;
            var upd = {
                shapes:       a.buildShapes(),
                'title.text': a.buildTitle()
            };
            if (scroll && bt.length && bi < bt.length) {
                var beatDur  = 60.0 / a.tempo;
                var barDur   = beatDur * bpb;
                var barStart = bt[bi];
                var viewDur  = barDur * 16;
                var vStart   = Math.max(0, barStart - barDur);
                upd['xaxis.range']     = [vStart, vStart + viewDur];
                upd['xaxis.autorange'] = false;
            }
            window.Plotly.relayout(gd, upd);
        },

        updateLive: function() {
            var a  = window.ann;
            var bt = a.beatTimes, bi = a.beatIdx, bpb = a.bpb;
            var nB = bt.length > bi ? Math.ceil((bt.length - bi) / bpb) : 0;
            var ext = a.barExtents();
            var el = document.getElementById('status-live');
            if (el) {
                el.textContent = 'beat_idx=' + bi
                    + '   bpb=' + bpb
                    + '   ~' + nB + ' bars'
                    + '   bar_start=' + ext[0].toFixed(2) + 's'
                    + '   ' + (a.playing ? '\u25B6 PLAYING' : '\u23F8 paused');
            }
        },

        updateAudioLoop: function() {
            var a   = window.ann;
            var ext = a.barExtents();
            a.loopStart = ext[0];
            a.loopEnd   = ext[1];
            var audio = document.getElementById('audio-el');
            if (audio && a.playing) {
                audio.currentTime = a.loopStart;
            }
        },

        setAudio: function(tok) {
            var audio = document.getElementById('audio-el');
            if (!audio) return;
            if (audio.getAttribute('data-token') === tok) return;
            audio.setAttribute('data-token', tok);
            audio.src = tok ? ('/audio/' + tok) : '';
            audio.load();
            audio.pause();
            window.ann.playing = false;
            if (!audio._loopH) {
                audio._loopH = function() {
                    var a = window.ann;
                    if (a.playing && audio.currentTime >= a.loopEnd) {
                        audio.currentTime = a.loopStart;
                    }
                };
                audio.addEventListener('timeupdate', audio._loopH);
            }
        }
    });

    /* Keyboard handler — all graph/audio updates happen here, no server round-trip */
    document.addEventListener('keydown', function(e) {
        var a  = window.ann;
        var bt = a.beatTimes;
        if (!bt) return;

        if (e.key === 'h') {
            /* Shift onset offset: move downbeat one beat earlier */
            e.preventDefault();
            a.beatIdx = Math.max(0, a.beatIdx - 1);
            a.update(true); a.updateLive(); a.updateAudioLoop();

        } else if (e.key === 'l') {
            /* Shift onset offset: move downbeat one beat later */
            e.preventDefault();
            a.beatIdx = Math.min(bt.length - 1, a.beatIdx + 1);
            a.update(true); a.updateLive(); a.updateAudioLoop();

        } else if (e.key === 'j') {
            /* Navigate forward one bar */
            e.preventDefault();
            a.beatIdx = Math.min(bt.length - 1, a.beatIdx + a.bpb);
            a.update(true); a.updateLive(); a.updateAudioLoop();

        } else if (e.key === 'k') {
            /* Navigate backward one bar */
            e.preventDefault();
            a.beatIdx = Math.max(0, a.beatIdx - a.bpb);
            a.update(true); a.updateLive(); a.updateAudioLoop();

        } else if ('123456789'.indexOf(e.key) >= 0) {
            e.preventDefault();
            a.bpb = parseInt(e.key);
            a.update(false); a.updateLive(); a.updateAudioLoop();

        } else if (e.key === ' ') {
            e.preventDefault();
            a.playing = !a.playing;
            var audio = document.getElementById('audio-el');
            if (audio) {
                if (a.playing) {
                    a.updateAudioLoop();
                    audio.currentTime = a.loopStart;
                    audio.play().catch(function(){});
                } else {
                    audio.pause();
                }
            }
            a.updateLive();

        } else if (e.key === 'Enter') {
            e.preventDefault();
            a.pendingAction = {type: 'save', beat_idx: a.beatIdx, bpb: a.bpb, t: Date.now()};

        } else if (e.key === 'Escape') {
            e.preventDefault();
            a.pendingAction = {type: 'skip', t: Date.now()};
        }
    });

    window.ann._ready = true;
    return true;
}
"""

_JS_FILE_CHANGE = """
function(state, waveform) {
    /* Rebuild the full Plotly figure whenever a new file is loaded. */
    var a = window.ann;
    if (!a || !a._ready) return window.dash_clientside.no_update;
    if (!state || !state.current_file) {
        return {
            data: [],
            layout: {
                title: {text: 'All done \u2014 you can close this window.',
                        font: {color: '#eee'}},
                paper_bgcolor: '#1a1a2e', plot_bgcolor: '#12122a',
                font: {color: '#eee'}, margin: {l:50,r:20,t:60,b:40}
            }
        };
    }

    /* Sync JS state from Dash state */
    a.beatTimes = state.beat_times || [];
    a.beatIdx   = 0;
    a.bpb       = state.beats_per_bar || 4;
    a.tempo     = state.tempo || 120;
    a.playing   = false;
    var parts   = (state.current_file || '').split('/');
    a.filename  = parts[parts.length - 1];
    var ext     = a.barExtents();
    a.loopStart = ext[0];
    a.loopEnd   = ext[1];

    /* Connect audio element to new file */
    a.setAudio(state.token || '');

    /* Update live status */
    a.updateLive && a.updateLive();

    /* Build Plotly figure */
    var traces = [];
    var wf = waveform || {};
    if (wf.times && wf.times.length) {
        traces.push({
            x: wf.times, y: wf.values,
            mode: 'lines', type: 'scatter',
            line: {color: '#4a9eff', width: 1},
            name: 'waveform'
        });
    }

    /* Initial view: show first ~16 bars, rangeslider shows full song */
    var beatDur   = 60.0 / a.tempo;
    var barDur    = beatDur * a.bpb;
    var initRange = [0, barDur * 16];

    return {
        data: traces,
        layout: {
            shapes: a.buildShapes(),
            title:  {text: a.buildTitle(), font: {color: '#eee', size: 13}},
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor:  '#12122a',
            font:   {color: '#eee'},
            margin: {l: 50, r: 20, t: 50, b: 40},
            xaxis:  {
                title: {text: 'seconds'},
                color: '#aaa',
                gridcolor: '#2a2a4a',
                range: initRange,
                autorange: false,
                rangeslider: {
                    visible: true,
                    thickness: 0.07,
                    bgcolor: '#12122a',
                    bordercolor: '#2a2a4a',
                    borderwidth: 1
                }
            },
            yaxis:  {showticklabels: false, gridcolor: '#2a2a4a'},
            showlegend: false
        }
    };
}
"""

_JS_ACTION_POLL = """
function(n) {
    /* Pick up a pending Enter/Esc action and forward it to the server. */
    var a = window.ann;
    if (a && a.pendingAction) {
        var act = a.pendingAction;
        a.pendingAction = null;
        return act;
    }
    return window.dash_clientside.no_update;
}
"""


# ── App builder ───────────────────────────────────────────────────────────────

def build_app(args):
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Skip already-processed stems
    done_stems: set[str] = set()
    for f in out_path.glob('*.ogg'):
        parts = f.stem.rsplit('__', 1)
        if len(parts) == 2:
            done_stems.add(parts[0])

    # Collect audio files
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

    server = flask.Flask(__name__)
    # update_title=None stops the browser tab from flickering "Updating…"
    app    = dash.Dash(__name__, server=server, update_title=None)

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
        tempo=120.0, playing=False, token='',
    )

    app.layout = html.Div([
        dcc.Store(id='state',    data=initial_state),
        dcc.Store(id='waveform', data={'times': [], 'values': []}),
        dcc.Store(id='action',   data=None),
        dcc.Store(id='_kb-ready', data=False),

        html.Audio(id='audio-el', style={'display': 'none'}),

        dcc.Graph(id='graph', config={'displayModeBar': False},
                  figure={'data': [], 'layout': {
                      'paper_bgcolor': '#1a1a2e', 'plot_bgcolor': '#12122a',
                      'font': {'color': '#eee'}, 'margin': {'l':50,'r':20,'t':50,'b':40},
                  }},
                  style={'height': '320px'}),

        # Server-updated: file path + remaining count
        html.Div(id='status-file', style={
            'fontFamily': 'monospace', 'padding': '4px 12px',
            'fontSize': '12px', 'color': '#888',
        }),
        # JS-updated: beat_idx, bpb, playing state (instant, no server round-trip)
        html.Div(id='status-live', style={
            'fontFamily': 'monospace', 'padding': '4px 12px',
            'fontSize': '13px', 'color': '#ccc',
        }),

        html.Pre(
            'Keys:  h / l = shift beat onset ±1   j / k = next / prev bar'
            '   1-9 = beats per bar   Space = loop   Enter = save + next   Esc = skip',
            style={'color': '#555', 'fontSize': '11px',
                   'padding': '4px 12px', 'margin': 0},
        ),

        # One-shot: triggers first file load + JS setup
        dcc.Interval(id='init-iv', interval=300, max_intervals=1),
        # Polls window.ann.pendingAction (Enter/Esc) every 100 ms
        dcc.Interval(id='act-poll', interval=100),
    ], style={
        'backgroundColor': '#1a1a2e', 'color': '#eee',
        'minHeight': '100vh', 'fontFamily': 'sans-serif',
    })

    # ── clientside: one-time JS setup ─────────────────────────────────────────

    app.clientside_callback(
        _JS_SETUP,
        Output('_kb-ready', 'data'),
        Input('init-iv', 'n_intervals'),
    )

    # ── clientside: rebuild figure on file change ──────────────────────────────

    app.clientside_callback(
        _JS_FILE_CHANGE,
        Output('graph', 'figure'),
        Input('state', 'data'),
        Input('waveform', 'data'),
    )

    # ── clientside: action poll (Enter / Esc → server) ────────────────────────

    app.clientside_callback(
        _JS_ACTION_POLL,
        Output('action', 'data'),
        Input('act-poll', 'n_intervals'),
    )

    # ── server: load first file on startup ────────────────────────────────────

    @app.callback(
        Output('state',    'data'),
        Output('waveform', 'data'),
        Input('init-iv', 'n_intervals'),
        State('state', 'data'),
        prevent_initial_call=True,
    )
    def on_init(_n, state):
        return _load_next(state)

    # ── server: handle Enter / Esc ────────────────────────────────────────────

    @app.callback(
        Output('state',    'data', allow_duplicate=True),
        Output('waveform', 'data', allow_duplicate=True),
        Input('action', 'data'),
        State('state',  'data'),
        prevent_initial_call=True,
    )
    def on_action(action, state):
        if not action or not state.get('current_file'):
            return dash.no_update, dash.no_update

        if action['type'] == 'save':
            cf = state.get('current_file')
            if cf and cf in _audio_cache:
                beat_times = state.get('beat_times', [])
                beat_idx   = action.get('beat_idx', 0)
                bpb        = action.get('bpb', 4)
                bar_starts = compute_bar_starts(beat_times, beat_idx, bpb)
                stem = Path(cf).stem
                n    = process_audio(
                    _audio_cache[cf], bar_starts, stem,
                    str(out_path), args.chunks_per_song, args.pitch_range,
                    overwrite=False,
                )
                print(f'  Saved {n} chunks for {stem}')

        elif action['type'] == 'skip':
            print(f'  Skipped {Path(state.get("current_file", "")).name}')

        return _load_next(state)

    # ── server: update file-level status line ────────────────────────────────

    @app.callback(
        Output('status-file', 'children'),
        Input('state', 'data'),
    )
    def update_status(state):
        cf = state.get('current_file')
        if not cf:
            return 'No more files to annotate.'
        n_remaining = len(state.get('files', []))
        tempo       = state.get('tempo', 0.0)
        n_beats     = len(state.get('beat_times', []))
        return (f'{cf}   |   {tempo:.1f} BPM   {n_beats} beats'
                f'   |   {n_remaining} files remaining')

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
