"""
Inspector web UI

Source A — Chunks (labeled):  waveform + phase label + optional prediction
Source B — Real-world:        waveform + prediction only (no label)

Usage:
    venv/bin/python inspector.py \\
        --chunks-path data/chunks/<run> \\
        --test-data-path realworld-test-data \\
        --checkpoints-path checkpoints
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import soundfile
import torch
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import send_file

import config
from models.selector import loadModel

# ── CLI args ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--chunks-path', type=str, required=True)
parser.add_argument('--test-data-path', type=str, default='realworld-test-data')
parser.add_argument('--checkpoints-path', type=str, default='checkpoints')
parser.add_argument('--port', type=int, default=8050)
args = parser.parse_args()

# ── Discovery ──────────────────────────────────────────────────────────────────

AUDIO_EXTS = {'.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac'}

def find_chunks(path):
    if not os.path.isdir(path):
        print(f'[inspector] chunks path not found: {path!r}')
        return []
    all_files = sorted(os.listdir(path))
    ogg_files = [f for f in all_files if f.endswith('.ogg')]
    names = []
    for f in ogg_files:
        stem = f[:-4]
        if os.path.exists(os.path.join(path, stem + '.phase')):
            names.append(stem)
    if not names:
        print(f'[inspector] no .ogg+.phase pairs in {path!r} '
              f'({len(ogg_files)} .ogg, {len(all_files)} total files)')
    return names

def find_test_files(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        f for f in os.listdir(path)
        if Path(f).suffix.lower() in AUDIO_EXTS
    )

def find_checkpoints(path):
    pts = sorted(Path(path).rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(path)) for p in pts]

chunk_names       = find_chunks(args.chunks_path)
test_files        = find_test_files(args.test_data_path)
checkpoint_options = find_checkpoints(args.checkpoints_path)

if not chunk_names and not test_files:
    raise SystemExit('No chunks and no test files found — nothing to inspect.')

# ── Audio loading ──────────────────────────────────────────────────────────────

MAX_WAVEFORM_PTS = 8000

def load_audio_ffmpeg(path, target_sr=config.samplerate):
    """Decode any audio format to mono float32 at target_sr via ffmpeg."""
    cmd = [
        'ffmpeg', '-v', 'warning',
        '-i', path,
        '-ac', '1',
        '-ar', str(target_sr),
        '-f', 'f32le',
        '-',
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed: {result.stderr.decode()}')
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return audio, target_sr

def load_audio(path):
    """Load audio from any format; returns (audio_np_f32_mono, sr)."""
    ext = Path(path).suffix.lower()
    if ext in {'.ogg', '.wav', '.flac'}:
        audio, sr = soundfile.read(path, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    else:
        return load_audio_ffmpeg(path)

def _downsample_waveform(audio, sr):
    duration = len(audio) / sr
    times = np.linspace(0, duration, len(audio))
    if len(audio) > MAX_WAVEFORM_PTS:
        step = len(audio) // MAX_WAVEFORM_PTS
        times = times[::step]
        audio = audio[::step]
    return times, audio, duration

def _bar_times_from_phase(phase_times, phase):
    bar_times = []
    for i in range(1, len(phase)):
        if phase[i - 1] > 0.8 and phase[i] < 0.2:
            frac = (1.0 - phase[i - 1]) / (1.0 - phase[i - 1] + phase[i])
            bar_times.append(phase_times[i - 1] + (phase_times[i] - phase_times[i - 1]) * frac)
    return bar_times

def load_chunk(chunk_name):
    ogg_path   = os.path.join(args.chunks_path, chunk_name + '.ogg')
    phase_path = os.path.join(args.chunks_path, chunk_name + '.phase')

    audio, sr = soundfile.read(ogg_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    with open(phase_path) as f:
        phase = np.array([float(l) for l in f if l.strip()], dtype=np.float32)

    duration    = len(audio) / sr
    frame_dur   = duration / len(phase)
    phase_times = np.arange(len(phase)) * frame_dur + frame_dur / 2

    audio_times, audio_ds, _ = _downsample_waveform(audio, sr)
    bar_times = _bar_times_from_phase(phase_times, phase)

    return audio_times, audio_ds, phase_times, phase, bar_times, duration

def load_test_file(filename):
    path = os.path.join(args.test_data_path, filename)
    audio, sr = load_audio(path)
    audio_times, audio_ds, duration = _downsample_waveform(audio, sr)
    return audio_times, audio_ds, duration

# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(checkpoint_rel, audio_path, anticipation_s):
    ckpt_path = os.path.join(args.checkpoints_path, checkpoint_rel)
    audio, sr = load_audio(audio_path)

    audio_t  = torch.from_numpy(audio)
    length   = len(audio_t)
    frames_n = length // config.frame_size
    offset   = length % config.frame_size
    seq = audio_t[offset:offset + frames_n * config.frame_size].reshape(1, frames_n, config.frame_size)

    model, _ = loadModel(ckpt_path)
    model.eval()
    ant = torch.tensor([anticipation_s], dtype=torch.float32)

    with torch.no_grad():
        try:
            out, _ = model(seq, anticipation=ant)
        except TypeError:
            out, _ = model(seq)

    if out.shape[-1] == 2:
        angles = torch.atan2(out[0, :, 0], out[0, :, 1])
        phases = torch.remainder(angles / (2 * torch.pi), 1.0).numpy()
    else:
        phases = out[0, :, 0].numpy()

    duration  = length / sr
    frame_dur = duration / frames_n
    times = np.arange(frames_n) * frame_dur + frame_dur / 2
    return times, phases

# ── Figure builders ────────────────────────────────────────────────────────────

DARK = dict(
    plot_bgcolor='#1a1a2e',
    paper_bgcolor='#16213e',
    font=dict(color='#eee'),
)

def _base_fig(title, audio_times, audio):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.06,
        subplot_titles=['Waveform', 'Bar Phase'],
    )
    fig.add_trace(
        go.Scatter(x=audio_times, y=audio, mode='lines', name='waveform',
                   line=dict(color='steelblue', width=0.6), hoverinfo='skip'),
        row=1, col=1,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=580,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation='h', y=1.07),
        dragmode='pan',
        hovermode='x unified',
        **DARK,
    )
    fig.update_xaxes(showgrid=True, gridcolor='#2a2a4a', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#2a2a4a', zeroline=False)
    fig.update_yaxes(title_text='amplitude', row=1, col=1)
    fig.update_yaxes(title_text='phase [0,1)', range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text='time (s)', row=2, col=1)
    return fig

def _pred_color(ant_ms):
    return '#40e0d0' if ant_ms > 0 else '#7fff7f'  # cyan if anticipation, green if 0ms

def _add_pred_trace(fig, times, phases, color, name):
    fig.add_trace(
        go.Scatter(x=times, y=phases, mode='lines', name=name,
                   line=dict(color=color, width=1.2)),
        row=2, col=1,
    )
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    bar_rgba = f'rgba({r},{g},{b},0.35)'
    for t in _bar_times_from_phase(times, phases):
        fig.add_shape(dict(
            type='line', xref='x', yref='paper',
            x0=t, x1=t, y0=0, y1=1,
            line=dict(color=bar_rgba, width=1),
        ))

def build_chunk_figure(chunk_name, pred_times=None, pred_phases=None,
                       ant_ms=0, ref_times=None, ref_phases=None):
    audio_times, audio, phase_times, phase, bar_times, duration = load_chunk(chunk_name)
    fig = _base_fig(chunk_name, audio_times, audio)

    fig.add_trace(
        go.Scatter(x=phase_times, y=phase, mode='lines', name='label',
                   line=dict(color='darkorange', width=1.4)),
        row=2, col=1,
    )

    if pred_times is not None and pred_phases is not None:
        label = f'prediction ({ant_ms:.0f} ms)' if ant_ms > 0 else 'prediction'
        _add_pred_trace(fig, pred_times, pred_phases, _pred_color(ant_ms), label)

    if ref_times is not None and ref_phases is not None:
        _add_pred_trace(fig, ref_times, ref_phases, '#7fff7f', 'prediction (0 ms ref)')

    for t in bar_times:
        fig.add_shape(dict(
            type='line', xref='x', yref='paper',
            x0=t, x1=t, y0=0, y1=1,
            line=dict(color='rgba(200,60,60,0.4)', width=1, dash='dot'),
        ))

    return fig

def build_test_figure(filename, pred_times=None, pred_phases=None,
                      ant_ms=0, ref_times=None, ref_phases=None):
    audio_times, audio, duration = load_test_file(filename)
    fig = _base_fig(filename, audio_times, audio)

    if pred_times is not None and pred_phases is not None:
        label = f'prediction ({ant_ms:.0f} ms)' if ant_ms > 0 else 'prediction'
        _add_pred_trace(fig, pred_times, pred_phases, _pred_color(ant_ms), label)
        if ref_times is not None and ref_phases is not None:
            _add_pred_trace(fig, ref_times, ref_phases, '#7fff7f', 'prediction (0 ms ref)')
    else:
        fig.add_annotation(
            text='Run inference to see phase prediction',
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color='#666'),
            row=2, col=1,
        )

    return fig

# ── Dash app ───────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = 'Dance Inspector'
server = app.server

@server.route('/chunk-audio/<path:filename>')
def serve_chunk_audio(filename):
    return send_file(os.path.join(args.chunks_path, filename), mimetype='audio/ogg')

@server.route('/test-audio/<path:filename>')
def serve_test_audio(filename):
    path = os.path.join(args.test_data_path, filename)
    ext = Path(filename).suffix.lower()
    mime = {'mp3': 'audio/mpeg', 'ogg': 'audio/ogg', 'wav': 'audio/wav',
            'flac': 'audio/flac', 'm4a': 'audio/mp4', 'aac': 'audio/aac'}.get(ext[1:], 'audio/mpeg')
    return send_file(path, mimetype=mime)

# styles
CTRL = {'display': 'flex', 'alignItems': 'center', 'gap': '12px',
        'flexWrap': 'wrap', 'padding': '10px 16px', 'background': '#0f3460'}
LBL  = {'color': '#ccc', 'fontSize': '13px', 'whiteSpace': 'nowrap'}
STAT = {'color': '#7fff7f', 'fontSize': '13px', 'minWidth': '220px'}
SEP  = html.Span('│', style={'color': '#444', 'fontSize': '20px', 'margin': '0 4px'})

initial_chunk = chunk_names[0] if chunk_names else None
initial_test  = test_files[0] if test_files else None

app.layout = html.Div([

    # ── source toggle + file pickers ─────────────────────────────────────────
    html.Div([
        html.Span('Source:', style=LBL),
        dcc.RadioItems(
            id='source-radio',
            options=[
                {'label': f' Chunks ({len(chunk_names)})', 'value': 'chunk'},
                {'label': f' Real-world ({len(test_files)})', 'value': 'test',
                 'disabled': len(test_files) == 0},
            ],
            value='chunk' if chunk_names else 'test',
            inline=True,
            labelStyle={'color': '#ccc', 'fontSize': '13px', 'marginRight': '12px'},
        ),
        SEP,
        # chunk picker
        html.Div([
            dcc.Dropdown(
                id='chunk-dd',
                options=[{'label': n, 'value': n} for n in chunk_names],
                value=initial_chunk,
                clearable=False,
                style={'width': '360px', 'color': '#111'},
            ),
        ], id='chunk-picker-wrap'),
        # test file picker
        html.Div([
            dcc.Dropdown(
                id='test-dd',
                options=[{'label': f, 'value': f} for f in test_files],
                value=initial_test,
                clearable=False,
                style={'width': '360px', 'color': '#111'},
            ),
        ], id='test-picker-wrap', style={'display': 'none'}),
        SEP,
        html.Span('Checkpoint:', style=LBL),
        dcc.Dropdown(
            id='ckpt-dd',
            options=[{'label': c, 'value': c} for c in checkpoint_options],
            value=checkpoint_options[0] if checkpoint_options else None,
            clearable=True,
            placeholder='(none)',
            style={'width': '300px', 'color': '#111'},
        ),
        html.Span('Anticipation (ms):', style=LBL),
        dcc.Input(
            id='ant-input',
            type='number', value=0, min=0, max=500, step=10,
            style={'width': '70px', 'padding': '4px', 'background': '#1a1a2e',
                   'color': '#eee', 'border': '1px solid #444', 'borderRadius': '4px'},
        ),
        html.Button('Run inference', id='run-btn', n_clicks=0, style={
            'background': '#1a6b3c', 'color': '#fff', 'border': 'none',
            'borderRadius': '4px', 'padding': '6px 14px', 'cursor': 'pointer',
            'fontSize': '13px',
        }),
        html.Span('', id='status-msg', style=STAT),
    ], style=CTRL),

    # ── audio player ─────────────────────────────────────────────────────────
    html.Div([
        html.Audio(id='audio-player', controls=True, autoPlay=False,
                   style={'width': '100%', 'margin': '0'}),
    ], style={'background': '#0d2137', 'padding': '6px 16px'}),

    # ── graph ─────────────────────────────────────────────────────────────────
    dcc.Graph(
        id='main-graph',
        figure=build_chunk_figure(initial_chunk) if initial_chunk else go.Figure(),
        config={'scrollZoom': True, 'displayModeBar': True},
        style={'height': '600px'},
    ),

    dcc.Store(id='pred-store', data=None),

], style={'background': '#16213e', 'minHeight': '100vh'})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output('chunk-picker-wrap', 'style'),
    Output('test-picker-wrap', 'style'),
    Input('source-radio', 'value'),
)
def toggle_pickers(source):
    show = {'display': 'block'}
    hide = {'display': 'none'}
    return (show, hide) if source == 'chunk' else (hide, show)


@app.callback(
    Output('audio-player', 'src'),
    Input('source-radio', 'value'),
    Input('chunk-dd', 'value'),
    Input('test-dd', 'value'),
)
def update_audio(source, chunk_name, test_file):
    if source == 'chunk' and chunk_name:
        return f'/chunk-audio/{chunk_name}.ogg'
    if source == 'test' and test_file:
        return f'/test-audio/{test_file}'
    return ''


@app.callback(
    Output('pred-store', 'data'),
    Output('status-msg', 'children'),
    Input('run-btn', 'n_clicks'),
    State('source-radio', 'value'),
    State('chunk-dd', 'value'),
    State('test-dd', 'value'),
    State('ckpt-dd', 'value'),
    State('ant-input', 'value'),
    prevent_initial_call=True,
)
def run_inference_cb(n_clicks, source, chunk_name, test_file, ckpt_rel, anticipation):
    if not ckpt_rel:
        return None, 'Select a checkpoint first.'
    try:
        ant_ms = float(anticipation or 0.0)
        ant_s  = ant_ms / 1000.0
        if source == 'chunk' and chunk_name:
            audio_path = os.path.join(args.chunks_path, chunk_name + '.ogg')
        elif source == 'test' and test_file:
            audio_path = os.path.join(args.test_data_path, test_file)
        else:
            return None, 'No file selected.'
        times, phases = run_inference(ckpt_rel, audio_path, ant_s)
        result = {'times': times.tolist(), 'phases': phases.tolist(), 'ant_ms': ant_ms}
        if ant_ms > 1:
            ref_times, ref_phases = run_inference(ckpt_rel, audio_path, 0.0)
            result['ref_times']  = ref_times.tolist()
            result['ref_phases'] = ref_phases.tolist()
        return result, f'Done — {ckpt_rel}'
    except Exception as e:
        return None, f'Error: {e}'


@app.callback(
    Output('pred-store', 'data', allow_duplicate=True),
    Input('chunk-dd', 'value'),
    Input('test-dd', 'value'),
    Input('source-radio', 'value'),
    prevent_initial_call=True,
)
def clear_pred_on_change(_a, _b, _c):
    # clear prediction whenever the selected file changes
    return None


@app.callback(
    Output('main-graph', 'figure'),
    Input('source-radio', 'value'),
    Input('chunk-dd', 'value'),
    Input('test-dd', 'value'),
    Input('pred-store', 'data'),
)
def update_graph(source, chunk_name, test_file, pred):
    pt     = np.array(pred['times'])              if pred else None
    pp     = np.array(pred['phases'])             if pred else None
    ant_ms = pred.get('ant_ms', 0)                if pred else 0
    rt     = np.array(pred['ref_times'])          if pred and 'ref_times'  in pred else None
    rp     = np.array(pred['ref_phases'])         if pred and 'ref_phases' in pred else None

    if source == 'chunk' and chunk_name:
        return build_chunk_figure(chunk_name, pt, pp, ant_ms, rt, rp)
    if source == 'test' and test_file:
        return build_test_figure(test_file, pt, pp, ant_ms, rt, rp)
    return go.Figure()


if __name__ == '__main__':
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f'\nDance Inspector')
    print(f'  chunks:      {args.chunks_path}  ({len(chunk_names)} chunks)')
    print(f'  test data:   {args.test_data_path}  ({len(test_files)} files)')
    print(f'  checkpoints: {args.checkpoints_path}  ({len(checkpoint_options)} found)')
    print(f'  open http://{local_ip}:{args.port}\n')
    app.run(debug=False, host='0.0.0.0', port=args.port)
