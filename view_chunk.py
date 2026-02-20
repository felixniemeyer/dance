"""
Interactive chunk viewer — waveform + phase overlay.

Usage:
    venv/bin/python view_chunk.py --chunks-path data/chunks/<run>
    venv/bin/python view_chunk.py --chunks-path data/chunks/<run> --chunk A_N522-KORv3)-0

Then open http://127.0.0.1:8050 in a browser.

Controls:
  - Dropdown to select any chunk in the directory
  - Linked zoom/pan on both panels (drag, scroll, box-select)
  - Phase trace + vertical bar lines at phase resets
  - Waveform downsampled for performance
"""

import argparse
import os

import numpy as np
import soundfile
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--chunks-path', type=str, required=True,
                    help='Directory containing .ogg and .phase chunk files')
parser.add_argument('--chunk', type=str, default=None,
                    help='Initial chunk name to display (without extension)')
parser.add_argument('--port', type=int, default=8050)
args = parser.parse_args()

# ── Discover chunks ────────────────────────────────────────────────────────────

def find_chunks(path):
    names = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.ogg'):
            stem = f[:-4]
            if os.path.exists(os.path.join(path, stem + '.phase')):
                names.append(stem)
    return names

chunk_names = find_chunks(args.chunks_path)
if not chunk_names:
    raise SystemExit(f'No paired .ogg/.phase files found in {args.chunks_path}')

initial_chunk = args.chunk if args.chunk in chunk_names else chunk_names[0]

# ── Load helpers ───────────────────────────────────────────────────────────────

MAX_WAVEFORM_POINTS = 8000  # downsample for rendering performance

def load_chunk(chunk_name):
    ogg_path   = os.path.join(args.chunks_path, chunk_name + '.ogg')
    phase_path = os.path.join(args.chunks_path, chunk_name + '.phase')

    audio, sr = soundfile.read(ogg_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    with open(phase_path, 'r') as f:
        phase = np.array([float(l) for l in f if l.strip()], dtype=np.float32)

    duration    = len(audio) / sr
    audio_times = np.linspace(0, duration, len(audio))
    frame_dur   = duration / len(phase)
    phase_times = np.arange(len(phase)) * frame_dur + frame_dur / 2  # frame centres

    # downsample waveform
    if len(audio) > MAX_WAVEFORM_POINTS:
        step = len(audio) // MAX_WAVEFORM_POINTS
        audio_times = audio_times[::step]
        audio       = audio[::step]

    # find bar boundaries: frames where phase wraps (previous value > 0.8 and current < 0.2)
    bar_times = []
    for i in range(1, len(phase)):
        if phase[i - 1] > 0.8 and phase[i] < 0.2:
            # linear interpolation for sub-frame accuracy
            t = phase_times[i - 1] + (phase_times[i] - phase_times[i - 1]) * (
                (1.0 - phase[i - 1]) / (1.0 - phase[i - 1] + phase[i])
            )
            bar_times.append(t)

    return audio_times, audio, phase_times, phase, bar_times, duration, sr


def make_figure(chunk_name):
    audio_times, audio, phase_times, phase, bar_times, duration, sr = load_chunk(chunk_name)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.06,
        subplot_titles=['Waveform', 'Bar Phase'],
    )

    # waveform
    fig.add_trace(
        go.Scatter(x=audio_times, y=audio, mode='lines', name='audio',
                   line=dict(color='steelblue', width=0.6), hoverinfo='skip'),
        row=1, col=1,
    )

    # phase trace
    fig.add_trace(
        go.Scatter(x=phase_times, y=phase, mode='lines', name='phase',
                   line=dict(color='darkorange', width=1.2)),
        row=2, col=1,
    )

    # bar boundary vertical lines (shapes span both subplots via paper coords)
    shapes = []
    for t in bar_times:
        x_frac = t / duration
        shapes.append(dict(
            type='line',
            xref='x', yref='paper',
            x0=t, x1=t,
            y0=0, y1=1,
            line=dict(color='rgba(200,60,60,0.45)', width=1, dash='dot'),
        ))

    fig.update_layout(
        shapes=shapes,
        title=dict(text=chunk_name, font=dict(size=13)),
        height=600,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation='h', y=1.06),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#eee'),
        dragmode='pan',
        hovermode='x unified',
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#2a2a4a',
        zeroline=False,
        rangeslider=dict(visible=False),
    )
    fig.update_yaxes(showgrid=True, gridcolor='#2a2a4a', zeroline=False)
    fig.update_yaxes(title_text='amplitude', row=1, col=1)
    fig.update_yaxes(title_text='phase [0,1)', range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text='time (s)', row=2, col=1)

    return fig


# ── Dash app ───────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = 'Chunk viewer'

app.layout = html.Div([
    html.Div([
        html.Label('Chunk:', style={'color': '#eee', 'marginRight': '8px'}),
        dcc.Dropdown(
            id='chunk-dropdown',
            options=[{'label': n, 'value': n} for n in chunk_names],
            value=initial_chunk,
            clearable=False,
            style={'width': '600px', 'display': 'inline-block', 'color': '#111'},
        ),
        html.Span(
            f'  {len(chunk_names)} chunks',
            style={'color': '#aaa', 'marginLeft': '12px', 'fontSize': '13px'},
        ),
    ], style={'padding': '12px 16px', 'background': '#0f3460'}),

    dcc.Graph(
        id='chunk-graph',
        figure=make_figure(initial_chunk),
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline'],
        },
        style={'height': '620px'},
    ),
], style={'background': '#16213e', 'minHeight': '100vh'})


@app.callback(Output('chunk-graph', 'figure'), Input('chunk-dropdown', 'value'))
def update_graph(chunk_name):
    return make_figure(chunk_name)


if __name__ == '__main__':
    print(f'\nChunk viewer — {len(chunk_names)} chunks in {args.chunks_path}')
    print(f'Open http://127.0.0.1:{args.port}\n')
    app.run(debug=False, port=args.port)
