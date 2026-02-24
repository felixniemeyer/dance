"""
PhaseLSTMMel — LSTM model with sliding-window mel spectrogram frontend.

Naturally stateful: hidden state (h, c) is passed forward frame-by-frame
during real-time inference.

Output has 3 channels: [sin(phase), cos(phase), phase_rate]
  - phase_rate: radians per frame (implicitly learned, no direct supervision)
  - Application layer can apply any time offset: phase += phase_rate * offset_frames

Input:
  x     : [batch, seq_len, frame_size]  — raw audio frames
  state : (h, c) or None               — LSTM state

Output:
  ([batch, seq_len, 3], (h, c))
"""

import torch
import torch.nn as nn

from config import frame_size
from .mel_frontend import MelFrontend

# ── Hyperparameters ────────────────────────────────────────────────────────────
FFT_FRAMES = 4      # FFT window = FFT_FRAMES * frame_size samples
N_MELS     = 64
F_MIN      = 27.5   # Hz
F_MAX      = 8000.0 # Hz

HIDDEN     = 256
N_LAYERS   = 2
DROPOUT    = 0.2    # applied between LSTM layers (not after last)


class PhaseLSTMMel(nn.Module):
    hparams = {
        'fft_frames': FFT_FRAMES,
        'n_mels':     N_MELS,
        'f_min':      F_MIN,
        'f_max':      F_MAX,
        'hidden':     HIDDEN,
        'n_layers':   N_LAYERS,
        'dropout':    DROPOUT,
    }

    def __init__(self):
        super().__init__()
        self.frontend    = MelFrontend(FFT_FRAMES, N_MELS, F_MIN, F_MAX)
        self.input_proj  = nn.Linear(N_MELS, HIDDEN)
        self.lstm        = nn.LSTM(
            input_size=HIDDEN,
            hidden_size=HIDDEN,
            num_layers=N_LAYERS,
            batch_first=True,
            dropout=DROPOUT if N_LAYERS > 1 else 0.0,
        )
        self.output_head = nn.Linear(HIDDEN, 3)  # sin, cos, phase_rate

    def forward(self, x, state=None, **kwargs):
        mel = self.frontend(x)               # [batch, seq_len, n_mels]
        x   = self.input_proj(mel)           # [batch, seq_len, hidden]
        out, new_state = self.lstm(x, state) # [batch, seq_len, hidden]
        return self.output_head(out), new_state

    def export_to_onnx(self, path, device):
        dummy_frames = torch.zeros(1, 100, frame_size, device=device)
        torch.onnx.export(
            self,
            (dummy_frames,),
            path,
            input_names=['frames'],
            output_names=['phase_out', 'state_h', 'state_c'],
            dynamic_axes={
                'frames':    {0: 'batch', 1: 'seq_len'},
                'phase_out': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
        )
        print(f'Exported ONNX to {path}')
