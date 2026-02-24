"""
PhaseLSTMMel — LSTM model with sliding-window mel spectrogram frontend.

Naturally stateful: hidden state (h, c) is passed forward frame-by-frame
during inference, making it well-suited for real-time use.

Input:
  x            : [batch, seq_len, frame_size]  — raw audio frames
  anticipation : [batch]                        — anticipation in seconds
  state        : ((h, c)) or None              — LSTM state

Output:
  ([batch, seq_len, 2], (h, c))  — sin/cos phase encoding + new state
"""

import torch
import torch.nn as nn

from config import frame_size, samplerate
from .mel_frontend import MelFrontend

# ── Hyperparameters ────────────────────────────────────────────────────────────
FFT_FRAMES = 4      # FFT window = FFT_FRAMES * frame_size samples
                    # latency added: (FFT_FRAMES-1) * frame_size / samplerate seconds
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
        self.input_proj  = nn.Linear(N_MELS + 1, HIDDEN)   # +1 for anticipation
        self.lstm        = nn.LSTM(
            input_size=HIDDEN,
            hidden_size=HIDDEN,
            num_layers=N_LAYERS,
            batch_first=True,
            dropout=DROPOUT if N_LAYERS > 1 else 0.0,
        )
        self.output_head = nn.Linear(HIDDEN, 2)

    def forward(self, x, anticipation=None, state=None):
        batch, seq_len, _ = x.shape

        if anticipation is None:
            anticipation = torch.zeros(batch, device=x.device)

        mel = self.frontend(x)                                      # [batch, seq_len, n_mels]
        ant = anticipation.unsqueeze(1).expand(batch, seq_len).unsqueeze(2)
        x   = torch.cat([mel, ant], dim=2)                         # [batch, seq_len, n_mels+1]
        x   = self.input_proj(x)                                   # [batch, seq_len, hidden]

        out, new_state = self.lstm(x, state)                       # [batch, seq_len, hidden]

        return self.output_head(out), new_state

    def export_to_onnx(self, path, device):
        dummy_frames = torch.zeros(1, 100, frame_size, device=device)
        dummy_ant    = torch.zeros(1, device=device)
        torch.onnx.export(
            self,
            (dummy_frames, dummy_ant, None),
            path,
            input_names=['frames', 'anticipation'],
            output_names=['phase_sincos', 'state_h', 'state_c'],
            dynamic_axes={
                'frames':       {0: 'batch', 1: 'seq_len'},
                'anticipation': {0: 'batch'},
                'phase_sincos': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
        )
        print(f'Exported ONNX to {path}')
