"""
PhaseGRUMel — GRU model with sliding-window mel spectrogram frontend.

Naturally stateful: hidden state h is passed forward frame-by-frame
during real-time inference.

Output has 3 channels: [sin(phase), cos(phase), phase_rate]
  - phase_rate: radians per frame (implicitly learned, no direct supervision)
  - Application layer can apply any time offset: phase += phase_rate * offset_frames

Input:
  x     : [batch, seq_len, frame_size]  — raw audio frames
  state : h or None                     — GRU hidden state

Output:
  ([batch, seq_len, 3], h)
"""

import torch
import torch.nn as nn

from config import frame_size
from .mel_frontend import MelFrontend

# ── Default hyperparameters ────────────────────────────────────────────────────
FFT_FRAMES = 2      # FFT window = FFT_FRAMES * frame_size samples
N_MELS     = 64
F_MIN      = 27.5   # Hz
F_MAX      = 8000.0 # Hz

HIDDEN     = 256
N_LAYERS   = 2
DROPOUT    = 0.2    # applied between GRU layers (not after last)


class PhaseGRUMel(nn.Module):

    def __init__(
        self,
        fft_frames = FFT_FRAMES,
        n_mels     = N_MELS,
        f_min      = F_MIN,
        f_max      = F_MAX,
        hidden     = HIDDEN,
        n_layers   = N_LAYERS,
        dropout    = DROPOUT,
    ):
        super().__init__()
        self.hparams = dict(
            fft_frames=fft_frames, n_mels=n_mels,
            f_min=f_min, f_max=f_max,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        )
        self.frontend    = MelFrontend(fft_frames, n_mels, f_min, f_max)
        self.input_proj  = nn.Linear(n_mels, hidden)
        self.gru         = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden, 3)  # sin, cos, phase_rate

    def forward(self, x, state=None, **kwargs):
        mel = self.frontend(x)                         # [batch, seq_len, n_mels]
        x   = self.input_proj(mel)                     # [batch, seq_len, hidden]
        out, new_state = self.gru(x, state)            # [batch, seq_len, hidden]
        return self.output_head(out), new_state

    def export_to_onnx(self, path, device):
        dummy_frames = torch.zeros(1, 100, frame_size, device=device)
        torch.onnx.export(
            self,
            (dummy_frames,),
            path,
            input_names=['frames'],
            output_names=['phase_out', 'state_h'],
            dynamic_axes={
                'frames':    {0: 'batch', 1: 'seq_len'},
                'phase_out': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
        )
        print(f'Exported ONNX to {path}')
