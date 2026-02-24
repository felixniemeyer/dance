"""
PhaseGRU — GRU with learnable CNN frame encoder.

Instead of a fixed mel filterbank, this model learns its own frequency
features from raw audio frames via a small 1-D CNN applied per frame.

Architecture:
  FrameEncoder  : Conv1d × 2 + AdaptiveAvgPool + Linear  (per frame, no temporal context)
  input_proj    : Linear  (FRAME_FEATURES → HIDDEN)
  GRU           : N_LAYERS layers, HIDDEN units
  output_head   : Linear  (HIDDEN → 3)

Output: [sin(phase), cos(phase), phase_rate] per frame.

Input:
  x     : [batch, seq_len, frame_size]  — raw audio frames
  state : h or None                     — GRU hidden state

Output:
  ([batch, seq_len, 3], h)
"""

import torch
import torch.nn as nn

from config import frame_size

# ── Hyperparameters ────────────────────────────────────────────────────────────
FRAME_FEATURES = 128   # output dim of per-frame CNN encoder
HIDDEN         = 256
N_LAYERS       = 2
DROPOUT        = 0.3   # slightly higher than mel models


class FrameEncoder(nn.Module):
    """
    1-D CNN that processes each audio frame (frame_size=320 samples) independently.
    Acts as a learnable filterbank replacement for the mel frontend.

    Input : [N, frame_size]
    Output: [N, out_features]
    """
    def __init__(self, out_features):
        super().__init__()
        # frame_size=320
        # Conv1(k=8, s=2) → (320-8)/2+1 = 157,  32 ch
        # Conv2(k=4, s=2) → (157-4)/2+1 = 77,   64 ch
        # AdaptiveAvgPool1d(8)            → 8,    64 ch  → flatten = 512
        self.convs = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 8, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [N, frame_size]
        x = x.unsqueeze(1)    # [N, 1, frame_size]
        x = self.convs(x)     # [N, 64, 8]
        x = x.flatten(1)      # [N, 512]
        return self.proj(x)   # [N, out_features]


class PhaseGRU(nn.Module):
    hparams = {
        'frame_features': FRAME_FEATURES,
        'hidden':         HIDDEN,
        'n_layers':       N_LAYERS,
        'dropout':        DROPOUT,
    }

    def __init__(self):
        super().__init__()
        self.encoder     = FrameEncoder(FRAME_FEATURES)
        self.input_proj  = nn.Linear(FRAME_FEATURES, HIDDEN)
        self.gru         = nn.GRU(
            input_size=HIDDEN,
            hidden_size=HIDDEN,
            num_layers=N_LAYERS,
            batch_first=True,
            dropout=DROPOUT if N_LAYERS > 1 else 0.0,
        )
        self.output_head = nn.Linear(HIDDEN, 3)   # sin, cos, phase_rate

    def forward(self, x, state=None, **kwargs):
        batch, seq_len, _ = x.shape
        flat     = x.reshape(batch * seq_len, frame_size)        # [B*T, frame_size]
        features = self.encoder(flat)                             # [B*T, FRAME_FEATURES]
        features = features.reshape(batch, seq_len, FRAME_FEATURES)
        z        = self.input_proj(features)                      # [batch, seq_len, HIDDEN]
        out, new_state = self.gru(z, state)                       # [batch, seq_len, HIDDEN]
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
