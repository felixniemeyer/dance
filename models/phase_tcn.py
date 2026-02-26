"""
Temporal Convolutional Network for bar-phase estimation (raw-frame variant).

Input:
  x : [batch, seq_len, frame_size]  — audio frames

Output:
  ([batch, seq_len, 2], None)   — sin/cos encoding of current phase
"""

import torch
import torch.nn as nn

from config import frame_size

HIDDEN = 64
KERNEL = 3
N_BLOCKS = 8


class _CausalConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class _TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = _CausalConv(channels, kernel_size, dilation)
        self.conv2 = _CausalConv(channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [batch, channels, seq_len]
        residual = x
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        return x + residual


class PhaseTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(frame_size, HIDDEN)
        self.blocks = nn.ModuleList([
            _TCNBlock(HIDDEN, KERNEL, dilation=2 ** i)
            for i in range(N_BLOCKS)
        ])
        self.output_head = nn.Linear(HIDDEN, 2)

    def forward(self, x, state=None, **kwargs):
        x = self.input_proj(x)          # [batch, seq_len, hidden]
        x = x.transpose(1, 2)           # [batch, hidden, seq_len]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)           # [batch, seq_len, hidden]
        return self.output_head(x), None

    def export_to_onnx(self, path, device):
        dummy_frames = torch.zeros(1, 100, frame_size, device=device)
        torch.onnx.export(
            self,
            (dummy_frames,),
            path,
            input_names=['frames'],
            output_names=['phase_sincos'],
            dynamic_axes={
                'frames':      {0: 'batch', 1: 'seq_len'},
                'phase_sincos': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
        )
        print(f'Exported ONNX to {path}')
