"""
Temporal Convolutional Network for bar-phase estimation.

Input:
  x            : [batch, seq_len, frame_size]  — audio frames
  anticipation : [batch]                        — anticipation in seconds (explicit model input)

Output:
  ([batch, seq_len, 2], None)   — sin/cos encoding of predicted phase at t + anticipation
  The None keeps the same (output, state) interface as the RNN models.
"""

import torch
import torch.nn as nn

from config import frame_size

HIDDEN   = 64
KERNEL   = 3
N_BLOCKS = 8   # receptive field: sum of 2*(kernel-1)*dilation for each block


class _CausalConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=self.pad)

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
        self.act   = nn.GELU()

    def forward(self, x):
        # x: [batch, channels, seq_len]
        r = x
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        return x + r


class PhaseTCN(nn.Module):
    def __init__(self):
        super().__init__()
        # +1 for anticipation scalar appended per frame
        self.input_proj = nn.Linear(frame_size + 1, HIDDEN)
        self.blocks = nn.ModuleList([
            _TCNBlock(HIDDEN, KERNEL, dilation=2 ** i)
            for i in range(N_BLOCKS)
        ])
        self.output_head = nn.Linear(HIDDEN, 2)

    def forward(self, x, anticipation=None, state=None):
        batch, seq_len, _ = x.shape

        if anticipation is None:
            anticipation = torch.zeros(batch, device=x.device)

        # Broadcast anticipation across time axis and append to frames
        ant = anticipation.unsqueeze(1).expand(batch, seq_len).unsqueeze(2)
        x = torch.cat([x, ant], dim=2)          # [batch, seq_len, frame_size+1]

        x = self.input_proj(x)                   # [batch, seq_len, hidden]
        x = x.transpose(1, 2)                    # [batch, hidden, seq_len]

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)                    # [batch, seq_len, hidden]
        return self.output_head(x), None          # [batch, seq_len, 2], no state

    def export_to_onnx(self, path, device):
        dummy_frames = torch.zeros(1, 100, frame_size, device=device)
        dummy_ant    = torch.zeros(1, device=device)
        torch.onnx.export(
            self,
            (dummy_frames, dummy_ant),
            path,
            input_names=['frames', 'anticipation'],
            output_names=['phase_sincos'],
            dynamic_axes={
                'frames':       {0: 'batch', 1: 'seq_len'},
                'anticipation': {0: 'batch'},
                'phase_sincos': {0: 'batch', 1: 'seq_len'},
            },
            opset_version=17,
        )
        print(f'Exported ONNX to {path}')
