"""
Causal log-mel frontend + pure TCN for phase prediction.
Outputs sin/cos components to avoid phase-wrap discontinuity.
"""

import torch
from torch import nn
import torch.nn.functional as nnf
import torchaudio

from config import frame_size, samplerate


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(nnf.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.activation(y)
        y = self.dropout(y)
        return x + y


class PhaseTCN(nn.Module):
    def __init__(
        self,
        mel_bins=64,
        n_fft=512,
        channels=128,
        blocks=8,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.n_fft = n_fft
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

        mel_fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=30.0,
            f_max=samplerate / 2.0,
            n_mels=mel_bins,
            sample_rate=samplerate,
            norm="slaney",
            mel_scale="htk",
        )
        self.register_buffer("mel_fbanks", mel_fbanks)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        self.input_projection = nn.Conv1d(mel_bins + 1, channels, kernel_size=1)
        dilations = [2 ** i for i in range(blocks)]
        self.tcn_blocks = nn.Sequential(*[
            TCNBlock(channels, kernel_size=kernel_size, dilation=d, dropout=dropout) for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 2, kernel_size=1),
            nn.Tanh(),
        )

    def compute_log_mel(self, batch_inputs):
        # batch_inputs shape: [batch, frames, frame_size]
        if self.n_fft > frame_size:
            right_pad = self.n_fft - frame_size
            padded = nnf.pad(batch_inputs, (0, right_pad))
        else:
            padded = batch_inputs[..., :self.n_fft]

        window = self.window.to(padded.device, dtype=padded.dtype)
        spectrum = torch.fft.rfft(padded * window, n=self.n_fft, dim=-1)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        mel = torch.matmul(power, self.mel_fbanks.to(power.device, dtype=power.dtype))
        return torch.log1p(mel)

    def forward(self, batch_inputs, anticipation=None, state=None):
        batch_size, sequence_length, _ = batch_inputs.shape

        if anticipation is None:
            anticipation = torch.zeros(batch_size, device=batch_inputs.device, dtype=batch_inputs.dtype)

        log_mel = self.compute_log_mel(batch_inputs)  # [batch, frames, mel_bins]
        anticipation_feature = anticipation.view(batch_size, 1, 1).expand(-1, sequence_length, 1)
        x = torch.cat([log_mel, anticipation_feature], dim=2).transpose(1, 2)  # [batch, channels, frames]

        x = self.input_projection(x)
        x = self.tcn_blocks(x)
        out = self.head(x).transpose(1, 2)  # [batch, frames, 2]

        return out, None

    def export_to_onnx(self, outfile, device):
        print('Exporting to onnx.')
        inputs = torch.rand(1, 1, frame_size, device=device)
        anticipation = torch.rand(1, device=device)

        _output, _ = self(inputs, anticipation=anticipation)

        torch.onnx.export(
            self,
            (inputs, anticipation),
            outfile,
            input_names=['input', 'anticipation'],
            output_names=['output'],
            dynamic_axes={'input': {1: 'frames'}, 'output': {1: 'frames'}},
        )
