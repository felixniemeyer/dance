"""
PhaseTCNMel — same TCN architecture as PhaseTCN but with a mel-spectrogram
frontend instead of raw audio frames.

Each input frame (frame_size raw samples) is converted to N_MELS log-mel
bins before being fed to the TCN.  Everything else — dilated causal convs,
anticipation input, sin/cos output — is identical, so the two models can be
trained on the same chunks and compared directly.

Input:
  x            : [batch, seq_len, frame_size]  — raw audio frames (same as PhaseTCN)
  anticipation : [batch]                        — anticipation in seconds

Output:
  ([batch, seq_len, 2], None)   — sin/cos encoding of phase at t + anticipation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import frame_size, samplerate

# ── Mel frontend hyper-parameters ─────────────────────────────────────────────
N_MELS   = 64    # mel bins  — input to TCN per frame
N_FFT    = frame_size          # FFT size = one frame (320 → 161 positive freqs)
HOP      = frame_size          # hop = frame, so one spectrum per frame (no overlap)
F_MIN    = 27.5   # Hz  (A0, lowest piano note)
F_MAX    = 8000.0 # Hz  (Nyquist at 16 kHz)

# ── TCN hyper-parameters (same as PhaseTCN) ───────────────────────────────────
HIDDEN   = 64
KERNEL   = 3
N_BLOCKS = 8


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hz_to_mel(f):
    return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0))

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _build_mel_filterbank(n_fft, n_mels, sr, f_min, f_max):
    """Returns [n_mels, n_fft//2+1] float32 filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    freqs   = torch.linspace(0, sr / 2, n_freqs)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = _mel_to_hz(mel_pts)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up   = (freqs - lo)  / (ctr - lo + 1e-8)
        down = (hi - freqs)  / (hi - ctr + 1e-8)
        fb[m] = torch.clamp(torch.minimum(up, down), min=0.0)

    # Normalise each filter by its width (area = 1)
    widths = fb.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return fb / widths


class _MelFrontend(nn.Module):
    """
    Converts [batch, seq_len, frame_size] raw audio to
    [batch, seq_len, n_mels] log-mel features.
    No learned parameters — pure signal processing.
    """
    def __init__(self):
        super().__init__()
        fb = _build_mel_filterbank(N_FFT, N_MELS, samplerate, F_MIN, F_MAX)
        self.register_buffer('filterbank', fb)          # [n_mels, n_freqs]
        window = torch.hann_window(N_FFT)
        self.register_buffer('window', window)

    def forward(self, x):
        # x: [batch, seq_len, frame_size]
        batch, seq_len, _ = x.shape
        flat = x.reshape(batch * seq_len, N_FFT)        # treat each frame independently

        # FFT magnitude spectrum
        spec = torch.fft.rfft(flat * self.window.unsqueeze(0), n=N_FFT)
        power = spec.real ** 2 + spec.imag ** 2         # [B*T, n_freqs]

        # Apply mel filterbank
        mel = power @ self.filterbank.T                 # [B*T, n_mels]
        log_mel = torch.log(mel + 1e-6)                 # log compression

        return log_mel.reshape(batch, seq_len, N_MELS)  # [batch, seq_len, n_mels]


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
        r = x
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        return x + r


class PhaseTCNMel(nn.Module):
    hparams = {
        'n_mels':   N_MELS,
        'n_fft':    N_FFT,
        'f_min':    F_MIN,
        'f_max':    F_MAX,
        'hidden':   HIDDEN,
        'kernel':   KERNEL,
        'n_blocks': N_BLOCKS,
    }

    def __init__(self):
        super().__init__()
        self.frontend    = _MelFrontend()
        # +1 for anticipation
        self.input_proj  = nn.Linear(N_MELS + 1, HIDDEN)
        self.blocks      = nn.ModuleList([
            _TCNBlock(HIDDEN, KERNEL, dilation=2 ** i)
            for i in range(N_BLOCKS)
        ])
        self.output_head = nn.Linear(HIDDEN, 2)

    def forward(self, x, anticipation=None, state=None):
        # x: [batch, seq_len, frame_size]  raw audio
        batch, seq_len, _ = x.shape

        if anticipation is None:
            anticipation = torch.zeros(batch, device=x.device)

        x = self.frontend(x)                             # [batch, seq_len, n_mels]

        ant = anticipation.unsqueeze(1).expand(batch, seq_len).unsqueeze(2)
        x = torch.cat([x, ant], dim=2)                  # [batch, seq_len, n_mels+1]

        x = self.input_proj(x)                          # [batch, seq_len, hidden]
        x = x.transpose(1, 2)                           # [batch, hidden, seq_len]

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)                           # [batch, seq_len, hidden]
        return self.output_head(x), None

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
