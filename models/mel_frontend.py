"""
Shared sliding-window mel spectrogram frontend.

Takes [batch, seq_len, frame_size] raw audio frames and produces
[batch, seq_len, n_mels] log-mel features.

Each output frame's spectrum is computed from a causal window of
fft_frames consecutive input frames (looking back in time), giving:
  - temporal resolution = frame_size / samplerate  (unchanged, no data pipeline change)
  - frequency resolution = samplerate / (fft_frames * frame_size)

Example at samplerate=16000, frame_size=320:
  fft_frames=1 → 50 Hz/bin  (20ms window,  current behaviour)
  fft_frames=4 → 12.5 Hz/bin (80ms window,  good for kick/bass)
  fft_frames=8 → 6.25 Hz/bin (160ms window, very high freq resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import frame_size, samplerate


def _hz_to_mel(f):
    return 2595.0 * torch.log10(torch.tensor(1.0 + f / 700.0))

def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _build_mel_filterbank(n_fft, n_mels, sr, f_min, f_max):
    """Returns [n_mels, n_fft//2+1] normalised filterbank matrix."""
    n_freqs = n_fft // 2 + 1
    freqs   = torch.linspace(0, sr / 2, n_freqs)

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = _mel_to_hz(mel_pts)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up   = (freqs - lo)  / (ctr - lo  + 1e-8)
        down = (hi - freqs)  / (hi  - ctr + 1e-8)
        fb[m] = torch.clamp(torch.minimum(up, down), min=0.0)

    widths = fb.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return fb / widths


class MelFrontend(nn.Module):
    """
    No learned parameters — pure signal processing.

    Args:
        fft_frames : number of input frames per FFT window (>=1)
        n_mels     : number of mel bins
        f_min      : lowest frequency (Hz)
        f_max      : highest frequency (Hz)
    """
    def __init__(self, fft_frames=4, n_mels=64, f_min=27.5, f_max=8000.0):
        super().__init__()
        self.fft_frames = fft_frames
        self.n_mels     = n_mels
        n_fft = fft_frames * frame_size

        fb = _build_mel_filterbank(n_fft, n_mels, samplerate, f_min, f_max)
        self.register_buffer('filterbank', fb)          # [n_mels, n_freqs]
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, x):
        # x: [batch, seq_len, frame_size]
        batch, seq_len, _ = x.shape
        n_fft = self.fft_frames * frame_size

        # Flatten to audio stream, then causal-pad so frame 0 has fft_frames-1
        # zero-frames of context behind it.
        audio = x.reshape(batch, seq_len * frame_size)
        audio = F.pad(audio, ((self.fft_frames - 1) * frame_size, 0))

        # Sliding windows: [batch, seq_len, n_fft]
        windows = audio.unfold(-1, n_fft, frame_size)

        # Spectrogram
        flat  = windows.reshape(batch * seq_len, n_fft)
        spec  = torch.fft.rfft(flat * self.window.unsqueeze(0), n=n_fft)
        power = spec.real ** 2 + spec.imag ** 2         # [B*T, n_freqs]

        mel     = power @ self.filterbank.T             # [B*T, n_mels]
        log_mel = torch.log(mel + 1e-6)

        return log_mel.reshape(batch, seq_len, self.n_mels)
