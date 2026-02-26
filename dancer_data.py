"""
torch dataset implementation for loading phase chunks
"""

import math
import os
from bisect import bisect_right

import numpy as np
import soundfile
import torch

from torch.utils.data import Dataset

import config


def load_bar_starts(bars_file_path):
    """Load bar start times (in seconds) from a .bars file."""
    values = []
    with open(bars_file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return values


def compute_labels(bar_starts, n_frames, frame_size, samplerate):
    """
    Compute phase and phase_rate labels from bar start times.

    Args:
        bar_starts: list of bar start times in seconds (relative to chunk start)
        n_frames:   number of output frames
        frame_size: samples per frame
        samplerate: audio sample rate

    Returns:
        phase_tensor: [n_frames] float32, phase in [0, 1)
        rate_tensor:  [n_frames] float32, phase_rate in radians/frame
                      (2π per bar, constant within each bar)
    """
    phase = np.zeros(n_frames, dtype=np.float32)
    rate  = np.zeros(n_frames, dtype=np.float32)

    if len(bar_starts) < 2:
        return torch.from_numpy(phase), torch.from_numpy(rate)

    bar_starts = sorted(bar_starts)

    for fi in range(n_frames):
        t = fi * frame_size / samplerate
        idx = bisect_right(bar_starts, t) - 1
        if idx < 0 or idx + 1 >= len(bar_starts):
            continue  # outside bar coverage — leave 0
        bar_start = bar_starts[idx]
        bar_end   = bar_starts[idx + 1]
        bar_dur   = bar_end - bar_start
        if bar_dur <= 0:
            continue
        phase[fi] = (t - bar_start) / bar_dur
        # rate: 2π radians per bar, expressed per frame
        rate[fi]  = 2.0 * math.pi * frame_size / (bar_dur * samplerate)

    return torch.from_numpy(phase), torch.from_numpy(rate)


class DanceDataset(Dataset):
    def __init__(self, data_path, frame_size, samplerate):
        self.path       = data_path
        self.frame_size = frame_size
        self.samplerate = samplerate

        filenames = [f[:-4] for f in os.listdir(data_path) if f.endswith('.ogg')]
        self.chunk_names = [
            fn for fn in filenames
            if os.path.exists(os.path.join(data_path, fn + '.bars'))
        ]

    def __len__(self):
        return len(self.chunk_names)

    def __getitem__(self, index):
        chunk_name = self.chunk_names[index]

        audio_file = os.path.join(self.path, chunk_name + '.ogg')
        bars_file  = os.path.join(self.path, chunk_name + '.bars')

        audio, sr = soundfile.read(audio_file, dtype='float32')
        assert sr == self.samplerate, f'sample rate mismatch: {sr} != {self.samplerate}'

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.from_numpy(audio)
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        expected_frames  = int(config.chunk_duration * self.samplerate) // self.frame_size
        expected_samples = expected_frames * self.frame_size

        if audio.shape[0] < expected_samples:
            audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[0]))
        else:
            audio = audio[:expected_samples]

        bar_starts = load_bar_starts(bars_file)
        phase_labels, rate_labels = compute_labels(
            bar_starts, expected_frames, self.frame_size, self.samplerate)

        frames = audio.reshape(expected_frames, self.frame_size)

        return frames, phase_labels, rate_labels, audio_file
