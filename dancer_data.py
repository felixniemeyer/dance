"""
torch dataset implementation for loading phase chunks
"""

import os

import numpy as np
import soundfile
import torch

from torch.utils.data import Dataset

import config

def load_phase_labels(phase_file_path):
    values = []
    with open(phase_file_path, 'r', encoding='utf8') as file:
        for line in file:
            line = line.strip()
            if line != '':
                values.append(float(line))
    return torch.tensor(values, dtype=torch.float32)

class DanceDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_size,
        samplerate
    ):
        self.path = data_path
        self.frame_size = frame_size
        self.samplerate = samplerate

        filenames = [f[:-4] for f in os.listdir(data_path) if f.endswith(".ogg")]
        self.chunk_names = []
        for filename in filenames:
            if os.path.exists(os.path.join(data_path, filename + ".phase")):
                self.chunk_names.append(filename)

    def __len__(self):
        return len(self.chunk_names)

    def __getitem__(self, index):
        chunk_name = self.chunk_names[index]

        audio_file_path = os.path.join(self.path, chunk_name + ".ogg")
        phase_file_path = os.path.join(self.path, chunk_name + ".phase")

        audio, samplerate = soundfile.read(audio_file_path, dtype='float32')

        assert samplerate == self.samplerate, "sample rate mismatch"

        # convert to mono if stereo
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.from_numpy(audio)
        # normalize
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        expected_frames = int(config.chunk_duration * self.samplerate) // self.frame_size
        expected_samples = expected_frames * self.frame_size

        # pad or truncate audio to exact length
        if audio.shape[0] < expected_samples:
            audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[0]))
        else:
            audio = audio[:expected_samples]

        phase_labels = load_phase_labels(phase_file_path)
        # Keep however many label frames the file contains. Only pad if short.
        if phase_labels.shape[0] < expected_frames:
            phase_labels = torch.nn.functional.pad(phase_labels, (0, expected_frames - phase_labels.shape[0]))

        frames = audio.reshape(expected_frames, self.frame_size)

        return frames, phase_labels, audio_file_path
