"""
torch dataset implementation for loading phase chunks
"""

import os

import torchaudio
import torch

from torch.utils.data import Dataset

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

        audio, samplerate = torchaudio.load(audio_file_path)

        assert samplerate == self.samplerate, "sample rate mismatch"

        # convert to mono
        audio = audio.mean(0)
        # normalize
        audio = audio / audio.abs().max()

        number_of_frames = audio.shape[0] // self.frame_size
        sequence_size = number_of_frames * self.frame_size

        phase_labels = load_phase_labels(phase_file_path)
        number_of_frames = min(number_of_frames, phase_labels.shape[0])
        sequence_size = number_of_frames * self.frame_size

        frames = audio[:sequence_size].reshape(number_of_frames, self.frame_size)
        phase_labels = phase_labels[:number_of_frames]

        assert number_of_frames == frames.shape[0], "sequence size mismatch"
        assert number_of_frames == phase_labels.shape[0], "phase labels size mismatch"

        return frames, phase_labels, audio_file_path
