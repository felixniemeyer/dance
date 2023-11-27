"""
torch dataset implementation for loading chunks
"""

import os
import math

import torchaudio
import torch

from torch.utils.data import Dataset

from audio_event import AudioEvent

def loadEventsAsLabels(events_file_path, number_of_frames, samplerate, frame_size):
    labels = torch.zeros(number_of_frames, 2)

    mapping = {
        35: 0, # kick
        36: 0, # kick
        38: 1, # snare
        40: 1, # snare
    }
    with open(events_file_path, 'r', encoding='utf8') as file:
        for line in file:
            event = AudioEvent.from_csv_line(line)
            if event.note in mapping:
                frame = math.floor(event.time * samplerate / frame_size)
                group = mapping[event.note]
                labels[frame][group] = 1 - ((1 - labels[frame][group]) * (1 - event.volume))

    return labels

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
        # check that .kicks and .snares files exist. if yes add to self.chunk_names
        count = 0
        self.chunk_names = []
        for filename in filenames:
            if os.path.exists(os.path.join(data_path, filename + ".events")):
                self.chunk_names.append(filename)
                count += 1

    def __len__(self):
        return len(self.chunk_names)

    def __getitem__(self, index):
        chunk_name = self.chunk_names[index]

        audio_file_path = os.path.join(self.path, chunk_name + ".ogg")
        events_file_path = os.path.join(self.path, chunk_name + ".events")

        audio, samplerate = torchaudio.load(audio_file_path)

        assert samplerate == self.samplerate, "sample rate mismatch"

        # convert to mono
        audio = audio.mean(0)
        # normalize
        audio = audio / audio.abs().max()

        number_of_frames = audio.shape[0] // self.frame_size
        sequence_size = number_of_frames * self.frame_size

        frames = audio[:sequence_size].reshape(-1, self.frame_size)

        assert number_of_frames == frames.shape[0], "sequence size mismatch"

        labels = loadEventsAsLabels(events_file_path, number_of_frames, samplerate, self.frame_size)

        return frames, labels, audio_file_path
