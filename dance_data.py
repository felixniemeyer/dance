import os
import torchaudio
import torch

from torch.utils.data import Dataset

def readEventsFileIntoLabels(filename, buffer_count, buffer_size, samplerate):
    events = []
    with open(filename, 'r') as file:
        for line in file:
            events.append(float(line) * samplerate)

    # create tensor with shape (sequence_size, 1)
    intensities = torch.zeros(buffer_count)
    index = 0

    intensity = 0

    for i in range(0, buffer_count):
        start = i * buffer_size
        exclusive_end = start + buffer_size

        intensity = 0
        while index < len(events) and events[index] < exclusive_end:
            intensity = 1.
            index += 1

        intensities[i] = intensity

    return intensities

class DanceDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        buffer_size, 
        samplerate, 
        max_size = None, 
        print_filename=False, 
    ):
        self.path = data_path
        self.print_filename = print_filename        
        self.buffer_size = buffer_size
        self.samplerate = samplerate

        filenames = [f[:-4] for f in os.listdir(data_path) if f.endswith(".ogg")]
        # check that .kicks and .snares files exist. if yes add to self.chunk_names
        count = 0
        self.chunk_names = []
        for filename in filenames:
            if os.path.exists(os.path.join(data_path, filename + ".kicks")) and os.path.exists(os.path.join(data_path, filename + ".snares")):
                self.chunk_names.append(filename)
                count += 1

        if max_size is not None: 
            if max_size < len(self.chunk_names):
                self.chunk_names = self.chunk_names[:max_size]
            else: 
                print('warning: wanted', max_size, 'chunks, but only found', len(self.chunk_names))

    def __len__(self):
        return len(self.chunk_names)

    def __getitem__(self, index):
        chunk_name = self.chunk_names[index]

        audiofile = os.path.join(self.path, chunk_name + ".ogg")
        kicksfile = os.path.join(self.path, chunk_name + ".kicks")
        snaresfile = os.path.join(self.path, chunk_name + ".snares")

        if self.print_filename:
            print(audiofile)

        audio, samplerate = torchaudio.load(audiofile)

        assert samplerate == self.samplerate, "sample rate mismatch"

        # convert to mono
        audio = audio.mean(0)
        # normalize
        audio = audio / audio.abs().max()

        buffer_count = audio.shape[0] // self.buffer_size
        sequence_size = buffer_count * self.buffer_size

        buffers = audio[:sequence_size].reshape(-1, self.buffer_size)

        assert buffer_count == buffers.shape[0], "sequence size mismatch"
        
        kicks = readEventsFileIntoLabels(kicksfile, buffer_count, self.buffer_size, samplerate)
        snares = readEventsFileIntoLabels(snaresfile, buffer_count, self.buffer_size, samplerate)

        labels = torch.stack([kicks, snares], dim=1)

        return buffers, labels

