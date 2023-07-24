import os
import torchaudio
import torch

from torch.utils.data import Dataset

import config


class DanceDataset(Dataset):
    def __init__(self, path, max_size = None):
        self.path = path
        filenames = [f[:-4] for f in os.listdir(path) if f.endswith(".ogg")]
        # check that .kicks and .snares files exist. if yes add to self.chunk_names
        count = 0
        self.chunk_names = []
        for filename in filenames:
            if os.path.exists(os.path.join(path, filename + ".kicks")) and os.path.exists(os.path.join(path, filename + ".snares")):
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

        audio, samplerate = torchaudio.load(audiofile)

        assert samplerate == config.sample_rate, "sample rate mismatch"
        assert audio.shape[0] == config.channels, "channel mismatch"

        kicks = []
        with open(kicksfile, 'r') as file:
            for line in file:
                kicks.append(float(line) * samplerate)
        snares = []
        with open(snaresfile, 'r') as file:
            for line in file:
                snares.append(float(line) * samplerate)

        def time_to_samples(time):
            return int(time * samplerate)

        sequence_in = []
        sequence_out = []
        kicks_index = 0
        snares_index = 0

        for i in range(0, config.sequence_size):
            start = config.sequence_offset + i * config.buffer_size
            exclusive_end = start + config.buffer_size
            sequence_in.append(torch.stack([
                audio[0][start: exclusive_end],
                audio[1][start: exclusive_end]
            ]))

            buffer_has_kick = 0
            while kicks_index < len(kicks) and kicks[kicks_index] < exclusive_end:
                buffer_has_kick = 1.
                kicks_index += 1
            buffer_has_snare = False
            while snares_index < len(snares) and snares[snares_index] < exclusive_end:
                buffer_has_snare = 1.
                snares_index += 1
            sequence_out.append([buffer_has_kick, buffer_has_snare])

        return torch.stack(sequence_in), torch.tensor(sequence_out)
