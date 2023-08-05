import torch
import os

from dancer_model import DancerModel

import config

import argparse

import torchaudio

parser = argparse.ArgumentParser()

parser.add_argument("checkpoint", type=str, help="Checkpoint to use")
parser.add_argument("input", type=str, help="Audio file")

parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")

args = parser.parse_args()

# read audio file
audiofilename = args.input
filebase = ""
if args.input[-4:] == '.ogg': 
    filebase = args.input[:-4]
else: 
    print('unsupported audio format') 
    exit()

kicksfile = filebase + '.kick_presence'
snaresfile = filebase + '.snare_presence'

audio, samplerate = torchaudio.load(args.input)

assert samplerate == config.sample_rate, "sample rate mismatch"
assert audio.shape[0] == config.channels, "channel mismatch"

print('input audio shape:', audio.shape) 

lenght = audio.shape[1]
buffers_in_file = lenght // config.buffer_size
offset = lenght % config.buffer_size
buffer_duration = config.buffer_size / samplerate

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from disk if it exists
model = DancerModel().to(device)

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])

kicks = []
snares = []

print('analyzing audio file...')
with open(kicksfile, 'w') as kf, open(snaresfile, 'w') as sf:
    for i in range(buffers_in_file):
        start = offset + i * config.buffer_size
        end = start + config.buffer_size
        buffer = audio[:, start:end]

        sequence = buffer.unsqueeze(0)
        batch = sequence.unsqueeze(0)

        model_input = batch.to(device)

        output = model(model_input)[0][0]

        # find the index of the highest value in the output vector

        kf.write(str(output[0].item()) + '\n')
        sf.write(str(output[1].item()) + '\n')

        if i % 100 == 0:
            print(f"\r{i}/{buffers_in_file} ({i/buffers_in_file*100:.2f}%)", end="\r")

print(f'{buffers_in_file}/{buffers_in_file} (100.00%), done.')
