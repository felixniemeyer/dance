import torch

from models.selector import loadModel

import config

import argparse

import soundfile
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("checkpoint", type=str, help="Checkpoint to use")
parser.add_argument("input", type=str, help="Audio file")

parser.add_argument("-o", "--outfile_prefix", type=str, help="out file prefix. Default: input file name")
parser.add_argument("-a", "--anticipation", type=float, default=0.0, help="anticipation in seconds")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")

args = parser.parse_args()

# read audio file
filebase = ""
if args.input[-4:] == '.ogg':
    filebase = args.input[:-4]
else:
    print('unsupported audio format')
    exit()

if args.outfile_prefix is None:
    args.outfile_prefix = filebase

phasefile = args.outfile_prefix + '.phase_prediction'

audio, samplerate = soundfile.read(args.input, dtype='float32')

assert samplerate == config.samplerate, "sample rate mismatch"

if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = torch.from_numpy(audio)

print('input audio shape:', audio.shape)

length = len(audio)
frames_in_file = length // config.frame_size
offset = length % config.frame_size

# Initialize the model
device = torch.device(args.device_type)

# Load model from disk if it exists
model, _obj = loadModel(args.checkpoint)
model.to(device)
model.eval()

anticipation = torch.tensor([args.anticipation], device=device, dtype=torch.float32)

sequence = audio[offset:offset + frames_in_file * config.frame_size].reshape(frames_in_file, config.frame_size)
batch = sequence.unsqueeze(0).to(device)

print('analyzing audio file...')
with torch.no_grad():
    try:
        output, _ = model(batch, anticipation=anticipation)
    except TypeError:
        output, _ = model(batch)

    if output.shape[-1] == 2:
        angles = torch.atan2(output[0, :, 0], output[0, :, 1])
        phases = torch.remainder(angles / (2 * torch.pi), 1.0)
    elif output.shape[-1] == 1:
        phases = output[0, :, 0]
    else:
        raise ValueError('Unsupported output shape: ' + str(output.shape))

    with open(phasefile, 'w', encoding='utf8') as pf:
        for i in range(frames_in_file):
            pf.write(str(phases[i].item()) + '\n')

print(f'{frames_in_file}/{frames_in_file} (100.00%), done.')
