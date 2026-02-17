import torch

from models.selector import loadModel

import config

import argparse

import torchaudio

parser = argparse.ArgumentParser()

parser.add_argument("checkpoint", type=str, help="Checkpoint to use")
parser.add_argument("input", type=str, help="Audio file")

parser.add_argument("-o", "--outfile_prefix", type=str, help="out file prefix. Default: input file name")
parser.add_argument("-a", "--anticipation", type=float, default=0.0, help="anticipation in seconds")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")

args = parser.parse_args()

# read audio file
audiofilename = args.input

filebase = ""
if args.input[-4:] == '.ogg':
    filebase = args.input[:-4]
else:
    print('unsupported audio format')
    exit()

if args.outfile_prefix is None:
    args.outfile_prefix = filebase

phasefile = args.outfile_prefix + '.phase_prediction'

audio, samplerate = torchaudio.load(args.input)

assert samplerate == config.samplerate, "sample rate mismatch"

audio = audio.mean(0)

print('input audio shape:', audio.shape)

length = len(audio)
frames_in_file = length // config.frame_size
offset = length % config.frame_size

# Initialize the model
device = torch.device(args.device_type)

# Load model from disk if it exists
model, obj = loadModel(args.checkpoint)
model.to(device)
model.eval()

anticipation = torch.tensor([args.anticipation], device=device, dtype=torch.float32)

print('analyzing audio file...')
state = None
with torch.no_grad():
    with open(phasefile, 'w', encoding='utf8') as pf:
        for i in range(frames_in_file):
            start = offset + i * config.frame_size
            end = start + config.frame_size
            frame = audio[start:end]

            sequence = frame.unsqueeze(0)
            batch = sequence.unsqueeze(0)

            model_input = batch.to(device)

            try:
                output, state = model(model_input, anticipation=anticipation, state=state)
            except TypeError:
                output, state = model(model_input, state)
            phase = output[0][0][0].item()

            pf.write(str(phase) + '\n')

            if i % 100 == 0:
                print(f"\r{i}/{frames_in_file} ({i/frames_in_file*100:.2f}%)", end="\r")

print(f'{frames_in_file}/{frames_in_file} (100.00%), done.')
