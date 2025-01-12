import torch

from models.selector import getModelClass, loadModel

import config

import argparse

import torchaudio

parser = argparse.ArgumentParser()

parser.add_argument("checkpoint", type=str, help="Checkpoint to use")
parser.add_argument("input", type=str, help="Audio file")

parser.add_argument("-o", "--outfile_prefix", type=str, help="out file prefix. Default: input file name")

parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
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

if args.outfile_prefix == None:
    args.outfile_prefix = filebase

kicksfile = args.outfile_prefix + '.kick_presence'
snaresfile = args.outfile_prefix + '.snare_presence'

audio, samplerate = torchaudio.load(args.input)

assert samplerate == config.samplerate, "sample rate mismatch"

audio = audio.mean(0)

print('input audio shape:', audio.shape) 

lenght = len(audio)
frames_in_file = lenght // config.frame_size
offset = lenght % config.frame_size
frame_duration = config.frame_size / samplerate

# Initialize the model
device = torch.device(args.device_type)

# Load model from disk if it exists
model, obj = loadModel(args.checkpoint)
model.to(device)

print('analyzing audio file...')
with open(kicksfile, 'w') as kf, open(snaresfile, 'w') as sf:
    for i in range(frames_in_file):
        start = offset + i * config.frame_size
        end = start + config.frame_size
        frame = audio[start:end]

        sequence = frame.unsqueeze(0)
        batch = sequence.unsqueeze(0)

        model_input = batch.to(device)

        labels = model(model_input)[0][0]

        # find the index of the highest value in the output vector

        kf.write(str(labels[0].item()) + '\n')
        sf.write(str(labels[1].item()) + '\n')

        if i % 100 == 0:
            print(f"\r{i}/{frames_in_file} ({i/frames_in_file*100:.2f}%)", end="\r")

print(f'{frames_in_file}/{frames_in_file} (100.00%), done.')

