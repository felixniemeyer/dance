import torch

from models.selector import getModelClass

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

assert samplerate == config.sample_rate, "sample rate mismatch"

audio = audio.mean(0)

print('input audio shape:', audio.shape) 

lenght = len(audio)
buffers_in_file = lenght // config.buffer_size
offset = lenght % config.buffer_size
buffer_duration = config.buffer_size / samplerate

# Initialize the model
device = torch.device(args.device_type)

# Load model from disk if it exists

checkpoint = torch.load(args.checkpoint)
model_type = ''
if "model_type" in checkpoint: 
    model_type = checkpoint['model_type']
else:
    # determine model type from file name
    if 'cnn_only' in args.checkpoint:
        model_type = 'cnn_only'
    elif 'rnn_only' in args.checkpoint:
        model_type = 'rnn_only'
    elif 'cnn_and_rnn' in args.checkpoint:
        model_type = 'cnn_and_rnn'
    elif 'cnn_and_rnn_and_funnel' in args.checkpoint:
        model_type = 'cnn_and_rnn_and_funnel'
    else:
        print("don't know which model type to use")
        exit()

model = None
modelClass = getModelClass(model_type)

model = modelClass().to(device)

model.load_state_dict(checkpoint['model_state_dict'])

kicks = []
snares = []

print('analyzing audio file...')
with open(kicksfile, 'w') as kf, open(snaresfile, 'w') as sf:
    for i in range(buffers_in_file):
        start = offset + i * config.buffer_size
        end = start + config.buffer_size
        buffer = audio[start:end]

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

