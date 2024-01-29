import torch

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

kicksfile = filebase + '.kicks'
snaresfile = filebase + '.snares'

audio, samplerate = torchaudio.load(args.input)

assert samplerate == config.sample_rate, "sample rate mismatch"
assert audio.shape[0] == config.channels, "channel mismatch"

print('input audio shape:', audio.shape) 

lenght = audio.shape[1]
frames_in_file = lenght // config.frame_size
offset = lenght % config.frame_size
frame_duration = config.frame_size / samplerate

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from disk if it exists
model = DancerModel().to(device)

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])

kicks = []
snares = []

# feed the rnn one frame at a time
print('analyzing audio file...')
for i in range(frames_in_file):
    start = offset + i * config.frame_size
    end = start + config.frame_size
    frame = audio[:, start:end]

    sequence = frame.unsqueeze(0)
    batch = sequence.unsqueeze(0)

    model_input = batch.to(device)

    output = model(model_input)[0][0] 

    # find the index of the highest value in the output vector

    t = (i + 0.5) * frame_duration
    if output[0] > 0.8:
        kicks.append(t)
    if output[1] > 0.8:
        snares.append(t)

    if i % 100 == 0:
        print(f"\r{i}/{frames_in_file} ({i/frames_in_file*100:.2f}%)", end="\r")

print(f'{frames_in_file}/{frames_in_file} (100.00%), done.')

print(kicksfile)
# write relevant kicks and snares to file
with open(kicksfile, 'w') as f:
    for kick in kicks:
        f.write(str(kick) + '\n')

with open(snaresfile, 'w') as f:
    for snare in snares:
        f.write(str(snare) + '\n')

