# this visualizes one chunk's wav and the kicks and snare positions from the respective files
import os
import argparse

import soundfile as sf

import numpy as np

import matplotlib 

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    'audio and timestamps viewer',
    'View a chunk of audio and its kicks and snares', 
    'Either provide a path to a chunk (--chunk) or to a rendered midi (--rendered-midi)'
) 

parser.add_argument('--chunk', help='chunk to view. timestamp files are expected to be named the same as the .ogg audio file but with file endings .kicks or .snares respectively', type=str)
parser.add_argument('--rendered-midi', help='rendered midi to view. timestamp files are expected to be in the same directory and named kicks.txt or snares.txt respectively') 

args = parser.parse_args()

if args.chunk is not None:
    # strip .ogg from chunk name if present
    if args.chunk.endswith('.ogg'):
        args.chunk = args.chunk[:-4]
    audiofile = args.chunk + '.ogg'
    kickfile = args.chunk + '.kicks'
    snarefile = args.chunk + '.snares'
elif args.rendered_midi is not None:
    audiofile = args.rendered_midi
    # get the directory of the rendered midi
    directory = os.path.dirname(args.rendered_midi)
    kickfile = os.path.join(directory, 'kicks.txt')
    snarefile = os.path.join(directory, 'snares.txt')
else: 
    print('No file provided. Provide either a file either with --chunk or --rendered-midi') 
    parser.print_help()
    exit(0)

audio_data, samplerate = sf.read(audiofile) 

if len(audio_data.shape) > 1: 
    audio_data = audio_data[:, 0]

duration = len(audio_data) / samplerate

print('samplerate', samplerate) 

# Create time array
time = np.linspace(0, duration, num=len(audio_data))

# Plot WAV file
plt.plot(time, audio_data, color='black', linewidth=0.1)

# Mark kicks on the time axis
with open(kickfile, 'r') as file:
    for line in file: 
        plt.axvline(x=float(line), ymax=0.5, color='r', linestyle='--')

# Mark snares on the time axis
with open(snarefile, 'r') as file:
    for line in file: 
        plt.axvline(x=float(line), ymax=0.5, color='g', linestyle='--')

# Set labels and title
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Chunk with kicks (red) and snares (green)')

# Display the plot
plt.show()

