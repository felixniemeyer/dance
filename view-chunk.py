# this visualizes one chunk's wav and the kicks and snare positions from the respective files
import argparse

import soundfile as sf

import numpy as np

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('chunk', help='chunk to view', type=str)

args = parser.parse_args()

audio_data, samplerate = sf.read(args.chunk + '.opus')

if len(audio_data.shape) > 1: 
    audio_data = audio_data[:, 0]

duration = len(audio_data) / samplerate

print('samplerate', samplerate) 

# Create time array
time = np.linspace(0, duration, num=len(audio_data))

# Plot WAV file
plt.plot(time, audio_data, color='black', linewidth=0.1)

# Mark kicks on the time axis
with open(args.chunk + '.kicks', 'r') as file:
    for line in file: 
        plt.axvline(x=float(line), ymax=0.5, color='r', linestyle='--')

# Mark snares on the time axis
with open(args.chunk + '.snares', 'r') as file:
    for line in file: 
        plt.axvline(x=float(line), ymax=0.5, color='g', linestyle='--')

# Set labels and title
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Chunk with kicks (red) and snares (green)')

# Display the plot
plt.show()

