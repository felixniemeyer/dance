# this visualizes one chunk's wav and the kicks and snare positions from the respective files
import argparse

import soundfile as sf

import numpy as np

import matplotlib.pyplot as plt

from config import buffer_size, sample_rate

parser = argparse.ArgumentParser(
    'audio and timestamps viewer',
    'View an audio file and its kicks and snares presences', 
) 

parser.add_argument('audiofile', help='The audiofile to open. A kick_presence and snare_presence file is expected to be in the same directory', type=str)

args = parser.parse_args()

# strip .ogg from audiofile name if present
if args.audiofile.endswith('.ogg'):
    args.audiofile = args.audiofile[:-4]
audiofile = args.audiofile + '.ogg'
kickfile = args.audiofile + '.kick_presence'
snarefile = args.audiofile + '.snare_presence'

audio_data, samplerate = sf.read(audiofile) 

assert samplerate == sample_rate, "sample rate mismatch. Project wide setting is " + str(sample_rate) + ", but file has " + str(samplerate) + " samples per second"

# average all channels
audio_data = np.mean(audio_data, axis=1)

duration = len(audio_data) / samplerate

print('samplerate', samplerate) 

# Create time array
time = np.linspace(0, duration, num=len(audio_data))

# Plot WAV file
plt.plot(time, audio_data, color='black', linewidth=0.1)

kick_presence = []
snare_presence = []
# Mark kicks on the time axis
with open(kickfile, 'r') as file:
    for line in file: 
        kick_presence.append(float(line))

# Mark snares on the time axis
with open(snarefile, 'r') as file:
    for line in file: 
        snare_presence.append(float(line))

# x values are spaced config.buffer_size apart
x_values = np.arange(0, len(kick_presence)) * buffer_size / samplerate

# Plot kicks and snares as curves
plt.plot(x_values, kick_presence, color='red', linewidth=0.5)
plt.plot(x_values, snare_presence, color='green', linewidth=0.5)


# Set labels and title
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Chunk with kicks (red) and snares (green)')

# Display the plot
plt.show()

