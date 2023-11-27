# this visualizes one chunk's wav and the kicks and snare positions from the respective files
import argparse
import os

import soundfile as sf

import numpy as np

from dance_data import loadEventsAsLabels
from results_plotter import ResultsPlotter

from config import frame_size, samplerate 

parser = argparse.ArgumentParser(
    'audio and timestamps viewer',
    'View an audio file and its kicks and snares presences', 
)

parser.add_argument('audiofile', help='The audiofile to open. A kick_presence and snare_presence file is expected to be in the same directory.', type=str)
# support any number of kicks and snares files
parser.add_argument('eventfile', nargs="*", help='The eventfile to open. Default: <af>.events', type=str)

args = parser.parse_args()

# strip .ogg from audiofile name if present
if args.audiofile.endswith('.ogg'):
    args.audiofile = args.audiofile[:-4]
audiofile = args.audiofile + '.ogg'

if len(args.eventfile) == 0:
    args.eventfile = [args.audiofile + '.events']

audio_data, file_samplerate = sf.read(audiofile)

assert samplerate == file_samplerate, "sample rate mismatch. Project wide setting is " + str(samplerate) + ", but file has " + str(file_samplerate) + " samples per second"

plotter = ResultsPlotter(frame_size, samplerate)
plotter.plot_wav(audio_data)

frame_count = audio_data.shape[0] // frame_size

just_one = len(args.eventfile) == 1

for eventfile in args.eventfile:
    labels = loadEventsAsLabels(eventfile, frame_count, samplerate, frame_size)
    print(labels.shape)

    # swap dimensions
    labels = np.swapaxes(labels, 0, 1)

    color = np.random.rand(3,)
    plotter.plot_event_group(eventfile, labels, ['kicks', 'snares'], ['red', 'green'], 0.01, just_one)

plotter.finish()
