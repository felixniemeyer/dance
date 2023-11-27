# this visualizes one chunk's wav and the kicks and snare positions from the respective files
import argparse
import os

import soundfile as sf

import numpy as np

from results_plotter import ResultsPlotter

from config import frame_size, samplerate 

parser = argparse.ArgumentParser(
    'audio and timestamps viewer',
    'View an audio file and its kicks and snares presences', 
) 

parser.add_argument('audiofile', help='The audiofile to open. A kick_presence and snare_presence file is expected to be in the same directory.', type=str)
# support any number of kicks and snares files
parser.add_argument('eventfiles', nargs="*", help='The eventfiles to open. Default: <af>.kick_presence and <af>.snare_presence where <af> is the audio file name.', type=str)

args = parser.parse_args()

# strip .ogg from audiofile name if present
if args.audiofile.endswith('.ogg'):
    args.audiofile = args.audiofile[:-4]
audiofile = args.audiofile + '.ogg'

eventfiles = {}
if len(args.eventfiles) == 0:
    eventfiles["kicks"] = args.audiofile + '.kick_presence'
    eventfiles["snares"] = args.audiofile + '.snare_presence'
else:
    for eventfile in args.eventfiles:
        basename = os.path.basename(eventfile)
        eventfiles[basename] = eventfile

audio_data, samplerate = sf.read(audiofile) 

assert samplerate == samplerate, "sample rate mismatch. Project wide setting is " + str(samplerate) + ", but file has " + str(samplerate) + " samples per second"

# average all channels
audio_data = np.mean(audio_data, axis=1)

plotter = ResultsPlotter(frame_size, samplerate)
plotter.plot_wav(audio_data)

for name, eventfile in eventfiles.items():
    event_presence = []
    # Mark kicks on the time axis
    with open(eventfile, 'r') as file:
        for line in file: 
            event_presence.append(float(line))
    # make random color
    color = np.random.rand(3,)
    plotter.plot_events(event_presence, name, color)

plotter.finish()
