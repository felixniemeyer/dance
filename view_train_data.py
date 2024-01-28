"""
shows a random sample of the training data
"""

import numpy as np

from dance_data import DanceDataset
from config import samplerate, frame_size

from results_plotter import ResultsPlotter

ds = DanceDataset("data/chunks/lakh_clean", frame_size, samplerate)

while True:
    i = np.random.randint(0, len(ds))

    frames, labels, file = ds[i]

    audio_data = frames.numpy().reshape(-1)

    # TODO: adapt to new plotter interface
    plotter = ResultsPlotter(frame_size, samplerate)
    plotter.plot_wav(frames)

    plotter.plot_events(labels[:, 0], 'kicks', 'red')
    plotter.plot_events(labels[:, 1], 'snares', 'green')

    plotter.finish()
