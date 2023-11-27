from dance_data import DanceDataset
from config import samplerate, frame_size
import numpy as np

from results_plotter import ResultsPlotter

from config import samplerate, frame_size

ds = DanceDataset("data/chunks/lakh_clean", frame_size, samplerate, print_filename=True)

while True:
    i = np.random.randint(0, len(ds))

    frames, labels = ds[i]

    audio_data = frames.numpy().reshape(-1)

    plotter = ResultsPlotter(frame_size, samplerate)
    plotter.plot_wav(frames)

    plotter.plot_events(labels[:, 0], 'kicks', 'red')
    plotter.plot_events(labels[:, 1], 'snares', 'green')

    plotter.finish()
