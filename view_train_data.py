from dance_data import DanceDataset
from config import samplerate, buffer_size
import numpy as np

from results_plotter import ResultsPlotter

from config import samplerate, buffer_size

ds = DanceDataset("data/chunks/lakh_clean", buffer_size, samplerate, print_filename=True)

while True:
    i = np.random.randint(0, len(ds))

    buffers, labels = ds[i]

    audio_data = buffers.numpy().reshape(-1)

    plotter = ResultsPlotter(buffer_size, samplerate)
    plotter.plot_wav(buffers)

    plotter.plot_events(labels[:, 0], 'kicks', 'red')
    plotter.plot_events(labels[:, 1], 'snares', 'green')

    plotter.finish()
