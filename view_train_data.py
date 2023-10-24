from dance_data import DanceDataset
from config import samplerate, buffer_size
import numpy as np

from results_plotter import ResultsPlotter

from config import samplerate, buffer_size

ds = DanceDataset("data/chunks/lakh_clean", buffer_size, samplerate, print_filename=True)

while True:

    i = np.random.randint(0, len(ds))

    audio_data, label = ds[i]

    # convert tensor to numpy array
    audio_data = audio_data.numpy()
    audio_data = audio_data.reshape(-1) 

    label = label.numpy()

    # average all channels

    duration = len(audio_data) / samplerate

    plotter = ResultsPlotter(buffer_size, samplerate)
    plotter.plot_wav(audio_data)

    kicks = label[:, 0]
    snares = label[:, 1]

    plotter.plot_events(kicks, 'kicks', 'red')
    plotter.plot_events(snares, 'snares', 'green')

    plotter.finish()
