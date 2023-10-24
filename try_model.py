from dance_data import DanceDataset
from config import samplerate, buffer_size
import numpy as np

import argparse

from results_plotter import ResultsPlotter

from config import samplerate, buffer_size


parser = argparse.ArgumentParser()

parser.add_argument("checkpoint", type=str, help="Checkpoint to use")

parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")

args = parser.parse_args()

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
