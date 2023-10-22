from dance_data import DanceDataset
from config import sample_rate, buffer_size
import numpy as np

from results_plotter import ResultsPlotter

ds = DanceDataset("data/chunks/lakh_clean", kick_half_life=0.02, snare_half_life=0.02, print_filename=True)


while True:

    i = np.random.randint(0, len(ds))

    audio_data, label = ds[i]

    # convert tensor to numpy array
    audio_data = audio_data.numpy()
    audio_data = audio_data.reshape(-1) 

    label = label.numpy()

    samplerate = sample_rate
    print('samplerate', samplerate) 

    # average all channels

    duration = len(audio_data) / samplerate

    print()
    print('duration', duration)

    plotter = ResultsPlotter(buffer_size, samplerate)
    plotter.plot_wav(audio_data)

    kick_presence = label[:, 0]
    snare_presence = label[:, 1]

    plotter.plot_presence(kick_presence, 'kicks', 'red')
    plotter.plot_presence(snare_presence, 'snares', 'green')

    plotter.finish()
