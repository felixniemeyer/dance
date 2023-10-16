from dance_data import DanceDataset
from config import sample_rate, buffer_size
import numpy as np

import matplotlib.pyplot as plt

ds = DanceDataset("data/chunks/lakh_clean") 


while True:

    i = np.random.randint(0, len(ds))

    audio_data, label = ds[i]

    # convert tensor to numpy array
    audio_data = audio_data.numpy()
    audio_data = audio_data.reshape(-1, 2)
    label = label.numpy()

    samplerate = sample_rate
    print('samplerate', samplerate) 

    # average all channels
    audio_data = np.mean(audio_data, axis=1)

    duration = len(audio_data) / samplerate

    print('duration', duration)

    # Create time array
    time = np.linspace(0, duration, num=len(audio_data))

    # Plot WAV file
    plt.plot(time, audio_data, color='black', linewidth=0.1, label='audio')

    kick_presence = label[:, 0]
    snare_presence = label[:, 1]

    # x values are spaced config.buffer_size apart
    x_values = np.arange(0, len(kick_presence)) * buffer_size / samplerate

    # Plot kicks and snares as curves
    plt.plot(x_values, kick_presence, color='red', linewidth=0.5, label='kicks')
    plt.plot(x_values, snare_presence, color='green', linewidth=0.5, label='snares')

    plt.legend()

    # Set labels and title
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Chunk with kicks (red) and snares (green)')

    # Display the plot
    plt.show()

    plt.close()

    # clear plt
    plt.clf()

