import numpy as np

import matplotlib.pyplot as plt

from config import sample_rate, buffer_size

# this plots an audio waveform and a presence array

class ResultsPlotter: 
    def __init__(self, buffersize=buffer_size, samplerate=sample_rate): 
        self.buffersize = buffersize
        self.samplerate = samplerate


    def plot_wav(self, waveform): 
        self.samplesize = waveform.shape[0]
        self.duration = self.samplesize / self.samplerate

        time = np.linspace(0, self.duration, num=self.samplesize)

        _, ax1 = plt.subplots()
        # set title
        ax1.set_title('audio waveform and audio event presence')
        ax1.plot(time, waveform, color='black', linewidth=0.1)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Waveform amplitude')
        ax1.legend

        self.ax2 = ax1.twinx()
        self.ax2.set_ylabel('Audio event presence')

        print("self.samplesize", self.samplesize)
        print("buffersize", self.buffersize)
        self.ax2_xvalues = np.arange(0, self.samplesize // self.buffersize) * self.buffersize / self.samplerate

        print("ax2_xvalues", self.ax2_xvalues)

    def plot_presence(self, presence, name, color): 
        if self.ax2 is None: 
            raise Exception("You need to plot the waveform first")

        self.ax2.plot(self.ax2_xvalues, presence, color=color, linewidth=3, label=name)

    def finish(self): 
        self.ax2.legend()
        plt.show()
        plt.close()
