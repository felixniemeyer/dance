import numpy as np

import matplotlib.pyplot as plt

# this plots an audio waveform and a presence array

class ResultsPlotter: 
    def __init__(self, buffersize, samplerate): 
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

        self.buffers_per_file = self.samplesize // self.buffersize
        self.buffer_duration = self.buffersize / self.samplerate
        self.ax2_xvalues = np.arange(0, self.buffers_per_file) * self.buffer_duration + self.buffer_duration # shift bars to the right 

    def plot_events(self, intensities, name, color): 
        if self.ax2 is None: 
            raise Exception("You need to plot the waveform first")
        
        # combine self.ax2_xvalues and intensity into pairs for filtering
        pairs = np.array([self.ax2_xvalues, intensities])
        
        # filter out all pairs where y < 0.01
        pairs = pairs[:, pairs[1] > 0.01]

        self.ax2.bar(pairs[0], pairs[1], color=color, label=name, width=self.buffer_duration, alpha=0.5, align='edge')

    def finish(self): 
        self.ax2.legend()
        plt.show()
        plt.close()
