"""
Plots events on a waveform
"""

import numpy as np

import matplotlib.pyplot as plt

# this plots an audio waveform and a presence array

TITLE = 'audio waveform and audio event presence'

class ResultsPlotter: 
    def __init__(self, waveform, framesize, samplerate, title=TITLE, model_count=0 ):
        self.framesize = framesize
        self.samplerate = samplerate
        self.title = title
        self.model_count = model_count

        self.samplesize = waveform.shape[0]
        self.duration = self.samplesize / self.samplerate

        time = np.linspace(0, self.duration, num=self.samplesize)

        self.fig, self.axes = plt.subplots(1 + self.model_count, 1, sharex=True)

        if self.model_count == 0:
            self.axes = [self.axes]

        ax = self.axes[0]
        ax.set_title(self.title)
        ax.plot(time, waveform, color='black', linewidth=0.1)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Waveform amplitude')

        self.axis_index = 1

        self.frames_per_file = self.samplesize // self.framesize
        self.frame_duration = self.framesize / self.samplerate
        self.xvalues = np.arange(0, self.frames_per_file) * self.frame_duration

    def plot_event_group(self, name, values, event_names, colors, threshold=0.01, is_ground_truth=False ):
        if len(self.axes) == 0:
            raise Exception("You need to plot the waveform first")

        if is_ground_truth:
            ax = self.axes[0].twinx()
            ax.set_ylabel('Ground truth')
        else:
            if self.axis_index >= len(self.axes):
                raise Exception("You have already plotted the maximum number of models")
            ax = self.axes[self.axis_index]
            self.axis_index += 1

        ax.set_ylabel(name)

        for i, event_name in enumerate(event_names):
            self.plot_event(values[i], event_name, colors[i], threshold, ax)

    def plot_event(self, values, event_name, color, threshold, ax):
        # combine self.ax2_xvalues and intensity into pairs for filtering
        pairs = np.array([self.xvalues, values])
        # filter out all pairs where y < 0.01
        pairs = pairs[:, pairs[1] > threshold]

        ax.bar(pairs[0], pairs[1], color=color, label=event_name, width=self.frame_duration, alpha=0.5, align='edge')

    def finish(self):
        for ax in self.axes:
            # show legend
            ax.legend()

        plt.show()
        plt.close()
