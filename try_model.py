import numpy as np
import argparse

from dance_data import DanceDataset
from results_plotter import ResultsPlotter
from config import samplerate, buffer_size
from models.selector import loadModel

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="*", type=str, help="Checkpoints")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")
args = parser.parse_args()

ds = DanceDataset("data/chunks/lakh_clean", buffer_size, samplerate, print_filename=True)

print('showing random chunks from dataset')
print('for each applying the following checkpoints:' + '\n'.join(args.checkpoints))

while True:
    i = np.random.randint(0, len(ds))

    buffers, labels = ds[i]

    audio_data = buffers.numpy().reshape(-1)

    plotter = ResultsPlotter(buffer_size, samplerate)
    plotter.plot_wav(audio_data)

    plotter.plot_events(labels[:, 0], 'groundtruth kicks', 'red')
    plotter.plot_events(labels[:, 1], 'groundtruth snares', 'green')

    for checkpoint in args.checkpoints:
        model, obj = loadModel(checkpoint)
        model.to(args.device_type)

        kicks = []
        snares = []
        for buffer in buffers:
            sequence = buffer.unsqueeze(0)
            batch = sequence.unsqueeze(0)

            labels = model(batch)[0][0]
            kicks.append(labels[0].item())
            snares.append(labels[1].item())

        color = np.random.rand(3,)
        color = color / (1 + color.mean())
        plotter.plot_events(kicks, 'kicks ' + checkpoint, 'blue', 0.01)
        plotter.plot_events(snares, 'snares ' + checkpoint, 'yellow', 0.01)  
    
    plotter.finish()

