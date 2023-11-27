import numpy as np
import argparse
import subprocess

from dance_data import DanceDataset
from results_plotter import ResultsPlotter
from config import samplerate, frame_size
from models.selector import loadModel

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="*", type=str, help="Checkpoints")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")
args = parser.parse_args()

ds = DanceDataset("data/chunks/lakh_clean", frame_size, samplerate)

while True:
    i = np.random.randint(0, len(ds))

    frames, labels, file = ds[i]

    print("Trying on: ", file)
            
    audio_data = frames.numpy().reshape(-1)

    plotter = ResultsPlotter(frame_size, samplerate, file, len(args.checkpoints))
    plotter.plot_wav(audio_data)

    plotter.plot_event_group('ground truth', labels.numpy().T, ['kicks', 'snares'], ['red', 'green'], is_ground_truth=True)

    for checkpoint in args.checkpoints:
        model, obj = loadModel(checkpoint)
        model.to(args.device_type)

        state = None

        kicks = []
        snares = []
        for frame in frames:
            sequence = frame.unsqueeze(0)
            batch = sequence.unsqueeze(0)

            labels, state = model(batch, state)
            kicks.append(labels[0][0][0].item())
            snares.append(labels[0][0][1].item())

        plotter.plot_event_group(checkpoint, [kicks, snares], ['kicks', 'snares'], ['red', 'green'], 0.01)

    mplayer = subprocess.Popen(["mplayer", "-really-quiet", "-loop", "0", file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    plotter.finish()
    mplayer.kill()

