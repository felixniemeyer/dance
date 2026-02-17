import os

import numpy as np
import argparse
import subprocess
import matplotlib.pyplot as plt

from dancer_data import DanceDataset
from config import samplerate, frame_size
from models.selector import loadModel

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="common path to all checkpoints")
parser.add_argument("checkpoints", nargs="*", type=str, help="Checkpoints")
parser.add_argument("-p", "--dataset-path", type=str, default="data/chunks/lakh_clean", help="path to dataset")
parser.add_argument("-d", "--device-type", type=str, default="cpu", help="device type (cpu or cuda)")
parser.add_argument("-a", "--anticipation", type=float, default=0.0, help="anticipation in seconds")
args = parser.parse_args()

ds = DanceDataset(args.dataset_path, frame_size, samplerate)

while True:
    i = np.random.randint(0, len(ds))

    frames, labels, file = ds[i]

    print("Trying on: ", file)

    audio_data = frames.numpy().reshape(-1)
    time_audio = np.linspace(0, len(audio_data) / samplerate, num=len(audio_data))

    frame_duration = frame_size / samplerate
    xvalues = np.arange(0, len(labels)) * frame_duration

    fig, axes = plt.subplots(1 + len(args.checkpoints), 1, sharex=True)
    if len(args.checkpoints) == 0:
        axes = [axes]

    axes[0].plot(time_audio, audio_data, color='black', linewidth=0.1)
    gt = axes[0].twinx()
    gt.plot(xvalues, labels.numpy(), color='tab:blue', linewidth=1, label='ground truth phase')
    gt.set_ylim(0, 1)
    gt.legend(loc='upper right')

    for ci, checkpoint in enumerate(args.checkpoints):
        cpp = os.path.join(args.path, checkpoint)
        model, _obj = loadModel(cpp)
        model.to(args.device_type)
        model.eval()

        anticipation = np.array([args.anticipation], dtype=np.float32)
        anticipation_tensor = None

        state = None
        predicted_phase = []
        for frame in frames:
            sequence = frame.unsqueeze(0)
            batch = sequence.unsqueeze(0)

            if anticipation_tensor is None:
                import torch
                anticipation_tensor = torch.tensor(anticipation, device=args.device_type)

            try:
                output, state = model(batch.to(args.device_type), anticipation=anticipation_tensor, state=state)
            except TypeError:
                output, state = model(batch.to(args.device_type), state)
            predicted_phase.append(output[0][0][0].item())

        ax = axes[ci + 1]
        ax.plot(xvalues, predicted_phase, color='tab:orange', linewidth=1, label=checkpoint)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

    mplayer = subprocess.Popen(["mplayer", "-really-quiet", "-loop", "0", file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    plt.show()
    plt.close()
    mplayer.kill()
