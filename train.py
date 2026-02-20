"""
Train a model on the given dataset.
"""

import os
import sys
import time

import argparse
import re

import subprocess

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from dancer_data import DanceDataset

from models.selector import getModelClass, loadModel, getModels, saveModel

import config


parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="model to train. Options: " + ",".join(getModels()))

# in
parser.add_argument("--chunks-path", type=str, default='data/chunks/lakh_clean', help="path to chunks folder")
parser.add_argument("--continue-from", type=int, default=None, help="Epoch number to continue from.")
parser.add_argument("--continue-from-file", type=str, default=None, help="File to continue from. Overrides --continue-from.")

# out
parser.add_argument("--checkpoints-path", type=str, default='checkpoints', help="path to checkpoints folder. Structure: <checkpoints_path>/<tag>/<epoch>.pt")
parser.add_argument("-ci", "--checkpoint-interval", type=int, default=1, help="save checkpoint every n epochs")
parser.add_argument("-t", "--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("-rd", "--learning-rate-decay", type=float, default=0.95, help="learning rate decay")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="batch size")
parser.add_argument("--anticipation-min", type=float, default=0.0, help="minimum anticipation in seconds")
parser.add_argument("--anticipation-max", type=float, default=0.5, help="maximum anticipation in seconds")
parser.add_argument("--warmup-seconds", type=float, default=8.0, help="ignore loss in first N seconds of each sequence")

# misc
parser.add_argument("-d", "--dataset-size", type=int, default=None, help="truncate dataset to this size. For test runs.")
parser.add_argument("--summarize", action='store_true', help="don't log, show only summary at the end")
parser.add_argument("--plot-loss", action='store_true', default=False, help="plot loss after training")

parser.add_argument("--onnx", action='store_true', help="export model to onnx. Stored alongside checkpoints.")

args = parser.parse_args()

if args.anticipation_min < 0 or args.anticipation_max < 0:
    raise ValueError('anticipation values must be non-negative')
if args.anticipation_max < args.anticipation_min:
    raise ValueError('anticipation-max must be >= anticipation-min')
if args.warmup_seconds < 0:
    raise ValueError('warmup-seconds must be >= 0')

if args.tag is None:
    os.system('git log --decorate -n 1 > .git_tag.txt~')
    try:
        with open('.git_tag.txt~', 'r', encoding='utf8') as f:
            line = f.readline()
            match = re.search(r'\(HEAD.*tag: (.*?),', line)
            if match is None:
                raise Exception('no git tag found')
            tag = match.group(1)
            print('using git tag:', args.tag)
            args.tag = tag
    except Exception as e:
        print(e)
        print('Either provide tag by -t or create a git tag')
        sys.exit(0)
    finally:
        os.remove('.git_tag.txt~')

target_epoch = args.num_epochs
if args.continue_from is not None:
    target_epoch += args.continue_from

save_path = f"{args.checkpoints_path}/{args.tag}/"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

batch_size = args.batch_size

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset(args.chunks_path, config.frame_size, config.samplerate)
print('dataset size: ', len(dataset))
print()

data_size = len(dataset)
if args.dataset_size is not None:
    if args.dataset_size > data_size:
        print('dataset size is smaller than requested size')
        exit(0)
    else:
        data_size = args.dataset_size
        indizes = torch.randperm(len(dataset))[:data_size]
        dataset = torch.utils.data.Subset(dataset, indizes)

train_size = int(0.8 * data_size)
val_size = data_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load model from disk if it exists
first_epoch = 0
model = None
optimizer = None

model_class = getModelClass(args.model)
model = model_class().to(device)

if args.continue_from is not None:
    args.continue_from_file = f"{args.checkpoints_path}/{args.tag}/{args.continue_from}.pt"

if args.continue_from_file is not None:
    try:
        model, obj = loadModel(args.continue_from_file)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(obj['optimizer_state_dict'])

        first_epoch = obj['epoch']
        print('loaded checkpoint from', args.continue_from)
    except FileNotFoundError:
        print('no checkpoint found at', args.continue_from)
        sys.exit()
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# scale down learning rate after each epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_rate_decay, last_epoch=-1)

last_epoch = first_epoch + args.num_epochs
frame_duration = config.frame_size / config.samplerate
warmup_frames = int(args.warmup_seconds / frame_duration)


class CustomLoss(nn.Module):
    def __init__(self, warmup_frames_local):
        super().__init__()
        self.warmup_frames = warmup_frames_local

    def forward(self, full_predictions, full_phase_labels):
        # predictions: [batch, frames, 2] => sin/cos
        # labels: [batch, frames] => phase in [0,1)
        start = min(self.warmup_frames, full_phase_labels.shape[1] - 1)

        predictions = full_predictions[:, start:, :]
        labels = full_phase_labels[:, start:]

        target_angles = labels * 2 * torch.pi
        target_vectors = torch.stack([torch.sin(target_angles), torch.cos(target_angles)], dim=-1)

        prediction_norm = torch.norm(predictions, dim=-1, keepdim=True).clamp(min=1e-6)
        normalized_predictions = predictions / prediction_norm

        return torch.mean((normalized_predictions - target_vectors) ** 2)


criterion = CustomLoss(warmup_frames)


def sample_anticipation(batch_size_local, device_local):
    random_values = torch.rand(batch_size_local, device=device_local)
    return args.anticipation_min + random_values * (args.anticipation_max - args.anticipation_min)


def create_future_phase_labels(phase_labels, anticipation):
    # phase_labels shape: [batch, frames] with values in [0,1)
    sequence_length = phase_labels.shape[1]
    frame_indices = torch.arange(sequence_length, device=phase_labels.device, dtype=phase_labels.dtype).unsqueeze(0)
    offset_in_frames = (anticipation / frame_duration).unsqueeze(1)
    lookup_index = frame_indices + offset_in_frames

    low = torch.floor(lookup_index).long().clamp(max=sequence_length - 1)
    high = (low + 1).clamp(max=sequence_length - 1)
    w = lookup_index - low.float()

    low_values = phase_labels.gather(1, low)
    high_values = phase_labels.gather(1, high)

    low_angle = low_values * 2 * torch.pi
    high_angle = high_values * 2 * torch.pi

    x = torch.cos(low_angle) * (1 - w) + torch.cos(high_angle) * w
    y = torch.sin(low_angle) * (1 - w) + torch.sin(high_angle) * w
    angle = torch.atan2(y, x)

    return torch.remainder(angle / (2 * torch.pi), 1.0)


def model_forward(model_local, batch_inputs, anticipation):
    try:
        return model_local(batch_inputs, anticipation=anticipation)
    except TypeError:
        return model_local(batch_inputs)


avg_loss_sum = 0
toal_time = 0

forward_time = 0
loss_calc_time = 0
backpropagation_time = 0
start_calc = 0
to_device_time = 0

loss = None

# Training loop
for epoch in range(first_epoch, last_epoch):

    print(f"\nEpoch {epoch+1} of {last_epoch}")
    print('Training')

    model.train()  # Set the model to training mode
    start_time = time.time()

    for i, (batch_inputs, phase_labels, _) in enumerate(train_loader):

        if not args.summarize:
            print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)

        start_calc = time.time()
        batch_inputs = batch_inputs.to(device)
        phase_labels = phase_labels.to(device)
        anticipation = sample_anticipation(batch_inputs.shape[0], device)
        target_labels = create_future_phase_labels(phase_labels, anticipation)
        to_device_time = time.time() - start_calc

        # Forward pass
        start_calc = time.time()
        outputs, _ = model_forward(model, batch_inputs, anticipation)
        if outputs.shape[-1] != 2:
            raise ValueError('Model must output [sin, cos] per frame. Got shape: ' + str(outputs.shape))
        forward_time += time.time() - start_calc

        # Zero out the gradients
        optimizer.zero_grad()

        # Compute the loss
        start_calc = time.time()
        loss = criterion(outputs, target_labels)
        loss_calc_time += time.time() - start_calc

        # Backpropagation
        start_calc = time.time()
        loss.backward()
        backpropagation_time = time.time() - start_calc

        optimizer.step()

    print()

    scheduler.step()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    print('Evaluation')
    with torch.no_grad():

        total_loss = 0
        total_batches = 0

        for i, (val_inputs, val_phase_labels, _) in enumerate(val_loader):

            if not args.summarize:
                print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)

            val_inputs = val_inputs.to(device)
            val_phase_labels = val_phase_labels.to(device)
            anticipation = sample_anticipation(val_inputs.shape[0], device)
            val_target_labels = create_future_phase_labels(val_phase_labels, anticipation)

            # Forward pass
            val_outputs, _ = model_forward(model, val_inputs, anticipation)
            if val_outputs.shape[-1] != 2:
                raise ValueError('Model must output [sin, cos] per frame. Got shape: ' + str(val_outputs.shape))

            # Compute the loss
            loss = criterion(val_outputs, val_target_labels)

            total_loss += loss
            total_batches += 1

        avg_loss_sum += total_loss / total_batches
        print()
        print(f"Validation loss: {loss:.4f}")

    epoch_duration = time.time() - start_time
    toal_time += epoch_duration
    if not args.summarize:
        print(f"Epoch duration: {epoch_duration:.2f} seconds")

    epoch_1 = epoch + 1
    if (args.checkpoint_interval is not None and (epoch_1 - first_epoch) % args.checkpoint_interval == 0) or epoch_1 == last_epoch:
        file_path = f"{save_path}{epoch_1}.pt"
        print('\nSaving model to', file_path)
        saveModel(file_path, model, {
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'hyperparameters': {
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'anticipation_min': args.anticipation_min,
                'anticipation_max': args.anticipation_max,
                'warmup_seconds': args.warmup_seconds,
                'continued_from': args.continue_from,
            }
        })
        if args.onnx:
            onnx_file_path = f"{save_path}{epoch_1}.onnx"
            model.export_to_onnx(onnx_file_path, device)

    # append loss to log file
    with open(f"{save_path}loss.csv", 'a', encoding='utf8') as f:
        f.write(f"{epoch_1},{loss}\n")

print(f"\nTotal training time: {toal_time:.2f} seconds")
print(f"Total forward pass time: {forward_time:.2f} seconds ({forward_time / toal_time * 100:.2f}%)")
print(f"Total loss calculation time: {loss_calc_time:.2f} seconds ({loss_calc_time / toal_time * 100:.2f}%)")
print(f"Total backpropagation time: {backpropagation_time:.2f} seconds ({backpropagation_time / toal_time * 100:.2f}%)")
print(f"Total to device time: {to_device_time:.2f} seconds ({to_device_time / toal_time * 100:.2f}%)")

epoch_count = last_epoch - first_epoch
print(f"\nAverage time per epoch: {toal_time / epoch_count:.2f} seconds")
print(f"\nAverage validation loss: {avg_loss_sum / epoch_count:.4f}")
print(f"Final validation loss: {loss:.4f}")

if args.plot_loss:
    subprocess.run(['python', 'plot_loss.py', f"{save_path}loss.csv"])
