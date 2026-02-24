"""
Train a model on the given dataset.
"""

import os
import sys
import time

import argparse
import re

import subprocess

import mlflow
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
parser.add_argument("-ci", "--checkpoint-interval", type=int, default=5, help="save checkpoint every n epochs")
parser.add_argument("-t", "--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")
parser.add_argument("--experiment", type=str, default="dance", help="MLflow experiment name. All runs in the same experiment are comparable.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("-rd", "--learning-rate-decay", type=float, default=0.95, help="learning rate decay")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="batch size")
parser.add_argument("--warmup-seconds", type=float, default=8.0, help="ignore loss in first N seconds of each sequence")
parser.add_argument("--fft-frames", type=int, default=None, help="mel frontend: FFT window in frames (overrides model default)")
parser.add_argument("--rate-loss-weight", type=float, default=0.1, help="weight for phase_rate MSE loss term (0 disables)")

# misc
parser.add_argument("-d", "--dataset-size", type=int, default=None, help="truncate dataset to this size. For test runs.")
parser.add_argument("--summarize", action='store_true', help="don't log, show only summary at the end")
parser.add_argument("--plot-loss", action='store_true', default=False, help="plot loss after training")

parser.add_argument("--onnx", action='store_true', help="export model to onnx. Stored alongside checkpoints.")

args = parser.parse_args()

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

# Build model kwargs from CLI overrides
model_kwargs = {}
if args.fft_frames is not None:
    model_kwargs['fft_frames'] = args.fft_frames

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

train_size = max(1, int(0.8 * data_size))
val_size = data_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

num_workers = max(1, os.cpu_count() - 1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, multiprocessing_context='fork')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True, multiprocessing_context='fork') if val_size > 0 else None

# Load model from disk if it exists
first_epoch = 0
model_class = getModelClass(args.model)

if args.continue_from is not None:
    args.continue_from_file = f"{args.checkpoints_path}/{args.tag}/{args.continue_from}.pt"

if args.continue_from_file is not None:
    try:
        model, obj = loadModel(args.continue_from_file)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(obj['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

        first_epoch = obj['epoch']
        print('loaded checkpoint from', args.continue_from_file)
        print(f'learning rate set to {args.learning_rate}')
    except FileNotFoundError:
        print('no checkpoint found at', args.continue_from_file)
        sys.exit()
else:
    try:
        model = model_class(**model_kwargs).to(device)
    except TypeError as e:
        print(f'Warning: model does not accept kwargs {model_kwargs}: {e}')
        model = model_class().to(device)
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


def model_forward(model_local, batch_inputs):
    return model_local(batch_inputs)


mlflow.set_experiment(args.experiment)
mlflow_run = mlflow.start_run(run_name=args.tag)
mlflow.log_params({
    'model':          args.model,
    'learning_rate':  args.learning_rate,
    'lr_decay':       args.learning_rate_decay,
    'batch_size':     args.batch_size,
    'warmup_seconds': args.warmup_seconds,
    'continued_from': args.continue_from,
    'chunks_path':    args.chunks_path,
    'dataset_size':   len(dataset),
})
if hasattr(model, 'hparams'):
    mlflow.log_params(model.hparams)

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
    train_total_loss = 0
    train_num_batches = 0

    train_rate_loss_total = 0

    for i, (batch_inputs, phase_labels, rate_labels, _) in enumerate(train_loader):

        if not args.summarize:
            print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)

        start_calc = time.time()
        batch_inputs = batch_inputs.to(device, non_blocking=True)
        phase_labels = phase_labels.to(device, non_blocking=True)
        rate_labels  = rate_labels.to(device, non_blocking=True)
        input_frames  = batch_inputs.shape[1]
        target_labels = phase_labels[:, :input_frames]
        to_device_time = time.time() - start_calc

        # Forward pass
        start_calc = time.time()
        outputs, _ = model_forward(model, batch_inputs)
        phase_outputs = outputs[:, :, :2]   # sin/cos
        forward_time += time.time() - start_calc

        # Zero out the gradients
        optimizer.zero_grad()

        # Compute the loss
        start_calc = time.time()
        loss_phase = criterion(phase_outputs, target_labels)
        loss = loss_phase
        if args.rate_loss_weight > 0 and outputs.shape[-1] > 2:
            warmup = min(warmup_frames, input_frames - 1)
            rate_outputs = outputs[:, warmup:, 2]
            gt_rate = rate_labels[:, :input_frames][:, warmup:]
            loss_rate = ((rate_outputs - gt_rate) ** 2).mean()
            loss = loss_phase + args.rate_loss_weight * loss_rate
            train_rate_loss_total += loss_rate.item()
        loss_calc_time += time.time() - start_calc

        # Backpropagation
        start_calc = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        backpropagation_time = time.time() - start_calc

        optimizer.step()
        train_total_loss += loss.item()
        train_num_batches += 1

    train_loss = train_total_loss / train_num_batches
    train_rate_loss = train_rate_loss_total / train_num_batches
    val_loss = 0.0
    val_rate_loss = 0.0
    print()
    print(f"Training loss:   {train_loss:.4f}")

    scheduler.step()

    # Validation after each epoch
    if val_loader is not None:
        model.eval()
        print('Evaluation')
        with torch.no_grad():

            total_loss = 0
            total_batches = 0

            val_rate_loss_total = 0

            for i, (val_inputs, val_phase_labels, val_rate_labels, _) in enumerate(val_loader):

                if not args.summarize:
                    print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)

                val_inputs = val_inputs.to(device, non_blocking=True)
                val_phase_labels = val_phase_labels.to(device, non_blocking=True)
                val_rate_labels  = val_rate_labels.to(device, non_blocking=True)
                val_input_frames = val_inputs.shape[1]
                val_target_labels = val_phase_labels[:, :val_input_frames]

                val_outputs, _ = model_forward(model, val_inputs)
                val_phase_outputs = val_outputs[:, :, :2]

                loss = criterion(val_phase_outputs, val_target_labels)
                if args.rate_loss_weight > 0 and val_outputs.shape[-1] > 2:
                    warmup = min(warmup_frames, val_input_frames - 1)
                    val_rate_out = val_outputs[:, warmup:, 2]
                    val_gt_rate  = val_rate_labels[:, :val_input_frames][:, warmup:]
                    val_rate_loss_batch = ((val_rate_out - val_gt_rate) ** 2).mean()
                    val_rate_loss_total += val_rate_loss_batch.item()

                total_loss += loss
                total_batches += 1

            val_loss = total_loss / total_batches
            val_rate_loss = val_rate_loss_total / total_batches
            avg_loss_sum += val_loss
            print()
            print(f"Validation loss: {val_loss:.4f}")
    else:
        print('Validation skipped (dataset too small for split)')

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
                'warmup_seconds': args.warmup_seconds,
                'continued_from': args.continue_from,
            }
        })
        if args.onnx:
            onnx_file_path = f"{save_path}{epoch_1}.onnx"
            model.export_to_onnx(onnx_file_path, device)

    # append loss to log file
    with open(f"{save_path}loss.csv", 'a', encoding='utf8') as f:
        f.write(f"{epoch_1},{train_loss},{val_loss},{train_rate_loss}\n")

    mlflow.log_metrics({
        'train_loss':      float(train_loss),
        'val_loss':        float(val_loss),
        'train_rate_loss': float(train_rate_loss),
        'val_rate_loss':   float(val_rate_loss) if val_loader is not None else 0.0,
    }, step=epoch_1)

print(f"\nTotal training time: {toal_time:.2f} seconds")
print(f"Total forward pass time: {forward_time:.2f} seconds ({forward_time / toal_time * 100:.2f}%)")
print(f"Total loss calculation time: {loss_calc_time:.2f} seconds ({loss_calc_time / toal_time * 100:.2f}%)")
print(f"Total backpropagation time: {backpropagation_time:.2f} seconds ({backpropagation_time / toal_time * 100:.2f}%)")
print(f"Total to device time: {to_device_time:.2f} seconds ({to_device_time / toal_time * 100:.2f}%)")

epoch_count = last_epoch - first_epoch
print(f"\nAverage time per epoch: {toal_time / epoch_count:.2f} seconds")
print(f"\nAverage validation loss: {avg_loss_sum / epoch_count:.4f}")
print(f"Final validation loss: {val_loss:.4f}")

mlflow.end_run()

if args.plot_loss:
    subprocess.run(['python', 'plot_loss.py', f"{save_path}loss.csv"])
