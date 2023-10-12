import os 
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset
from dancer_model import DancerModel

from torch.utils.data import DataLoader, random_split

import config

import argparse

import re

parser = argparse.ArgumentParser()

# in
parser.add_argument("--chunks-path", type=str, default='data/chunks/lakh_clean', help="path to chunks folder")

# out
parser.add_argument("--checkpoints-path", type=str, default='checkpoints', help="path to checkpoints folder. Structure: <checkpoints_path>/<tag>/<run>/<epoch>.pt")
parser.add_argument("--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")
parser.add_argument("--run", type=str, default='0', help="name of the run")
parser.add_argument("--continue-from", type=int, default=None, help="Epoch number to continue from.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-1, help="learning rate")
parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")

# misc
parser.add_argument("-m", "--max-size", type=int, default=None, help="truncate dataset to this size. For test runs.")

args = parser.parse_args()

# If tfs is set to 0, it means that the model is using its own generated output as input for all time steps, which is standard autoregressive behavior (no teacher forcing). If tfs is set to a positive value, it implies that the model uses ground truth inputs for the initial tfs time steps and then switches to its own predictions afterward (partial teacher forcing). 
tfs = config.chunk_duration * config.sample_rate // 512 // 3

if args.tag is None:
    os.system('git log --decorate -n 1 > ~git_tag.txt')
    try: 
        with open('git_tag.txt', 'r') as f:
            line = f.readline()
            match = re.search(r'\(HEAD.*tag: (.*?),', line)
            if match is None:
                raise Exception('no git tag found')
            else: 
                print('using git tag:', git_tag)
                args.tag = match.group(1)
        os.remove('~git_tag.txt')
    except Exception as e:
        print(e)
        print('Either provide tag by -t or create a git tag')

target_epoch = args.num_epochs
if args.continue_from is not None:
    target_epoch += args.continue_from

args.save_to = f"{args.checkpoints_path}/{args.tag}/run-{args.run}/{target_epoch}.pt"

os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

batch_size = args.batch_size

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset(args.chunks_path, max_size=args.max_size, teacher_forcing_size=tfs)
print('dataset size: ', len(dataset))
print()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from disk if it exists
model = DancerModel().to(device)
first_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if args.continue_from is not None:
    try: 
        file = f"{args.checkpoints_path}/{args.tag}/run-{args.run}/{args.continue_from}.pt"
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        print('loaded checkpoint from', args.continue_from)
    except FileNotFoundError:
        print('no checkpoint found at', args.continue_from)
        exit()

last_epoch = first_epoch + args.num_epochs

# Define loss function and optimizer
criterion = nn.L1Loss()

# Training loop
for epoch in range(first_epoch, last_epoch):
    print(f"Epoch {epoch+1} of {last_epoch}")
    model.train()  # Set the model to training mode
    print('training') 
    start_time = time.time()
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)
        # print('shape', batch_inputs.shape, batch_labels.shape)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Zero the gradients from previous iterations
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        loss = criterion(outputs[:, tfs:, :], batch_labels[:, tfs:, :]) # double check this

        # Backpropagation
        loss.backward()
        optimizer.step()
    print()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    print('evaluation')
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for i, (val_inputs, val_labels) in enumerate(val_loader):
            print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs = model(val_inputs)

            # Compute the loss
            loss = criterion(val_outputs, val_labels)

            # Compute accuracy
            total_loss += loss.item() * val_inputs.size(0)
            total_samples += val_inputs.size(0)
            
        loss = total_loss / total_samples
        print()
        print(f"Validation loss: {loss:.4f}")

print('saving model to', args.save_to)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': last_epoch,
    'hyperparameters': {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'continued_from': args.continue_from,
    }
}, args.save_to)

print('done')
