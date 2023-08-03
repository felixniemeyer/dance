import os 

import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset
from dancer_model import DancerModel

from torch.utils.data import DataLoader, random_split

import config

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--continue_from", type=str, default=None, help="path to checkpoint to continue from")
parser.add_argument("-o", "--save_to", type=str, default=None, help="path to save checkpoint to")

parser.add_argument("-e", "--num_epochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("-r", "--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
parser.add_argument("-m", "--max_size", type=int, default=None, help="truncate dataset to this size")

args = parser.parse_args()

batch_size = args.batch_size

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset("./chunks", max_size=args.max_size)
print('dataset size: ', len(dataset))
print()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from disk if it exists
model = DancerModel().to(device)
first_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if args.continue_from is not None:
    checkpoint = torch.load(args.continue_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    first_epoch = checkpoint['epoch']
    print('loaded checkpoint from', args.continue_from)
    print()

last_epoch = first_epoch + args.num_epochs

# Define loss function and optimizer
criterion = nn.L1Loss()

# Training loop
for epoch in range(first_epoch, last_epoch):
    print(f"Epoch {epoch+1} of {last_epoch}")
    model.train()  # Set the model to training mode
    print('training') 
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)
        # print('shape', batch_inputs.shape, batch_labels.shape)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Zero the gradients from previous iterations
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

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
        print(f"Validation loss: {loss:.4f}\n")

# save checkpoint
if args.save_to is None:
    args.save_to = f"checkpoint-{last_epoch}.pt"
else: 
    # make path to file 
    os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

print('saving model to', args.save_to)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': last_epoch,
}, args.save_to)

print('done')
