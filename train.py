import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset
from dancer_model import DancerModel

from torch.utils.data import DataLoader, random_split

import config

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--continue_from", type=str, default=None)
parser.add_argument("--save_to", type=str, default='./checkpoint.pt')

parser.add_argument("--num_epochs", type=int, default=10)

args = parser.parse_args()

batch_size = 128 # number of sequences per batch
learning_rate = 5e-3

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset("./chunks") # , 4 * 4 * 5)
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
model = DancerModel(batch_size).to(device)
first_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if args.continue_from is not None:
    checkpoint = torch.load(args.continue_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    first_epoch = checkpoint['epoch']
    print('loaded checkpoint from', args.continue_from)
    print()

last_epoch = first_epoch + args.num_epochs

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(first_epoch, last_epoch):
    print(f"Epoch {epoch+1} of {last_epoch}")
    model.train()  # Set the model to training mode
    print('training') 
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        print('\rbatch', i + 1, 'of', train_size // batch_size, end='\r', flush=True)
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
        total_correct = 0
        total_samples = 0
        for i, (val_inputs, val_labels) in enumerate(val_loader):
            print('\rbatch', i + 1, 'of', val_size // batch_size, end='\r', flush=True)
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs = model(val_inputs)
            val_outputs = torch.where(val_outputs > 0.5, 1, 0)

            correct = torch.eq(val_outputs, val_labels)

            both_correct = torch.logical_and(correct[:,:,0], correct[:,:,1])
             
            total_correct += torch.sum(both_correct)
            total_samples += batch_size * config.sequence_size
            
        accuracy = total_correct / total_samples
        print(f"Validation accuracy: {accuracy:.4f}")
    print()

# save checkpoint
print('saving model to', args.save_to)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': last_epoch,
}, args.save_to)

print('done')
