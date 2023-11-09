import os 
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset

from models.selector import getModelClass, loadModel, getModels, saveModel 

from torch.utils.data import DataLoader, random_split

import config

import argparse

import re

parser = argparse.ArgumentParser()

parser.add_argument("model", type=str, help="model to train. Options: " + ",".join(getModels()))

# in
parser.add_argument("--chunks-path", type=str, default='data/chunks/lakh_clean', help="path to chunks folder")
parser.add_argument("--continue-from", type=int, default=None, help="Epoch number to continue from.")

# out
parser.add_argument("--checkpoints-path", type=str, default='checkpoints', help="path to checkpoints folder. Structure: <checkpoints_path>/<tag>/<epoch>.pt")
parser.add_argument("-t", "--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-2, help="learning rate")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="batch size")

# audio
parser.add_argument("--audio-event-half-life", type=float, default=0.02, help="half life of kicks and snares in seconds")

# misc
parser.add_argument("-d", "--dataset-size", type=int, default=None, help="truncate dataset to this size. For test runs.")
parser.add_argument("--summarize", action='store_true', help="don't log, show only summary at the end")

args = parser.parse_args()

relative_offset = 0.2


if args.tag is None:
    os.system('git log --decorate -n 1 > .git_tag.txt~')
    try: 
        with open('.git_tag.txt~', 'r') as f:
            line = f.readline()
            match = re.search(r'\(HEAD.*tag: (.*?),', line)
            if match is None:
                raise Exception('no git tag found')
            else: 
                tag = match.group(1)
                print('using git tag:', args.tag)
                args.tag = tag
    except Exception as e:
        print(e)
        print('Either provide tag by -t or create a git tag')
        exit(0)
    finally:
        os.remove('.git_tag.txt~')

target_epoch = args.num_epochs
if args.continue_from is not None:
    target_epoch += args.continue_from

args.save_to = f"{args.checkpoints_path}/{args.tag}/{target_epoch}.pt"

os.makedirs(os.path.dirname(args.save_to), exist_ok=True)

batch_size = args.batch_size

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset(args.chunks_path, config.buffer_size, config.samplerate, max_size=args.dataset_size)
print('dataset size: ', len(dataset))
print()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

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
    try: 
        file = f"{args.checkpoints_path}/{args.tag}/{args.continue_from}.pt"
        model, obj = loadModel(file)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(obj['optimizer_state_dict'])

        first_epoch = obj['epoch']
        print('loaded checkpoint from', args.continue_from)
    except FileNotFoundError:
        print('no checkpoint found at', args.continue_from)
        exit()
else: 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

last_epoch = first_epoch + args.num_epochs

criterion = nn.CrossEntropyLoss()

loss = 0
avg_loss = 0
toal_time = 0

forward_time = 0
loss_calc_time = 0
backpropagation_time = 0
start_calc = 0
to_device_time = 0


# Training loop
for epoch in range(first_epoch, last_epoch):

    print(f"Epoch {epoch+1} of {last_epoch}")
    print('\nTraining')

    model.train()  # Set the model to training mode
    start_time = time.time()

    for i, (batch_inputs, batch_labels) in enumerate(train_loader):

        if not args.summarize: print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)

        start_calc = time.time()
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        to_device_time = time.time() - start_calc

        # Forward pass
        start_calc = time.time()
        outputs, _ = model(batch_inputs)
        forward_time += time.time() - start_calc

        # Zero out the gradients
        optimizer.zero_grad()

        # Compute the loss
        start_calc = time.time()
        # teacher forcing: ignore outputs in the beginning of the sequence
        tfs = int(batch_inputs.shape[1] * relative_offset)
        kick_loss = criterion(outputs[:, tfs:, 0], batch_labels[:, tfs:, 0]) 
        snare_loss = criterion(outputs[:, tfs:, 1], batch_labels[:, tfs:, 1])
        combined_loss = kick_loss + snare_loss
        loss_calc_time += time.time() - start_calc

        # Backpropagation
        start_calc = time.time()
        combined_loss.backward()
        backpropagation_time = time.time() - start_calc

        optimizer.step()

    print()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    print('Evaluation')
    with torch.no_grad():

        total_loss = 0
        total_batches = 0

        for i, (val_inputs, val_labels) in enumerate(val_loader):

            if not args.summarize: print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)

            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs, _ = model(val_inputs)

            # Compute the loss
            tfs = int(val_inputs.shape[1] * relative_offset)
            kick_loss = criterion(val_outputs[:, tfs:, 0], val_labels[:, tfs:, 0])
            snare_loss = criterion(val_outputs[:, tfs:, 1], val_labels[:, tfs:, 1])
            combined_loss = kick_loss + snare_loss

            # Compute accuracy
            total_loss += combined_loss
            total_batches += 1
            
        avg_loss += total_loss / total_batches
        print()
        print(f"Validation loss: {loss:.4f}")

    epoch_duration = time.time() - start_time
    toal_time += epoch_duration
    if not args.summarize: print(f"Epoch duration: {epoch_duration:.2f} seconds")

print('\nSaving model to', args.save_to)
saveModel(args.save_to, model, {
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': last_epoch,
    'hyperparameters': {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'continued_from': args.continue_from,
    }
})

print(f"\nTotal training time: {toal_time:.2f} seconds")
print(f"Total forward pass time: {forward_time:.2f} seconds ({forward_time / toal_time * 100:.2f}%)")
print(f"Total loss calculation time: {loss_calc_time:.2f} seconds ({loss_calc_time / toal_time * 100:.2f}%)")
print(f"Total backpropagation time: {backpropagation_time:.2f} seconds ({backpropagation_time / toal_time * 100:.2f}%)")
print(f"Total to device time: {to_device_time:.2f} seconds ({to_device_time / toal_time * 100:.2f}%)")

epoch_count = last_epoch - first_epoch
print(f"\nAverage time per epoch: {toal_time / epoch_count:.2f} seconds")
print(f"\nAverage validation loss: {avg_loss / epoch_count:.4f}")
print(f"Final validation loss: {loss:.4f}")

