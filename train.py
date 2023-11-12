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
parser.add_argument("-ci", "--checkpoint-interval", type=int, default=1, help="save checkpoint every n epochs")
parser.add_argument("-t", "--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-4, help="learning rate")
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

save_path = f"{args.checkpoints_path}/{args.tag}/"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

batch_size = args.batch_size

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset(args.chunks_path, config.buffer_size, config.samplerate)
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


# scale down learning rate to 0.1 from initial value after 
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

last_epoch = first_epoch + args.num_epochs

class CustomLoss(nn.Module):
    def __init__(self, relative_offset):
        super(CustomLoss, self).__init__()
        self.relative_offset = relative_offset
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, full_predictions, full_labels):
        tfs = int(full_labels.shape[1] * self.relative_offset)
        predictions = full_predictions[:, tfs:, :]
        labels = full_labels[:, tfs:, :]

        # Define weights based on the condition where at least one output is 1
        # weights = torch.where(torch.logical_or(y_true == 1, y_pred == 1), torch.tensor(10.0), torch.tensor(1))

        # weights = (max of elements of last dimension) * 9 + 1
        weights = torch.max(labels, dim=2)[0] * 9 + 1
        
        # Apply weights to the binary cross-entropy loss for each output dimension
        loss_1 = nn.functional.binary_cross_entropy(predictions[:, :, 0], labels[:, :, 0], weight=weights)
        loss_2 = nn.functional.binary_cross_entropy(predictions[:, :, 1], labels[:, :, 1], weight=weights) 
        
        # Return the mean loss
        return (loss_1 + loss_2) / 2.0  # You can adjust this based on your preference

criterion = CustomLoss(0.2)

avg_loss = 0
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

    for i, (batch_inputs, batch_labels, _) in enumerate(train_loader):

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
        loss = criterion(outputs, batch_labels)
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

        for i, (val_inputs, val_labels, _) in enumerate(val_loader):

            if not args.summarize: print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)

            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs, _ = model(val_inputs)

            # Compute the loss
            loss = criterion(val_outputs, val_labels)

            # Compute accuracy
            total_loss += loss
            total_batches += 1
            
        avg_loss += total_loss / total_batches
        print()
        print(f"Validation loss: {loss:.4f}")

    epoch_duration = time.time() - start_time
    toal_time += epoch_duration
    if not args.summarize: print(f"Epoch duration: {epoch_duration:.2f} seconds")

    epoch_1 = epoch + 1
    if (args.checkpoint_interval != None and (epoch - first_epoch) % args.checkpoint_interval == 0) or epoch_1 == last_epoch: 
        file_path = f"{save_path}{epoch_1}.pt"
        print('\nSaving model to', file_path)
        saveModel(file_path, model, {
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
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

