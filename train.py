import os 
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset

from dancer_model import DancerModel

from models.cnn_only import CNNOnly
from models.rnn_only import RNNOnly
from models.cnn_and_rnn import CNNAndRNN
from models.cnn_and_rnn_and_funnel import CNNAndRNNAndFunnel

from torch.utils.data import DataLoader, random_split

import config

import argparse

import re

parser = argparse.ArgumentParser()

# in
parser.add_argument("--chunks-path", type=str, default='data/chunks/lakh_clean', help="path to chunks folder")

# out
parser.add_argument("--checkpoints-path", type=str, default='checkpoints', help="path to checkpoints folder. Structure: <checkpoints_path>/<tag>/<epoch>.pt")
parser.add_argument("-t", "--tag", type=str, help="Tag to save checkpoint to. Current github tag will be used as default.")
parser.add_argument("--continue-from", type=int, default=None, help="Epoch number to continue from.")

# hyperparameters
parser.add_argument("-e", "--num-epochs", type=int, default=1, help="number of epochs to train for")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-2, help="learning rate")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="batch size")

parser.add_argument("--audio-event-half-life", type=float, default=0.02, help="half life of kicks and snares in seconds")

parser.add_argument("--model", type=int, default=8, help="number of features in first layer of CNN")

parser.add_argument("--cnn-first-layer-feature-size", type=int, default=8, help="number of features in first layer of CNN")
parser.add_argument("--cnn-activation-function", type=str, default='tanh', help="activation function to use in CNN. Options: lrelu, tanh, sigmoid")
parser.add_argument("--cnn-layers", type=int, default=2, help="number of layers in CNN")
parser.add_argument("--cnn-dropout", type=float, default=0, help="dropout to use in CNN")

parser.add_argument("--rnn-hidden-size", type=int, default=64, help="number of hidden units in RNN")
parser.add_argument("--rnn-layers", type=int, default=2, help="number of layers in RNN")
parser.add_argument("--rnn-dropout", type=float, default=0, help="dropout to use in RNN")


parser.add_argument("--loss-function", type=str, default='mse', help="loss function to use. Options: mse, l1, smoothl1, bcew")

# misc
parser.add_argument("-m", "--max-size", type=int, default=None, help="truncate dataset to this size. For test runs.")

parser.add_argument("--summarize", action='store_true', help="don't log, show only summary at the end")

args = parser.parse_args()

# If tfs is set to 0, it means that the model is using its own generated output as input for all time steps, which is standard autoregressive behavior (no teacher forcing). If tfs is set to a positive value, it implies that the model uses ground truth inputs for the initial tfs time steps and then switches to its own predictions afterward (partial teacher forcing). 
tfs = config.chunk_duration * config.sample_rate // 512 // 3

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

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset(args.chunks_path, max_size=args.max_size, teacher_forcing_size=tfs, kick_half_life=args.audio_event_half_life)
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
first_epoch = 0

model_parameters = None
model = None
optimizer = None

if args.continue_from is not None:
    try: 
        file = f"{args.checkpoints_path}/{args.tag}/{args.continue_from}.pt"
        checkpoint = torch.load(file)

        print('ignoring model parameters and using the ones from the checkpoint')
        model_parameters = checkpoint['model_parameters']
        model = DancerModel(
            cnn_first_layer_feature_size=model_parameters['cnn_first_layer_feature_size'],
            cnn_activation_function=model_parameters['cnn_activation_function'],
            cnn_layers=model_parameters['cnn_layers'],
            cnn_dropout=model_parameters['cnn_dropout'],
            rnn_hidden_size=model_parameters['rnn_hidden_size'],
            rnn_layers=model_parameters['rnn_layers'],
            rnn_dropout=model_parameters['rnn_dropout'],
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        first_epoch = checkpoint['epoch']
        print('loaded checkpoint from', args.continue_from)
    except FileNotFoundError:
        print('no checkpoint found at', args.continue_from)
        exit()
else: 
    model_parameters = {
        'cnn_first_layer_feature_size': args.cnn_first_layer_feature_size,
        'cnn_activation_function': args.cnn_activation_function,
        'cnn_layers': args.cnn_layers,
        'cnn_dropout': args.cnn_dropout,
        'rnn_hidden_size': args.rnn_hidden_size,
        'rnn_layers': args.rnn_layers,
        'rnn_dropout': args.rnn_dropout,
    }
    model = DancerModel(
        cnn_first_layer_feature_size=args.cnn_first_layer_feature_size, 
        cnn_activation_function=args.cnn_activation_function,
        cnn_layers=args.cnn_layers,
        cnn_dropout=args.cnn_dropout,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    

last_epoch = first_epoch + args.num_epochs

# Define loss function and optimizer
if args.loss_function == 'mse':
    criterion = nn.MSELoss()
elif args.loss_function == 'l1':
    criterion = nn.L1Loss()
elif args.loss_function == 'smoothl1':
    criterion = nn.SmoothL1Loss()
elif args.loss_function == 'bcew':
    criterion = nn.BCEWithLogitsLoss()
else:
    raise Exception('invalid loss function')

loss = 0
avg_loss = 0
toal_time = 0
epoch_count = 0

# Training loop
for epoch in range(first_epoch, last_epoch):

    print(f"Epoch {epoch+1} of {last_epoch}")
    print('\nTraining')
    model.train()  # Set the model to training mode
    start_time = time.time()
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        if not args.summarize: print('\rbatch', i + 1, 'of', (train_size - 1) // batch_size + 1, end='\r', flush=True)
        # print('shape', batch_inputs.shape, batch_labels.shape)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Zero the gradients from previous iterations
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        # loss = criterion(outputs[:, tfs:, :], batch_labels[:, tfs:, :]) # Teacher Forcing
        loss = criterion(outputs, batch_labels) 

        # Backpropagation
        loss.backward()

        optimizer.step()

    print()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    print('Evaluation')
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for i, (val_inputs, val_labels) in enumerate(val_loader):
            if not args.summarize: print('\rbatch', i + 1, 'of', (val_size - 1) // batch_size + 1, end='\r', flush=True)
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs = model(val_inputs)

            # Compute the loss
            loss = criterion(val_outputs, val_labels)

            # Compute accuracy
            total_loss += loss.item() * val_inputs.size(0)
            total_samples += val_inputs.size(0)
            
        avg_loss += total_loss / total_samples
        print()
        print(f"Validation loss: {loss:.4f}")

    epoch_count += 1
    epoch_duration = time.time() - start_time
    toal_time += epoch_duration
    if not args.summarize: print(f"Epoch duration: {epoch_duration:.2f} seconds")

print('\nSaving model to', args.save_to)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': last_epoch,
    'model_parameters': model_parameters, 
    'hyperparameters': {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'continued_from': args.continue_from,
    }
}, args.save_to)


if args.summarize:
    exit(0)

print(f"Average time per epoch: {toal_time / epoch_count:.2f} seconds")
print(f"Average validation loss: {avg_loss / epoch_count:.4f}")
print(f"Final validation loss: {loss:.4f}")
