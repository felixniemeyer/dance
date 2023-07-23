import torch.nn as nn

import config

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=20)

class DancerModel(nn.Module):
    def __init__(self):
        super(DancerModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=8,
                kernel_size=3,
            ), 
            nn.ReLU(), 
            nn.Dropout(0.5), 

            # averge pooling
            nn.AvgPool1d(
                kernel_size=2,
            ), 

            # maybe more convolutional layers needed
        )

        self.rnn = nn.RNN(
            input_size=8,
            hidden_size=8,
            num_layers=2,
            batch_first=True,
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8, 8), 
            nn.ReLU(), 
        )

        self.output = nn.Linear(8, 2) # snare and kick

    def forward(self, x):
        x = self.conv_layers(x)
        x, _ = self.rnn(x)
        x = self.fc_layers(x)
        x = self.output(x)
        return x
