import torch.nn as nn

import argparse

from config import buffer_size

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=20)

class RNNOnly(nn.Module):
    def __init__(self):
        super(RnnOnly, self).__init__()

        self.rnn = nn.RNN(
            input_size=buffer_size,
            hidden_size=64,
            num_layers=4,
        )

        self.lin = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, batch_inputs):  # takes a batch of sequences
        x, _ = self.rnn(batch_inputs)
        y = self.lin(x)
        return y

