import torch.nn as nn
from config import buffer_size

class RNNOnly(nn.Module):
    def __init__(self):
        super(RNNOnly, self).__init__()
        hidden_size=64
        rnn_layers=4

        self.rnn = nn.RNN(
            input_size=buffer_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        )

        self.lin = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, batch_inputs):  # takes a batch of sequences
        x, _ = self.rnn(batch_inputs)
        y = self.lin(x)
        return y

