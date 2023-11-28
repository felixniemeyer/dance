import torch.nn as nn
from config import buffer_size

class SingleEvent(nn.Module):
    def __init__(self):
        super(SingleEvent, self).__init__()
        hidden_size=128
        rnn_layers=4

        self.rnn = nn.RNN(
            input_size=buffer_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        )

        previous_neurons = hidden_size
        neurons = hidden_size // 2

        layers = []
        while(neurons > 1):
            layers += [
                nn.Linear(in_features=previous_neurons, out_features=neurons),
                nn.LeakyReLU(),
            ]

            previous_neurons = neurons
            neurons //= 2

        self.funnel = nn.Sequential(
            *layers, 
            nn.Linear(in_features=previous_neurons, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, batch_inputs, state=None):  # takes a batch of sequences
        x, new_state = self.rnn(batch_inputs, state)
        is_snare = self.funnel(x)
        return is_snare, new_state

