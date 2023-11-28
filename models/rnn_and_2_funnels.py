import torch
import torch.nn as nn
from config import frame_size

class RNNAnd2Funnels(nn.Module):
    def __init__(self):
        super(RNNAnd2Funnels, self).__init__()
        hidden_size=128
        rnn_layers=4

        self.rnn = nn.RNN(
            input_size=frame_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        #    nonlinearity='relu',
        )

        self.funnels = nn.ModuleList()

        for _ in range(2): 
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

            self.funnels.append(nn.Sequential(
                *layers, 
                nn.Linear(in_features=previous_neurons, out_features=1),
                nn.Sigmoid(),
            ))

    def forward(self, batch_inputs, state=None):  # takes a batch of sequences
        x, new_state = self.rnn(batch_inputs, state)
        is_kick = self.funnels[0](x)
        is_snare = self.funnels[1](x)
        return torch.cat((is_kick, is_snare), dim=2), new_state

