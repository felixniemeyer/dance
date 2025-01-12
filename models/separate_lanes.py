import torch
import torch.nn as nn
from config import frame_size

class SeparateLanes(nn.Module):
    def __init__(self):
        super(SeparateLanes, self).__init__()
        hidden_size=128
        rnn_layers=4
        self.events = 2

        self.rnns = nn.ModuleList()

        self.funnels = nn.ModuleList()
        for _ in range(self.events): 

            previous_neurons = hidden_size
            neurons = hidden_size // 2

            self.rnns.append(nn.RNN(
                input_size=frame_size,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
            ))

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

    def forward(self, batch_inputs, state = None):  # takes a batch of sequences
        predicted = []
        new_state = []
        for i in range(self.events):
            substate = None if state is None else state[i]
            x, new_substate = self.rnns[i](batch_inputs, substate)
            new_state.append(new_substate)
            predicted.append(self.funnels[i](x))
        return torch.cat(predicted, dim=2), new_state

