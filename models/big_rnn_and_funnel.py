import torch.nn as nn
from config import frame_size

class BigRNNAndFunnel(nn.Module):
    def __init__(self):
        super(BigRNNAndFunnel, self).__init__()
        hidden_size=128
        rnn_layers=5

        self.rnn = nn.RNN(
            input_size=frame_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        #    nonlinearity='relu',
        )

        previous_neurons = hidden_size
        neurons = hidden_size // 2

        layers = []
        while(neurons > 2):
            layers += [
                nn.Linear(in_features=previous_neurons, out_features=neurons),
                nn.LeakyReLU(),
            ]

            previous_neurons = neurons
            neurons //= 2

        self.finalLayer = nn.Sequential(
            *layers, 
            nn.Linear(in_features=previous_neurons, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, batch_inputs):  # takes a batch of sequences
        x, _ = self.rnn(batch_inputs)
        y = self.finalLayer(x)
        return y

