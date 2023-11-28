import torch.nn as nn
from config import buffer_size

class CNNAndRNNAndFunnel(nn.Module):
    def __init__(self):
        super(CNNAndRNNAndFunnel, self).__init__()
        layers = []
        previous_feature_size = 1
        feature_size = 16
        cnn_stride = 2
        cnn_layers = 2
        pool_size = 2

        for _ in range(cnn_layers):
            layers += [
                nn.Conv1d(
                    in_channels=previous_feature_size,
                    out_channels=feature_size,
                    kernel_size=7,
                    stride=cnn_stride,
                    padding=3,
                ),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=pool_size,
                    stride=pool_size,
                )
            ]
            previous_feature_size = feature_size
            feature_size *= 2

        self.conv_layers = nn.Sequential(*layers)

        self.post_cnn_size = buffer_size * previous_feature_size // (cnn_stride + pool_size) ** cnn_layers

        print('post_cnn_size', self.post_cnn_size)

        rnn_hidden_size = self.post_cnn_size // 2
        rnn_layers = 1
        self.rnn = nn.RNN(
            input_size=self.post_cnn_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )

        previous_neurons = rnn_hidden_size
        neurons = rnn_hidden_size // 2

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

    def forward(self, batch_inputs, state = None):  # takes a batch of sequences
        buffers = batch_inputs.view(-1, 1, buffer_size)
        cnn_outputs = self.conv_layers(buffers)
        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], self.post_cnn_size) # batch id, sequence id, buffer id, feature id

        new_state = None
        if(state is None):
            x, new_state = self.rnn(cnn_outputs)
        else:
            x, new_state = self.rnn(cnn_outputs, state)

        return self.finalLayer(x), new_state

