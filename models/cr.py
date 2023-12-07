import torch.nn as nn
from config import frame_size

class CR(nn.Module):
    def __init__(self, 
        cnn_layers=3, 
        cnn_first_layer_feature_size=32, 
        rnn_hidden_size=64, 
        rnn_layers=2, 
        rnn_dropout=0, 
        cnn_dropout=0
    ):
        super(CR, self).__init__()

        layers = []
        previous_feature_size = 1
        feature_size = cnn_first_layer_feature_size

        pool_size = 2
        for _ in range(cnn_layers):
            layers += [
                nn.Conv1d(
                    in_channels=previous_feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(),
                nn.MaxPool1d(
                    kernel_size=pool_size,
                    stride=pool_size,
                )
                nn.dropout(cnn_dropout),
            ]
            previous_feature_size = feature_size
            feature_size *= 2


        self.conv_layers = nn.Sequential(*layers)

        self.post_cnn_size = frame_size // pool_size ** cnn_layers * previous_feature_size

        self.rnn = nn.RNN(
            input_size=self.post_cnn_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.finalLayer = nn.Sequential(
            nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size // 2),
            nn.Sigmoid(),
            nn.Linear(in_features=rnn_hidden_size // 2, out_features=2), 
            nn.Sigmoid()
        )

    def forward(self, batch_inputs, state=None):  # takes a batch of sequences
        frames = batch_inputs.view(-1, 1, frame_size)
        cnn_outputs = self.conv_layers(frames)
        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], self.post_cnn_size) # batch id, sequence id, buffer id, feature id
        x, new_state = self.rnn(cnn_outputs, state)
        return self.finalLayer(x), new_state

