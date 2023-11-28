import torch.nn as nn
from config import frame_size

class CNNOnly(nn.Module):
    def __init__(self):
        super(CNNOnly, self).__init__()
        cnn_layers=6
        pool_size = 2

        previous_feature_size = 1
        feature_size = 32

        layers = []
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
            ]
            previous_feature_size = feature_size
            feature_size *= 2


        self.conv_layers = nn.Sequential(*layers)

        self.post_cnn_size = frame_size * previous_feature_size // (pool_size) ** cnn_layers

        self.finalLayer = nn.Sequential(
            nn.Linear(in_features=self.post_cnn_size, out_features=self.post_cnn_size // 2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.post_cnn_size // 2, out_features=2), 
            nn.Sigmoid()
        )

    def forward(self, batch_inputs, state=None):  # takes a batch of sequences
        frames = batch_inputs.view(-1, 1, frame_size)
        cnn_outputs = self.conv_layers(frames)
        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], self.post_cnn_size) # batch id, sequence id, buffer id, feature id
        return self.finalLayer(cnn_outputs), state

