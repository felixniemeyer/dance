import torch.nn as nn

import argparse

import config

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=20)


class DancerModel(nn.Module):
    def __init__(self):
        super(DancerModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(
                kernel_size=4,
                stride=4,
            ),
            nn.Conv1d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(
                kernel_size=4,
                stride=4,
            ),
        )

        self.rnn = nn.RNN(
            input_size=config.buffer_size * 8 // 16,  # buffer_size // cnn downsample * num_channels
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
        )

        self.output = nn.Linear(8, 2)  # snare and kick
drive
    def forward(self, batch_inputs):  # takes a batch of sequences
        cnn_inputs = batch_inputs.view(-1, config.channels, config.buffer_size)
        cnn_outputs = self.conv_layers(cnn_inputs)

        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], -1)

        x, _ = self.rnn(cnn_outputs)

        x = self.fc_layers(x)
        x = self.output(x)
        return x
