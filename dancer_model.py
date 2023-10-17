import torch.nn as nn

import argparse

import config

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=20)


class DancerModel(nn.Module):
    def __init__(self, cnn_layers=3, cnn_first_layer_feature_size=16, cnn_activation_function='lrelu', cnn_dropout=0, rnn_hidden_size=64, rnn_layers=2, rnn_dropout=0):
        super(DancerModel, self).__init__()

        self.cnn_activation_function = cnn_activation_function
        self.cnn_dropout = cnn_dropout

        layers = []
        previous_feature_size = config.channels
        feature_size = cnn_first_layer_feature_size
        for _ in range(cnn_layers):
            layers += [
                nn.Conv1d(
                    in_channels=previous_feature_size,
                    out_channels=feature_size,
                    kernel_size=3,
                    padding=1,
                ),
                self.make_cnn_activation_layer(),
                self.make_cnn_dropout_layer(),
                nn.MaxPool1d(
                    kernel_size=2,
                    stride=2,
                )
            ]
            previous_feature_size = feature_size
            feature_size *= 2


        self.conv_layers = nn.Sequential(*layers)

        post_cnn_size = config.buffer_size // 2 ** cnn_layers * previous_feature_size

        self.rnn = nn.RNN(
            input_size=post_cnn_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=2)

    def forward(self, batch_inputs):  # takes a batch of sequences
        cnn_inputs = batch_inputs.view(-1, config.channels, config.buffer_size)
        cnn_outputs = self.conv_layers(cnn_inputs)

        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], -1)

        x, _ = self.rnn(cnn_outputs)

        x = self.fc(x)
        return x

    def make_cnn_activation_layer(self):
        if self.cnn_activation_function == 'lrelu':
            return nn.LeakyReLU() 
        elif self.cnn_activation_function == 'tanh':
            return nn.Tanh()
        elif self.cnn_activation_function == 'sigmoid':
            return nn.Sigmoid()

    def make_cnn_dropout_layer(self):
        return nn.Dropout(self.cnn_dropout)
