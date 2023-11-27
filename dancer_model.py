import torch.nn as nn

import argparse

from config import frame_size

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=20)

class DancerModel(nn.Module):
    def __init__(self, cnn_layers=3, cnn_first_layer_feature_size=4, cnn_activation_function='tanh', cnn_dropout=0, rnn_hidden_size=64, rnn_layers=2, rnn_dropout=0):
        super(DancerModel, self).__init__()

        self.cnn_activation_function = cnn_activation_function
        self.cnn_dropout = cnn_dropout

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
                self.make_cnn_activation_layer(),
                self.make_cnn_dropout_layer(),
                nn.MaxPool1d(
                    kernel_size=pool_size,
                    stride=pool_size,
                )
            ]
            previous_feature_size = feature_size
            feature_size *= 2


        self.conv_layers = nn.Sequential(*layers)

        self.post_cnn_size = frame_size // pool_size ** cnn_layers * previous_feature_size

        print('post_cnn_size', self.post_cnn_size)

        self.rnn = nn.RNN(
            input_size=self.post_cnn_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.finalLayer = nn.Sequential(
            nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(in_features=rnn_hidden_size // 2, out_features=2), 
            nn.Sigmoid()
        )

    def forward(self, batch_inputs):  # takes a batch of sequences
        frames = batch_inputs.view(-1, 1, frame_size)

        cnn_outputs = self.conv_layers(frames)

        cnn_outputs = cnn_outputs.view(batch_inputs.shape[0], batch_inputs.shape[1], self.post_cnn_size) # batch id, sequence id, frame id, feature id

        x, _ = self.rnn(cnn_outputs)

        return self.finalLayer(x)

    def make_cnn_activation_layer(self):
        if self.cnn_activation_function == 'lrelu':
            return nn.LeakyReLU() 
        elif self.cnn_activation_function == 'tanh':
            return nn.Tanh()
        elif self.cnn_activation_function == 'sigmoid':
            return nn.Sigmoid()

    def make_cnn_dropout_layer(self):
        return nn.Dropout(self.cnn_dropout)
