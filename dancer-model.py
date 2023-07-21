import torch.nn as nn

bitrate = 16
sample_rate = 44100
duration = 16
channels = 2
buffer_size = 512

size = bitrate * sample_rate * duration * channels

buffers_per_file = sample_rate * duration // buffer_size

offset = sample_rate * duration % buffer_size

# print info
print(f"bitrate: {bitrate}")
print(f"sample_rate: {sample_rate}")
print(f"duration: {duration}")
print(f"channels: {channels}")
print(f"size: {size}")
print(f"buffers_per_file: {buffers_per_file}")
print(f"offset: {offset}")

class Dancer(nn.Module):
    def __init__(self):
        super(Dancer, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=8,
                kernel_size=3,
            ), 
            nn.ReLU(), 
            nn.Dropout(0.5), 

            # averge pooling
            nn.AvgPool1d(
                kernel_size=2,
            ), 

            # maybe more convolutional layers needed
        )

        self.rnn = nn.RNN(
            input_size=8,
            hidden_size=8,
            num_layers=2,
            batch_first=True,
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8, 8), 
            nn.ReLU(), 
        )

        self.output = nn.Linear(8, 2) # snare and kick

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.rnn(x)
        x = self.fc_layers(x)
        x = self.output(x)
        return x

