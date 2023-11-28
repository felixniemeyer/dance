import torch.nn as nn
from config import buffer_size

class V2Coolio(nn.Module):
    def __init__(self):
        super(V2Coolio, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), 
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, 3, padding=1), 
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(64, 128, 3, padding=1),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.interface_size = buffer_size * 16

        hidden_size=64

        self.lstm = nn.LSTM(
            input_size=self.interface_size, 
            hidden_size=hidden_size, 
            num_layers=4, 
            dropout=0.5,
            batch_first=True
        )

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Sigmoid(),
            nn.Linear(hidden_size // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, batch_inputs, state = None):  # takes a batch of sequences
        batch_size, seq_len, buffer_size = batch_inputs.size()

        reshaped_for_cnn = batch_inputs.view(-1, 1, buffer_size)

        cnn_output = self.features(reshaped_for_cnn)

        reshaped_for_rnn = cnn_output.view(batch_size, seq_len, self.interface_size)

        new_state = None
        if(state is None): 
            lstm_output, new_state = self.lstm(reshaped_for_rnn)
        else: 
            lstm_output, new_state = self.lstm(reshaped_for_rnn, state)

        output = self.dense(lstm_output)

        return output, new_state

