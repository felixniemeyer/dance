import torch.nn as nn
from config import buffer_size

class V2(nn.Module):
    def __init__(self):
        super(V2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Dropout(0.3)
        )

        self.interface_size = buffer_size * 64 // 2 // 2

        hidden_size=64
        rnn_layers=4
        self.lstms = nn.LSTM(
            input_size=self.interface_size, 
            hidden_size=hidden_size, 
            num_layers=rnn_layers, 
            dropout=0.3,
            batch_first=True
        )

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, batch_inputs):  # takes a batch of sequences
        batch_size, seq_len, buffer_size = batch_inputs.size()


        reshaped_for_cnn = batch_inputs.view(-1, 1, buffer_size)

        cnn_output = self.features(reshaped_for_cnn)

        reshaped_for_rnn = cnn_output.view(batch_size, seq_len, self.interface_size)

        lstm_output, _ = self.lstms(reshaped_for_rnn)

        output = self.dense(lstm_output)

        return output

