import torch
import torch.nn as nn

class RST(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.3):
        super(RST, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, (hn, cn) = self.lstm(x)
        out = hn[-1]
        out = self.fc(out)
        return out.squeeze()
