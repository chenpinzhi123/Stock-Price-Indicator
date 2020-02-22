import torch
import torch.nn as nn
import torch.nn.functional as F

# Here we define our model as a class
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim):
        super().__init__()

        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(input_dim, self.feature_dim)

        self.hidden_layer_size = hidden_dim
        self.lstm = nn.LSTM(self.feature_dim, hidden_dim)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, input_seq):

        out = input_seq.view(len(input_seq), 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)

        out, self.hidden_cell = self.lstm(out, self.hidden_cell)
        out = out.view(len(out), -1)
        out = self.drop(out)

        predictions = self.fc2(out)
        return predictions
