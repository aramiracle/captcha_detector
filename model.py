import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CRNNModel(nn.Module):

    def __init__(self, num_classes=36, max_length=10, rnn_hidden_size=256):
        super(CRNNModel, self).__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.rnn_hidden_size = rnn_hidden_size

        # CNN for feature extraction
        self.model = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
        self.cnn = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        # GRU for sequence recognition
        self.gru_hidden_size = rnn_hidden_size
        self.gru = nn.GRU(1280, rnn_hidden_size, bidirectional=True, batch_first=True)

        # LSTM for sequence recognition
        self.lstm_hidden_size = rnn_hidden_size
        self.lstm = nn.LSTM(1280, rnn_hidden_size, bidirectional=True, batch_first=True)

        # Linear layer for mapping hidden states to output at each time step
        self.fc = nn.Linear(4 * rnn_hidden_size, max_length * (num_classes + 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        features = features.view(features.size(0), -1, 1280)  # Adjust the size for the RNN input

        # GRU sequence recognition
        gru_out, _ = self.gru(features)

        # LSTM sequence recognition
        lstm_out, _ = self.lstm(features)

        # Concatenate GRU and LSTM outputs
        combined_out = torch.cat((gru_out, lstm_out), dim=-1)

        output_sequence = self.fc(combined_out).view(combined_out.size(0), self.max_length, self.num_classes + 1)

        output_probs = self.softmax(output_sequence)

        return output_probs
    
class CNNModel(nn.Module):

    def __init__(self, num_classes=36, max_length=10):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.max_length = max_length

        # CNN for feature extraction
        self.model = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
        self.cnn = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        # Linear layer for mapping hidden states to output at each time step
        self.fc = nn.Linear(1280, max_length * (num_classes + 1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        features = features.view(features.size(0), -1, 1280)  # Adjust the size for the RNN input

        output_sequence = self.fc(features).view(features.size(0), self.max_length, self.num_classes + 1)
        
        output_probs = self.softmax(output_sequence)
        
        return output_probs