# built-in
from typing import List

# installed
import torch
import torch.nn as nn


class RNN(nn.Module):
    """PyTorch LSTM model with dropouts (no dropouts if p=0)"""

    def __init__(self, vocab_size: int, embed_dim: int, rnn_hidden_size: int, fc_hidden_size: int, p_fc1: float = 0, \
                 p_fc2: float = 0, bidirec: bool = False) -> None:

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=bidirec)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)  # fully connected layer_1
        self.dropout_fc1 = nn.Dropout(p_fc1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.dropout_fc2 = nn.Dropout(p_fc2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text: List[int], lengths: int) -> torch.Tensor:

        out = self.embedding(text)

        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.dropout_fc1(out)  # DROPOUT
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout_fc2(out)  # DROPOUT
        out = self.sigmoid(out)

        return out
