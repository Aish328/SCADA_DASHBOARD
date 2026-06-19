import torch
import torch.nn as nn
from dataload import FAULT_CLASSES, FAULT_COLORS   # re-export so other scripts import from here


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with attention pooling for 6-class fault detection.
    Input  : (batch, seq_len, input_size)
    Output : (batch, num_classes)  raw logits
    """

    def __init__(self, input_size=9, hidden_size=128,
                 num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _  = self.lstm(x)                          # (B, T, H*2)
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1)                 # (B, T, 1)
        context      = (lstm_out * attn_weights).sum(dim=1)  # (B, H*2)
        return self.classifier(context)                      # (B, C)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
        return preds, probs