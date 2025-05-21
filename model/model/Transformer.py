import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=6000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, hidden_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
        Returns:
            torch.Tensor: Tensor with positional encoding added, same shape as input.
        """
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size=None,
                 nhead: int = 8,
                 dropout: float = 0.1,
                 mode: str = "spike_detection"  # "spike_detection" or "feature_extraction"
                 ):   
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.nhead = nhead
        self.dropout = dropout
        self.mode = mode

        # Input projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # Fully connected layer for sequence prediction mode
        if mode == "spike_detection":
            self.fc = nn.Linear(hidden_size, output_size)
            #self.sigmoid = nn.Sigmoid()  # Optional for binary classification

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            torch.Tensor: 
                - If mode="feature_extraction": (batch_size, hidden_size).
                - If mode="sequence_prediction": (batch_size, seq_length, output_size).
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        if self.mode == "feature_extraction":
            return x[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        elif self.mode == "spike_detection":
            x = self.fc(x)  # Shape: (batch_size, seq_length, output_size)
            return x
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
