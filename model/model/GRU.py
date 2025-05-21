import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, **kwargs):   
        super(GRU, self).__init__()

        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs['num_layers']
        kwargs['batch_first'] = True
        self.gru = nn.GRU(**kwargs)

        # self.linear1 = nn.Linear(gru_hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        """
        # Set initial hidden states (h0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0) # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out[:, -1, :] # Return the hidden state of last time step
        return out
    
# Define the GRU2 model for spike detection
class GRU2(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int
                 ):   
        super(GRU2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid() # for binary classification

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        """
        out, _ = self.gru(x)
        #out = out[:, -1, :]
        out = self.fc(out)
        # out = self.sigmoid(out)
        return out

def info_nce_loss_with_triplets(anchor, positive, negative, temperature=0.1):
    """
    Args:
        anchor (torch.Tensor): Anchor sample embeddings, shape (batch_size, hidden_size)
        positive (torch.Tensor): Positive sample embeddings, shape (batch_size, hidden_size)
        negative (torch.Tensor): Negative sample embeddings, shape (batch_size, hidden_size)
        temperature (float): Temperature parameter for scaling logits
    Returns:
        loss (torch.Tensor): InfoNCE loss value
    """
    # Normalize the embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    # Calculate similarities (dot products or cosine similarities)
    positive_sim = torch.sum(anchor * positive, dim=-1) / temperature  # shape (batch_size,)
    negative_sim = torch.sum(anchor * negative, dim=-1) / temperature  # shape (batch_size,)
    
    # Combine positive and negative similarities
    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim.unsqueeze(1)], dim=1)  # shape (batch_size, 2)
    
    # Create labels: positive should be at index 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long).cuda()  # shape (batch_size,)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss
    