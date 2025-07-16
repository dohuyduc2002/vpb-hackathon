import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.nn import GATConv, GAE

class GATEncoderWithEdgeAttrs(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels,
                            heads=1, concat=False, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, hidden_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        edge_attr = self.edge_mlp(edge_attr)
        return x, edge_attr