import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.nn import GATConv, GAE

class EdgeMLPClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_classes):
        super().__init__()
        # z[row] + z[col] + processed_edge_attr + |z[row] - z[col]|
        input_dim = 2 * emb_dim + emb_dim + emb_dim  
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z, edge_index, edge_attr):
        row, col = edge_index
        edge_diff = torch.abs(z[row] - z[col])
        edge_feat = torch.cat([z[row], z[col], edge_attr, edge_diff], dim=1)
        return self.edge_mlp(edge_feat)