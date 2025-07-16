import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
import networkx as nx
import seaborn as sns
from torch import nn
from torch_geometric.nn import GATConv, GAE
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

#Load and preprocess data
d = pd.read_csv("./credit_card_transactions-ibm_v2.csv")
feature_name = 'Is Fraud?'
fraud_df = d[d['Is Fraud?'] == 'Yes']
nonfraud_df = d[d['Is Fraud?'] == 'No'].sample(n=70000, random_state=42)
df = pd.concat([fraud_df, nonfraud_df]).sample(frac=1, random_state=42)
df2 = df
df["Amount"] = df["Amount"].replace('[\$,]', '', regex=True).astype(float)
df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)
df["Errors?"] = df["Errors?"].fillna("No error")
# df = df[df["Errors?"] != "No error"]
df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)
df = df.drop(columns=["Time", "User", "Card", "Merchant State", "Zip"])
df["Merchant City"] = LabelEncoder().fit_transform(df["Merchant City"])
df["Use Chip"] = LabelEncoder().fit_transform(df["Use Chip"])
df["Errors?"] = LabelEncoder().fit_transform(df["Errors?"])

numeric_columns = ['Year', 'Month', 'Day', 'Amount']

scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df['Merchant City'] = LabelEncoder().fit_transform(df['Merchant City'])
df['MCC'] = LabelEncoder().fit_transform(df['MCC'])
df['Merchant Name'] = LabelEncoder().fit_transform(df['Merchant Name'])

#Build graph
G = nx.MultiGraph()
G.add_nodes_from(df["card_id"].unique(), type='card')
G.add_nodes_from(df["Merchant Name"].unique(), type='merchant')
node_mapping = {node: idx for idx, node in enumerate(G.nodes)}

total_edges = []
edge_attrs = []
labels = []
for _, row in df.iterrows():
    src = node_mapping[row["card_id"]]
    dst = node_mapping[row["Merchant Name"]]
    total_edges.append((src, dst))
    edge_attrs.append([row["Amount"], row["Merchant City"], 
                       row["Use Chip"],row["Year"],row["Month"],
                       row["Day"],row["Errors?"]])
    labels.append(row["Is Fraud?"])

edge_index = torch.tensor(np.array(total_edges).T, dtype=torch.long)
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)
x = torch.eye(len(node_mapping))

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)

data_train, data_test = train_test_split(
    list(range(data.edge_index.shape[1])), test_size=0.2, random_state=42, stratify=labels
)
train_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
train_mask[data_train] = True
test_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
test_mask[data_test] = True

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Training
encoder = GATEncoderWithEdgeAttrs(
    in_channels = x.shape[1],
    hidden_channels = 64,
    edge_attr_dim = edge_attr.shape[1]
).to(device)

model = GAE(encoder).to(device)

classifier = EdgeMLPClassifier(
    emb_dim = 64,
    hidden_dim = 64,
    num_classes = 2
).to(device)

data = data.to(device)

# 4. Optimizer and loss
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr = 0.001,
    weight_decay = 1e-4
)
criterion = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(500):
    model.eval()  # GAE is pretrained or frozen
    classifier.train()  # Classifier training mode
    optimizer.zero_grad()

    # Encode nodes with the GAT model (including edge attributes)
    z, processed_edge_attr = model.encode(data.x, data.edge_index, data.edge_attr)
    
    # Edge classifier: Use processed edge attributes
    out = classifier(z, data.edge_index, processed_edge_attr)
    
    # Calculate loss using the training mask
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    pred_train = out[train_mask].argmax(dim=1)
    acc_train = (pred_train == data.y[train_mask]).sum().item() / train_mask.sum().item()

    # Testing phase (with no gradients)
    classifier.eval()  # Set classifier to evaluation mode
    with torch.no_grad():
        # Encode the nodes again (for testing)
        z, processed_edge_attr = model.encode(data.x, data.edge_index, data.edge_attr)
        
        # Get predictions from the classifier
        out = classifier(z, data.edge_index, processed_edge_attr)
        
        # Calculate test loss and accuracy
        loss_test = criterion(out[test_mask], data.y[test_mask])
        pred_test = out[test_mask].argmax(dim=1)
        acc_test = (pred_test == data.y[test_mask]).sum().item() / test_mask.sum().item()

    # Track training and test metrics
    train_losses.append(loss.item())
    test_losses.append(loss_test.item())
    train_accs.append(acc_train)
    test_accs.append(acc_test)

    # Print training progress every 100 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Acc={acc_train:.4f} | Test Loss={loss_test.item():.4f}, Acc={acc_test:.4f}")