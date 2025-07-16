import torch
from torch_geometric.nn import GATConv, GAE
import torch.nn.functional as F
from input_preprocess import preprocess_single
from src.gat_encoder import GATEncoderWithEdgeAttrs
from src.edge_classifier import EdgeMLPClassifier

ckpt = torch.load("fraud_gnn_model.pt", map_location="cpu")
node_mapping = ckpt["node_mapping"]
scaler = ckpt["scaler"]
label_encoders = ckpt["label_encoders"]

encoder = GATEncoderWithEdgeAttrs(in_channels=ckpt["node_mapping"].__len__(), hidden_channels=64, edge_attr_dim=7)
encoder.load_state_dict(ckpt["encoder_state_dict"])
model = GAE(encoder)

classifier = EdgeMLPClassifier(emb_dim=64, hidden_dim=64, num_classes=2)
classifier.load_state_dict(ckpt["classifier_state_dict"])

encoder.eval()
classifier.eval()

sample = {
    "user": "user1",
    "card": "card1",
    "year": "2023",
    "month": "5",
    "day": "20",
    "time": "12:00",
    "amount": "1200",
    "use_chip": "Chip Transaction",
    "merchant_name": "3189517333335617109",
    "merchant_city": "Riverside",
    "merchant_state": "CA",
    "zip": "92505.0",
    "mcc": "7538",
    "errors": "Insufficient Balance"
}

edge_index, edge_attr = preprocess_single(sample, node_mapping, scaler, label_encoders)

x = torch.eye(len(node_mapping))

with torch.no_grad():
    z, processed_edge_attr = encoder(x, edge_index, edge_attr)
    out = classifier(z, edge_index, processed_edge_attr)
    pred = torch.argmax(out, dim=1).item()

print("Predicted:", "Fraud" if pred == 1 else "Not Fraud")
