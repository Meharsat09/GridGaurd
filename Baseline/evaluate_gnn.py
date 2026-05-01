import torch
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GRAPH_DIR = os.path.join(PROJECT_ROOT, "Graph")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")

GRAPH_FILES = [
    "graph_case14.pt",
    "graph_case30.pt",
    "graph_case118.pt",
    "graph_pglib_case118.pt",
    "graph_pglib_case300.pt"
]

MODEL_PATH = os.path.join(MODEL_DIR, "gnn_final_model.pt")

# -----------------------------
# Define GNN model (same as training)
# -----------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.conv2 = SAGEConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.fc(x)

# -----------------------------
# Load model
# -----------------------------
sample_graph = torch.load(os.path.join(GRAPH_DIR, GRAPH_FILES[0]), weights_only=False)
model = GraphSAGE(sample_graph.x.shape[1])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# -----------------------------
# Run inference
# -----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for file in GRAPH_FILES:
        graph = torch.load(os.path.join(GRAPH_DIR, file), weights_only=False)
        out = model(graph)
        preds = out.argmax(dim=1)

        y_true.extend(graph.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Metrics
# -----------------------------
print("\n=== GNN Evaluation Results ===\n")
print(classification_report(y_true, y_pred, digits=3))

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
