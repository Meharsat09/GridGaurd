import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🚀 Starting evaluation...")

# -------------------------------
# Load graphs
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(BASE_DIR)

graph_paths = [
    os.path.join(PROJECT_ROOT, "Graph", "graph_case14.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case30.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case118.pt")
]

graphs = []
for path in graph_paths:
    print(f"📂 Loading graph: {path}")
    # These .pt files store full Data objects; allow full unpickling.
    g = torch.load(path, weights_only=False)
    print(f"   ✔ Nodes: {g.num_nodes}, Edges: {g.num_edges}")
    graphs.append(g)

print("✅ All graphs loaded successfully\n")

# -------------------------------
# Define SAME model as training
# -------------------------------
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.conv2 = SAGEConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)

# -------------------------------
# Load trained model
# -------------------------------
print("📥 Loading trained model...")
in_channels = graphs[0].x.shape[1]
model = GraphSAGE(in_channels)
model_path = os.path.join(PROJECT_ROOT, "Model", "gnn_final_model.pt")
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model loaded successfully\n")

# -------------------------------
# Stress function
# -------------------------------
def apply_stress(graph, load_factor=1.3, voltage_drop=0.07):
    g = graph.clone()

    # x = [Vm, Pd, Qd]
    g.x[:, 1] = g.x[:, 1] * load_factor   # Increase load
    g.x[:, 2] = g.x[:, 2] * load_factor
    g.x[:, 0] = g.x[:, 0] - voltage_drop  # Decrease voltage

    g.y = ((g.x[:, 0] < 0.95) | (g.x[:, 0] > 1.05)).long()


    return g

# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(graphs, use_stress=False):
    all_true = []
    all_pred = []

    with torch.no_grad():
        for i, graph in enumerate(graphs):
            print(f"🔹 Processing graph {i+1}/{len(graphs)}")

            if use_stress:
                graph = apply_stress(graph)

            out = model(graph)
            pred = out.argmax(dim=1)

            all_true.extend(graph.y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    return acc, precision, recall, f1

# -------------------------------
# NORMAL EVALUATION
# -------------------------------
print("\n===== 🟢 NORMAL CONDITION =====")
acc, precision, recall, f1 = evaluate(graphs, use_stress=False)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# -------------------------------
# STRESSED EVALUATION
# -------------------------------
print("\n===== 🔴 STRESSED CONDITION =====")
acc_s, precision_s, recall_s, f1_s = evaluate(graphs, use_stress=True)

print(f"Accuracy  : {acc_s:.4f}")
print(f"Precision : {precision_s:.4f}")
print(f"Recall    : {recall_s:.4f}")
print(f"F1-score  : {f1_s:.4f}")

print("\n================================")
print("✅ Evaluation Complete")