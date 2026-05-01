import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random

# -----------------------------
# STEP 1: Load graphs
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

graph_paths = [
    os.path.join(PROJECT_ROOT, "Graph", "graph_case14.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case30.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case118.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_pglib_case118.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_pglib_case300.pt")
]

graphs = [torch.load(p, weights_only=False) for p in graph_paths]

# -----------------------------
# STEP 2: Define GraphSAGE model
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

model = GraphSAGE(graphs[0].x.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# -----------------------------
# STEP 3: Training loop
# -----------------------------
for epoch in range(1, 101):
    model.train()
    total_loss = 0

    for graph in graphs:
        num_nodes = graph.num_nodes
        indices = list(range(num_nodes))
        random.shuffle(indices)

        split = int(0.8 * num_nodes)
        train_idx = indices[:split]
        test_idx = indices[split:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True

        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# -----------------------------
# STEP 4: Save model
# -----------------------------
model_path = os.path.join(PROJECT_ROOT, "Model", "gnn_final_model.pt")
torch.save(model.state_dict(), model_path)
print("✅ Model trained on IEEE 14 + IEEE 30 + IEEE 118 + pglib and saved.")
