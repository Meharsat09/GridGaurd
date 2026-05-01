import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from config import MODEL_PATH

# Same architecture used during training
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

# Singleton model instance
_model = None

def load_model(in_channels: int):
    global _model
    if _model is None:
        _model = GraphSAGE(in_channels)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu")
        )
        _model.eval()
    return _model
