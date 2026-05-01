import torch
import os
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GRAPH_DIR = os.path.join(PROJECT_ROOT, "Graph")

GRAPH_FILES = [
    "graph_case14.pt",
    "graph_case30.pt",
    "graph_case118.pt",
    "graph_pglib_case118.pt",
    "graph_pglib_case300.pt"
]

def load_dataset():
    X_list = []
    y_list = []

    for file in GRAPH_FILES:
        graph = torch.load(os.path.join(GRAPH_DIR, file), weights_only=False)

        X = graph.x.cpu().numpy()
        y = graph.y.cpu().numpy()

        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)

    return X_all, y_all


if __name__ == "__main__":
    X, y = load_dataset()
    print("Dataset prepared")
    print("Total samples:", X.shape[0])
    print("Feature dimension:", X.shape[1])
