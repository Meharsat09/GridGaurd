import os

# Absolute paths (safe & portable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GRAPH_DIR = os.path.join(PROJECT_ROOT, "Graph")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")

MODEL_PATH = os.path.join(MODEL_DIR, "gnn_final_model.pt")

# Grid name → graph file mapping
GRAPH_MAP = {
    "ieee14": "graph_case14.pt",
    "ieee30": "graph_case30.pt",
    "ieee118": "graph_case118.pt",
    "pglib118": "graph_pglib_case118.pt",
    "pglib300": "graph_pglib_case300.pt"
}

# Phase-1 decision rule (remembered point)
RISK_THRESHOLD = 0.7
