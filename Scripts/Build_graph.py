import torch
import numpy as np
from torch_geometric.data import Data
import os
import re

# -------------------------------------------------
# Helper: extract matrix from MATPOWER .m file
# -------------------------------------------------
def extract_matrix(m_file, matrix_name):
    """
    Extracts numeric matrix (mpc.bus or mpc.branch) from MATPOWER .m file
    """
    with open(m_file, 'r') as f:
        text = f.read()

    # Capture content inside [ ... ];
    pattern = rf"{matrix_name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, re.S)

    if not match:
        raise ValueError(f"{matrix_name} not found in {m_file}")

    matrix_text = match.group(1)
    rows = matrix_text.strip().split(';')

    matrix = []
    for row in rows:
        row = row.strip()
        if row:
            matrix.append([float(x) for x in row.split()])

    return np.array(matrix)


# -------------------------------------------------
# Main graph builder (SAFE for large grids)
# -------------------------------------------------
def build_graph_from_matpower_m(m_file, output_file):
    # -----------------------------
    # STEP 1: Load bus & branch
    # -----------------------------
    bus = extract_matrix(m_file, "mpc.bus")
    branch = extract_matrix(m_file, "mpc.branch")

    # -----------------------------
    # STEP 2: Node features
    # -----------------------------
    # Vm, Pd, Qd
    Vm = bus[:, 7]
    Pd = bus[:, 2]
    Qd = bus[:, 3]

    x = torch.tensor(
        np.column_stack((Vm, Pd, Qd)),
        dtype=torch.float
    )

    # -----------------------------
    # STEP 3: BUS-ID REMAPPING (CRITICAL FIX)
    # -----------------------------
    # Original bus numbers (may be non-contiguous!)
    bus_ids = bus[:, 0].astype(int)

    # Map: bus_id -> continuous index [0 ... N-1]
    bus_id_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    # -----------------------------
    # STEP 4: Edge index (SAFE)
    # -----------------------------
    from_bus_raw = branch[:, 0].astype(int)
    to_bus_raw   = branch[:, 1].astype(int)

    from_bus = [bus_id_map[b] for b in from_bus_raw]
    to_bus   = [bus_id_map[b] for b in to_bus_raw]

    edge_index = np.vstack([
        np.concatenate([from_bus, to_bus]),
        np.concatenate([to_bus, from_bus])
    ])

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # -----------------------------
    # STEP 5: Labels
    # -----------------------------
    # Voltage violation rule
    y = torch.tensor(
        ((Vm < 0.95) | (Vm > 1.05)).astype(int),
        dtype=torch.long
    )

    # -----------------------------
    # STEP 6: Create graph object
    # -----------------------------
    data = Data(x=x, edge_index=edge_index, y=y)

    # -----------------------------
    # STEP 7: Sanity check (optional but useful)
    # -----------------------------
    assert edge_index.max().item() < x.shape[0], \
        "edge_index contains invalid node indices!"

    # -----------------------------
    # STEP 8: Save graph
    # -----------------------------
    torch.save(data, output_file)
    print(f"✅ Graph saved successfully: {output_file}")
    print(f"   Nodes: {data.num_nodes}, Edges: {data.num_edges}")


# -------------------------------------------------
# Run from command line
# -------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

    # 🔁 CHANGE THIS FILE NAME AS NEEDED
    # Examples:
    #   case14.m
    #   case30.m
    #   case118.m
    #   pglib_opf_case118_ieee.m
    CASE_FILE = "pglib_case300.m"

    m_file = os.path.join(PROJECT_ROOT, "Data", CASE_FILE)
    output_file = os.path.join(
        PROJECT_ROOT,
        "Graph",
        f"graph_{CASE_FILE.replace('.m','')}.pt"
    )

    print("📂 Loading:", m_file)
    build_graph_from_matpower_m(m_file, output_file)
