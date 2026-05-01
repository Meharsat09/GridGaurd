import torch
import os
from config import GRAPH_DIR, GRAPH_MAP, RISK_THRESHOLD
from model_loader import load_model

# -----------------------------
# Thresholds & constants
# -----------------------------
V_NOMINAL = 1.0
LOAD_HIGH = 0.75
TOP_K = 5


def risk_category(risk):
    if risk < 0.4:
        return "Low"
    elif risk < RISK_THRESHOLD:
        return "Medium"
    else:
        return "High"


def node_narrative(node):
    reasons = []

    if node["electrical"]["voltage_deviation"] > 0.05:
        reasons.append("voltage deviation")

    if node["electrical"]["load"] > LOAD_HIGH:
        reasons.append("high load")

    if node["neighbors"]["high_risk_neighbors"] > 0:
        reasons.append("stressed neighboring nodes")

    if not reasons:
        return "Node is operating within normal limits."

    return (
        f"Node is classified as {node['risk_category'].lower()} risk due to "
        + ", ".join(reasons) + "."
    )


def run_inference(grid_name: str):
    if grid_name not in GRAPH_MAP:
        raise ValueError("Invalid grid name")

    graph = torch.load(
        os.path.join(GRAPH_DIR, GRAPH_MAP[grid_name]),
        weights_only=False
    )

    graph.x = graph.x.float()
    graph.edge_index = graph.edge_index.long()

    model = load_model(graph.x.size(1))

    with torch.no_grad():
        probs = torch.softmax(model(graph), dim=1)

    # -----------------------------
    # Neighbor list
    # -----------------------------
    neighbors = {i: [] for i in range(graph.num_nodes)}
    for s, d in graph.edge_index.t().tolist():
        neighbors[s].append(d)

    nodes = []
    risk_values = []

    for i in range(graph.num_nodes):
        risk = probs[i][1].item()
        risk_values.append(risk)

        voltage = graph.x[i][0].item()
        load = graph.x[i][1].item() if graph.x.size(1) > 1 else 0.0
        v_dev = abs(voltage - V_NOMINAL)

        nbrs = neighbors[i]
        nbr_risks = [probs[n][1].item() for n in nbrs]
        avg_nbr_risk = sum(nbr_risks) / len(nbr_risks) if nbr_risks else 0.0
        high_risk_nbrs = sum(1 for r in nbr_risks if r >= RISK_THRESHOLD)

        degree = len(nbrs)

        stress_score = (
            0.4 * v_dev +
            0.4 * load +
            0.2 * avg_nbr_risk
        )

        node = {
            "id": i,
            "risk_score": round(risk, 4),
            "risk_category": risk_category(risk),
            "prediction": "Failure-prone" if risk >= RISK_THRESHOLD else "Safe",

            "electrical": {
                "voltage": round(voltage, 3),
                "voltage_deviation": round(v_dev, 3),
                "load": round(load, 3)
            },

            "topology": {
                "degree": degree
            },

            "neighbors": {
                "count": degree,
                "avg_neighbor_risk": round(avg_nbr_risk, 3),
                "high_risk_neighbors": high_risk_nbrs
            },

            "stress_score": round(stress_score, 3)
        }

        node["narrative"] = node_narrative(node)
        nodes.append(node)

    # -----------------------------
    # Phase-2: Grid-level metrics
    # -----------------------------
    high = sum(1 for n in nodes if n["risk_category"] == "High")
    medium = sum(1 for n in nodes if n["risk_category"] == "Medium")
    low = sum(1 for n in nodes if n["risk_category"] == "Low")

    top_nodes = sorted(
        nodes,
        key=lambda x: x["risk_score"],
        reverse=True
    )[:TOP_K]

    critical_nodes = [
        n["id"] for n in nodes
        if n["risk_category"] == "High" and n["topology"]["degree"] >= 4
    ]

    grid_risk_index = sum(sorted(risk_values, reverse=True)[:max(1, graph.num_nodes // 10)]) / max(1, graph.num_nodes // 10)

    grid_summary_text = (
        "The grid is currently stable with localized stress."
        if high < graph.num_nodes * 0.2
        else "The grid shows significant stress and requires attention."
    )

    # -----------------------------
    # Final response
    # -----------------------------
    return {
        "grid": grid_name,
        "num_nodes": graph.num_nodes,
        "threshold": RISK_THRESHOLD,

        "grid_summary": {
            "grid_risk_index": round(grid_risk_index, 3),
            "risk_distribution": {
                "low": low,
                "medium": medium,
                "high": high
            },
            "critical_nodes": critical_nodes,
            "summary_text": grid_summary_text
        },

        "top_risky_nodes": [
            {
                "id": n["id"],
                "risk_score": n["risk_score"]
            } for n in top_nodes
        ],

        "nodes": nodes,

        "edges": [
            {"source": s, "target": d}
            for s, d in graph.edge_index.t().tolist()
        ]
    }
