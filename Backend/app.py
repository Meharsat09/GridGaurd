from fastapi import FastAPI, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from inference import run_inference
from config import RISK_THRESHOLD
from explainer import explain_node, explain_node_scenario

app = FastAPI(
    title="Power Grid Failure Prediction Backend",
    description="GNN-based decision support API",
    version="1.0"
)

# 🔓 Allow frontend to access backend (for local demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (safe for local demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Backend running"}

@app.get("/predict/{grid_name}")
def predict(grid_name: str):
    try:
        return run_inference(grid_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict_scenario/{grid_name}")
def predict_scenario(grid_name: str, load_factor: float = 1.0, voltage_drop: float = 1.0):
    try:
        # Run baseline prediction
        result = run_inference(grid_name)
        
        # Apply stress factors to create stressed scenario
        stressed_nodes = []
        for node in result["nodes"]:
            stressed_node = node.copy()
            
            # Apply stress: increase load and reduce voltage
            stressed_node["electrical"]["load"] = min(1.0, node["electrical"]["load"] * load_factor)
            stressed_node["electrical"]["voltage"] = node["electrical"]["voltage"] * voltage_drop
            stressed_node["electrical"]["voltage_deviation"] = abs(stressed_node["electrical"]["voltage"] - 1.0)
            
            # Recalculate stress score
            stressed_node["stress_score"] = round(
                0.4 * stressed_node["electrical"]["voltage_deviation"] +
                0.4 * stressed_node["electrical"]["load"] +
                0.2 * node["neighbors"]["avg_neighbor_risk"],
                3
            )
            
            # Update risk based on stress (simplified heuristic)
            stress_increase = stressed_node["stress_score"] - node["stress_score"]
            new_risk_score = min(1.0, node["risk_score"] + stress_increase * 0.5)
            stressed_node["risk_score"] = round(new_risk_score, 4)
            
            # Update risk category
            if new_risk_score >= 0.7:
                stressed_node["risk_category"] = "High"
            elif new_risk_score >= 0.4:
                stressed_node["risk_category"] = "Medium"
            else:
                stressed_node["risk_category"] = "Low"

            # Update prediction label for scenario view
            stressed_node["prediction"] = (
                "Failure-prone" if new_risk_score >= RISK_THRESHOLD else "Safe"
            )
            
            stressed_nodes.append(stressed_node)
        
        # Recalculate grid summary for stressed scenario
        high = sum(1 for n in stressed_nodes if n["risk_category"] == "High")
        medium = sum(1 for n in stressed_nodes if n["risk_category"] == "Medium")
        low = sum(1 for n in stressed_nodes if n["risk_category"] == "Low")
        
        avg_risk = sum(n["risk_score"] for n in stressed_nodes) / len(stressed_nodes)
        
        stressed_summary = {
            "grid_risk_index": round(avg_risk, 3),
            "risk_distribution": {
                "low": low,
                "medium": medium,
                "high": high
            },
            "summary_text": (
                "Stressed scenario: Grid shows increased risk under load/voltage stress."
                if high > result["grid_summary"]["risk_distribution"]["high"]
                else "Stressed scenario: Grid maintains stability under moderate stress."
            )
        }
        
        return {
            "grid": grid_name,
            "baseline_summary": result["grid_summary"],
            "stressed_summary": stressed_summary,
            "nodes": stressed_nodes,
            "edges": result["edges"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/explain/{grid_name}/{node_id}")
def explain(grid_name: str, node_id: int, load_factor: Optional[float] = None, voltage_drop: Optional[float] = None):
    """
    Generate GNNExplainer-based explanation for a specific node prediction.
    
    Args:
        grid_name: Grid identifier (e.g., 'ieee14')
        node_id: Node index to explain
    
    Returns:
        Feature importance scores and human-readable explanation
    """
    try:
        if load_factor is not None or voltage_drop is not None:
            return explain_node_scenario(
                grid_name,
                node_id,
                load_factor=load_factor or 1.0,
                voltage_drop=voltage_drop or 1.0
            )

        return explain_node(grid_name, node_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
