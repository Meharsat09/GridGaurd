# GridGaurd

GridGaurd is a graph neural network (GNN) framework for power-grid failure prediction.
It treats a power grid as a graph (buses as nodes, lines as edges) and estimates
node-level risk of instability or failure. The project includes a FastAPI backend
for inference and explanation, plus a lightweight HTML/JS frontend for interactive
visualization and stress scenarios.

## What This Project Is About

Power grids are complex systems where local issues can cascade into broader
failures. GridGaurd aims to provide an interpretable, data-driven decision
support tool that highlights risky nodes, summarizes grid-level risk, and lets
users explore how stress conditions (higher load, lower voltage) shift risk.

The goal is not to replace traditional power flow tools, but to offer a fast,
graph-based screening layer for monitoring and what-if analysis.

## Features

- GNN-based risk prediction for IEEE and PGLib grids
- Scenario stress analysis (load/voltage perturbations)
- Node-level explanations via the backend
- Simple browser UI with risk and topology views

## How It Works (High Level)

- Convert MATPOWER case files into graph data with node features
- Train a GraphSAGE model on multiple grid cases
- Run inference to produce per-node risk scores and labels
- Visualize results in the UI and explore scenario stress effects

## Project Structure

- Backend/ - FastAPI app, inference, and explanation logic
- Frontend/ - Static UI (index.html, script.js, style.css)
- Scripts/ - Graph building and GNN training utilities
- Data/ - MATPOWER grid cases
- Graph/ - Serialized graph data (.pt)
- Model/ - Trained model weights

## Data and Labels

- Inputs come from MATPOWER .m files (bus and branch matrices)
- Node features include voltage magnitude and load values
- Labels are derived from voltage violations (rule-based)

## Backend API (high level)

- GET /predict/{grid_name}
- GET /predict_scenario/{grid_name}?load_factor=...&voltage_drop=...
- GET /explain/{grid_name}/{node_id}

Supported grid names include: ieee14, ieee30, ieee118, pglib118, pglib300.

## Frontend

The UI renders nodes and edges on a canvas with risk coloring, summaries, and
node details. A scenario panel allows load and voltage stress to see how the
grid risk profile changes.

## Notes

- The backend expects graph files in Graph/ and model weights in Model/.
- Training and graph generation scripts are in Scripts/.

## Limitations

- Labels are based on a simple voltage threshold rule
- Results are intended for research/demo use, not operational dispatch
- Model quality depends on the available grid cases and feature set

