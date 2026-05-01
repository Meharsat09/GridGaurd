# GridGaurd

GridGaurd is a graph neural network (GNN) framework for power-grid failure prediction.
It includes a FastAPI backend for inference and explanation, plus a lightweight
HTML/JS frontend for interactive visualization and stress scenarios.

## Features

- GNN-based risk prediction for IEEE and PGLib grids
- Scenario stress analysis (load/voltage perturbations)
- Node-level explanations via the backend
- Simple browser UI with risk and topology views

## Project Structure

- Backend/ - FastAPI app, inference, and explanation logic
- Frontend/ - Static UI (index.html, script.js, style.css)
- Scripts/ - Graph building and GNN training utilities
- Data/ - MATPOWER grid cases
- Graph/ - Serialized graph data (.pt)
- Model/ - Trained model weights

## Backend API (high level)

- GET /predict/{grid_name}
- GET /predict_scenario/{grid_name}?load_factor=...&voltage_drop=...
- GET /explain/{grid_name}/{node_id}

Supported grid names include: ieee14, ieee30, ieee118, pglib118, pglib300.

## Notes

- The backend expects graph files in Graph/ and model weights in Model/.
- Training and graph generation scripts are in Scripts/.

