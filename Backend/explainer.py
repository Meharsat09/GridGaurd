import torch
import os
from config import GRAPH_DIR, GRAPH_MAP
from model_loader import load_model

# Try to import GNNExplainer - handle different PyG versions
try:
    from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlgo
    USE_NEW_API = True
except ImportError:
    try:
        from torch_geometric.nn import GNNExplainer
        USE_NEW_API = False
    except ImportError:
        USE_NEW_API = None

# Feature names for interpretation
FEATURE_NAMES = ["voltage", "load", "degree"]


def _load_graph(grid_name: str):
    if grid_name not in GRAPH_MAP:
        raise ValueError("Invalid grid name")

    graph = torch.load(
        os.path.join(GRAPH_DIR, GRAPH_MAP[grid_name]),
        weights_only=False
    )

    graph.x = graph.x.float()
    graph.edge_index = graph.edge_index.long()

    return graph


def _apply_scenario_to_graph(graph, load_factor: float, voltage_drop: float):
    graph.x = graph.x.clone()

    if graph.x.size(1) >= 1:
        graph.x[:, 0] = graph.x[:, 0] * voltage_drop

    if graph.x.size(1) >= 2:
        graph.x[:, 1] = torch.clamp(graph.x[:, 1] * load_factor, max=1.0)

    return graph


def _build_explanation(model, graph, node_id: int):
    with torch.no_grad():
        out = model(graph)
        probs = torch.softmax(out, dim=1)
        predicted_class = out[node_id].argmax().item()
        confidence = probs[node_id][predicted_class].item()

    feature_importance = explain_with_gradients(model, graph, node_id, predicted_class)
    feature_importance = feature_importance / (feature_importance.sum() + 1e-10)

    feature_dict = {}
    for i, feat_name in enumerate(FEATURE_NAMES[:len(feature_importance)]):
        feature_dict[feat_name] = round(float(feature_importance[i]), 4)

    sorted_features = sorted(
        feature_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = [
        {"feature": feat, "importance": score}
        for feat, score in sorted_features
    ]

    textual_explanation = generate_textual_explanation(
        top_features,
        predicted_class,
        graph.x[node_id].detach().numpy()
    )

    return {
        "node_id": node_id,
        "predicted_label": "Failure-prone" if predicted_class == 1 else "Safe",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "feature_importance": feature_dict,
        "top_features": top_features,
        "textual_explanation": textual_explanation
    }


def explain_node(grid_name: str, node_id: int):
    """
    Generate GNNExplainer-based explanation for a specific node prediction.
    
    Args:
        grid_name: Grid identifier (e.g., 'ieee14')
        node_id: Node index to explain
    
    Returns:
        Dictionary with feature importance and textual explanation
    """
    graph = _load_graph(grid_name)

    if node_id < 0 or node_id >= graph.num_nodes:
        raise ValueError(f"Invalid node_id. Must be between 0 and {graph.num_nodes - 1}")
    
    model = load_model(graph.x.size(1))
    return _build_explanation(model, graph, node_id)


def explain_node_scenario(grid_name: str, node_id: int, load_factor: float = 1.0, voltage_drop: float = 1.0):
    graph = _load_graph(grid_name)

    if node_id < 0 or node_id >= graph.num_nodes:
        raise ValueError(f"Invalid node_id. Must be between 0 and {graph.num_nodes - 1}")

    graph = _apply_scenario_to_graph(graph, load_factor, voltage_drop)
    model = load_model(graph.x.size(1))

    result = _build_explanation(model, graph, node_id)
    result["scenario"] = True
    result["load_factor"] = float(load_factor)
    result["voltage_drop"] = float(voltage_drop)

    return result


def explain_with_new_api(model, graph, node_id):
    """Use new PyG Explainer API"""
    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainerAlgo(epochs=100),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )
        
        explanation = explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            index=node_id
        )
        
        return explanation.node_mask[node_id].detach().numpy()
    except Exception as e:
        print(f"New API failed: {e}")
        raise


def explain_with_old_api(model, graph, node_id):
    """Use old PyG GNNExplainer API"""
    try:
        explainer = GNNExplainer(model, epochs=100, return_type='raw')
        node_feat_mask, edge_mask = explainer.explain_node(node_id, graph.x, graph.edge_index)
        return node_feat_mask.detach().numpy()
    except Exception as e:
        print(f"Old API failed: {e}")
        raise


def explain_with_gradients(model, graph, node_id, target_class):
    """Fallback: Use gradient-based feature attribution"""
    model.eval()
    
    # Create a fresh copy of features with gradient tracking
    x_grad = graph.x.detach().clone().requires_grad_(True)
    
    # Forward pass with gradient-enabled features
    out = model(type('Data', (), {'x': x_grad, 'edge_index': graph.edge_index})())
    
    # Compute gradients with respect to node features
    target_output = out[node_id, target_class]
    target_output.backward()
    
    # Get gradients for the specific node
    gradients = x_grad.grad[node_id].abs().detach().numpy()
    
    return gradients


def generate_textual_explanation(top_features, predicted_class, node_features):
    """
    Generate human-readable explanation based on feature importance.
    
    Args:
        top_features: List of features sorted by importance
        predicted_class: Predicted class (0=Safe, 1=Failure-prone)
        node_features: Raw feature values for the node
    
    Returns:
        Human-readable explanation string
    """
    if not top_features:
        return "Unable to generate explanation."
    
    top_feat = top_features[0]
    feat_name = top_feat["feature"]
    importance = top_feat["importance"]
    
    # Get feature value
    feat_idx = FEATURE_NAMES.index(feat_name)
    feat_value = round(float(node_features[feat_idx]), 3)
    
    # Build explanation
    if predicted_class == 1:  # Failure-prone
        explanation = f"The model predicted this node as <b>failure-prone</b>. "
        
        if feat_name == "voltage":
            if feat_value < 0.95:
                explanation += f"<b>Low voltage ({feat_value})</b> was the primary contributor (importance: {importance:.2%}). "
            elif feat_value > 1.05:
                explanation += f"<b>High voltage ({feat_value})</b> was the primary contributor (importance: {importance:.2%}). "
            else:
                explanation += f"<b>Voltage deviation ({feat_value})</b> significantly influenced the prediction (importance: {importance:.2%}). "
        
        elif feat_name == "load":
            if feat_value > 0.7:
                explanation += f"<b>High load ({feat_value})</b> was the primary contributor (importance: {importance:.2%}). "
            else:
                explanation += f"<b>Load level ({feat_value})</b> significantly influenced the prediction (importance: {importance:.2%}). "
        
        elif feat_name == "degree":
            if feat_value > 5:
                explanation += f"<b>High connectivity (degree: {int(feat_value)})</b> was the primary contributor (importance: {importance:.2%}). "
            else:
                explanation += f"<b>Network position (degree: {int(feat_value)})</b> significantly influenced the prediction (importance: {importance:.2%}). "
        
        # Add secondary factor if present
        if len(top_features) > 1:
            second_feat = top_features[1]
            explanation += f"Additionally, <b>{second_feat['feature']}</b> contributed {second_feat['importance']:.2%} to the decision."
    
    else:  # Safe
        explanation = f"The model predicted this node as <b>safe</b>. "
        explanation += f"The node's <b>{feat_name}</b> (importance: {importance:.2%}) and other features indicate stable operation."
    
    return explanation
