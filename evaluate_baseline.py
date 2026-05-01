import os
import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("🚀 Starting Baseline Evaluation...")

# -------------------------------
# Load graphs
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(BASE_DIR)

graph_paths = [
    os.path.join(PROJECT_ROOT, "Graph", "graph_case14.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case30.pt"),
    os.path.join(PROJECT_ROOT, "Graph", "graph_case118.pt")
]

X = []
y = []

for path in graph_paths:
    print(f"📂 Loading: {path}")
    # These .pt files store full Data objects; allow full unpickling.
    g = torch.load(path, weights_only=False)
    X.append(g.x.cpu().numpy())
    y.append(g.y.cpu().numpy())

X = np.vstack(X)
y = np.hstack(y)

print("✅ Data prepared")

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Logistic Regression
# -------------------------------
print("\n🔹 Logistic Regression")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_lr, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_lr, zero_division=0))

# -------------------------------
# Random Forest
# -------------------------------
print("\n🔹 Random Forest")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_rf, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_rf, zero_division=0))