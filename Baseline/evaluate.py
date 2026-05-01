import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prepare_dataset import load_dataset

# -----------------------------
# Load dataset
# -----------------------------
X, y = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== DATASET DISTRIBUTION ===")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")

# -----------------------------
# Logistic Regression (Balanced)
# -----------------------------
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\n=== Logistic Regression (Balanced) ===")
print(classification_report(y_test, log_pred, digits=3))

# -----------------------------
# Random Forest (Balanced)
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n=== Random Forest (Balanced) ===")
print(classification_report(y_test, rf_pred, digits=3))
