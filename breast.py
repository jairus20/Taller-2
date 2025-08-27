# logistic_breast_cancer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Ruta al CSV de Kaggle (uciml/breast-cancer-wisconsin-data)
df = pd.read_csv("Taller 2/Data/data.csv")

# Eliminar columnas irrelevantes (id, unnamed, etc. si existen)
if "id" in df.columns:
    df = df.drop(columns=["id"])
if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

# Target y features
y = df["diagnosis"].map({"M": 1, "B": 0})  # Maligno=1, Benigno=0
X = df.drop(columns=["diagnosis"])

# Preprocesamiento + Modelo
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("[Regresión Logística - Breast Cancer Wisconsin]")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc:.4f}")
print("Confusion Matrix:")
print(cm)
