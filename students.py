# polynomial_students.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Ruta al CSV de Kaggle (spscientist/students-performance-in-exams)
df = pd.read_csv("Taller 2\Data\StudentsPerformance.csv")

# Renombrar columnas si vienen con espacios/capitalización diferente
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Target y features (predicimos math_score)
target_col = "math_score"
y = df[target_col]
X = df.drop(columns=[target_col])

# Identificar columnas (pueden variar en tu archivo)
num_cols = [c for c in X.columns if "score" in c]  # reading_score, writing_score
cat_cols = [c for c in X.columns if c not in num_cols]

# Preprocesamiento
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=3, include_bias=False))  # Ajusta grado (2–4)
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# Modelo
model = Pipeline([
    ("pre", pre),
    ("lr", LinearRegression())
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenamiento
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("[Regresión Polinomial - Students]")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")
