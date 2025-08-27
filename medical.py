# linear_insurance.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Ruta al CSV de Kaggle (mirichoi0218/insurance)
df = pd.read_csv("Taller 2/Data/insurance.csv")

# Target y features
y = df["charges"]
X = df.drop(columns=["charges"])

# Columnas
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# Preprocesamiento
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
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

print("[Regresión Lineal - Insurance]")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

# Validación cruzada opcional
cv_rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5)
print(f"CV RMSE (5-fold): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
