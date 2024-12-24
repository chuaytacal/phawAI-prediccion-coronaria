import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# Cargar los datos
train_path = "train.csv"  # Cambia esto por la ruta local de tu archivo
test_private_path = "test_private.csv"  # Cambia esto por la ruta local de tu archivo
test_public_path = "test_public.csv"  # Cambia esto por la ruta local de tu archivo

train_data = pd.read_csv(train_path)
test_private = pd.read_csv(test_private_path)
test_public = pd.read_csv(test_public_path)

# Preprocesamiento
# 1. Separar características y etiquetas
X = train_data.drop(columns=["CHD_OR_MI", "ID"])
y = train_data["CHD_OR_MI"]

# 2. Imputar valores nulos
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 3. Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4. Dividir datos para entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Entrenamiento del modelo con Random Forest
model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, class_weight="balanced")
model.fit(X_train_smote, y_train_smote)

# Evaluar el modelo
y_val_pred = model.predict(X_val)
f1 = f1_score(y_val, y_val_pred)
print("F1-Score en validación:", f1)

# Usar directamente el mejor umbral conocido
best_threshold = 0.2
print(f"Usando el umbral fijo: {best_threshold}")

# Preprocesar datos de test_private y test_public
X_test_private = test_private.drop(columns=["ID"], errors='ignore')
X_test_public = test_public.drop(columns=["ID", "CHD_OR_MI"], errors='ignore')

X_test_private_imputed = imputer.transform(X_test_private)
X_test_public_imputed = imputer.transform(X_test_public)

X_test_private_scaled = scaler.transform(X_test_private_imputed)
X_test_public_scaled = scaler.transform(X_test_public_imputed)

# Hacer predicciones con el mejor umbral
probabilities_test_private = model.predict_proba(X_test_private_scaled)[:, 1]
probabilities_test_public = model.predict_proba(X_test_public_scaled)[:, 1]

predictions_private = (probabilities_test_private >= best_threshold).astype(int)
predictions_public = (probabilities_test_public >= best_threshold).astype(int)

# Combinar resultados de test_private y test_public
resultados_private = pd.DataFrame({
    "ID": test_private["ID"],
    "CHD_OR_MI": predictions_private
})
resultados_public = pd.DataFrame({
    "ID": test_public["ID"],
    "CHD_OR_MI": predictions_public
})

resultados_combinados = pd.concat([resultados_private, resultados_public], axis=0)

# Guardar el archivo combinado
output_path = "resultados_test_combinado.csv"  # Cambia esto por la ruta donde quieras guardar el archivo
resultados_combinados.to_csv(output_path, index=False)

print(f"Archivo de predicciones combinado guardado en: {output_path}")
