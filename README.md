# Predicción de Enfermedades Coronarias - Solución para PhawAI

## Descripción del Problema
El desafío consiste en desarrollar un modelo de aprendizaje automático para predecir la presencia de enfermedades coronarias en individuos, utilizando un conjunto de datos proporcionado por el **Behavioral Risk Factor Surveillance System (BRFSS) de 2022**. 

El conjunto de datos contiene información demográfica, comportamientos de salud y condiciones previas diagnosticadas, con un **desequilibrio de clases significativo** entre los casos positivos y negativos de enfermedad coronaria. El objetivo es optimizar la métrica **F1-Score** para mejorar la precisión y el recall del modelo en la identificación de individuos con enfermedades coronarias.

---

## Enfoque y Solución
### 1. **Preprocesamiento de los Datos**
- **Imputación de valores nulos**: Se usó `SimpleImputer` para reemplazar valores nulos con la mediana de cada característica.
- **Estandarización**: Se utilizó `StandardScaler` para escalar las características y garantizar que todas tengan la misma importancia relativa en el modelo.

### 2. **Balanceo de Clases**
Debido al desequilibrio de clases en el conjunto de datos, se utilizó **SMOTE (Synthetic Minority Oversampling Technique)** para generar ejemplos sintéticos de la clase minoritaria y balancear las clases.

### 3. **Modelo Usado**
Se utilizó un modelo de **Random Forest Classifier** con los siguientes parámetros:
- `n_estimators=300` (300 árboles en el bosque).
- `max_depth=20` (profundidad máxima de cada árbol).
- `class_weight="balanced"` para tratar el desequilibrio de clases sin sobreajustar.

### 4. **Optimización del Umbral**
El umbral óptimo para maximizar el F1-Score se fijó en **0.2**, lo que mejora el equilibrio entre precisión y recall.

---

## Métrica de Validación
El modelo logró un **F1-Score en validación de 0.9575** con el umbral de 0.2, lo que lo posiciona dentro del rango competitivo del desafío.

---

## Pasos para Ejecutar el Código
1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd phawAI-prediccion-coronaria
   ```

2. **Instalar las dependencias**:
   Asegúrate de tener Python 3.8 o superior. Instala las dependencias con:
   ```bash
   pip install -r requirements.txt
   ```

3. **Colocar los datos**:
   - Descarga los archivos `train.csv`, `test_private.csv` y `test_public.csv` del desafío y colócalos en la misma carpeta que el script.

4. **Ejecutar el script**:
   Corre el script principal para generar el archivo de predicciones:
   ```bash
   python prueba.py
   ```

5. **Resultado**:
   El archivo de predicciones combinado se guardará como `resultados_test_combinado.csv` en la carpeta principal. Este archivo incluye las predicciones para los conjuntos `test_public` y `test_private`.

---

## Archivos Incluidos
- `prueba.py`: Script principal para preprocesar los datos, entrenar el modelo y generar las predicciones.
- `requirements.txt`: Lista de dependencias necesarias.
- `README.md`: Documentación del proyecto.
- `resultados_test_combinado.csv` (opcional): Archivo de predicciones generado por el modelo.
- `train.csv`: Conjunto de datos de entrenamiento, utilizado para ajustar y entrenar los modelos.
- `test_public`: Conjunto de datos de prueba con etiquetas, para realizar evaluaciones preliminares del modelo.
- `test_private`: Conjunto de datos de prueba sin etiquetas, destinado a la evaluación final del modelo.

---

## Observaciones Finales
Este modelo se diseñó para ser eficiente y efectivo en la predicción de enfermedades coronarias, utilizando técnicas avanzadas de balanceo de clases y un modelo robusto de Random Forest. Si tienes preguntas o sugerencias, ¡no dudes en contribuir o abrir un issue en el repositorio!
