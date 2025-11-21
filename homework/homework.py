# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import json
import gzip
import pickle
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================================
# PASO 1: FUNCIONES DE CARGA Y LIMPIEZA DE DATOS
# ============================================================================

def _leer_zip_csv(ruta_zip: str, nombre_interno: str) -> pd.DataFrame:
    """
    Lee un archivo CSV desde dentro de un archivo ZIP.
    
    Args:
        ruta_zip: Ruta al archivo ZIP
        nombre_interno: Nombre del archivo CSV dentro del ZIP
        
    Returns:
        DataFrame con los datos leídos
    """
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        with zf.open(nombre_interno) as f:
            return pd.read_csv(f)


def _depurar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el dataset según los requisitos del Paso 1.
    
    Realizaciones:
    - Elimina la columna 'ID'
    - Renombra 'default payment next month' a 'default'
    - Remueve registros con valores faltantes (NaN)
    - Elimina educación y matrimonio no disponibles (0)
    - Agrupa educación > 4 en categoría 'others' (4)
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpiado
    """
    # Crear copia para evitar modificar el original
    out = df.copy()
    
    # Eliminar columna ID innecesaria
    out = out.drop("ID", axis=1)
    
    # Renombrar columna objetivo a 'default'
    out = out.rename(columns={"default payment next month": "default"})
    
    # Eliminar registros con valores faltantes
    out = out.dropna()
    
    # Filtrar registros con EDUCATION=0 (N/A) y MARRIAGE=0 (N/A)
    out = out[(out["EDUCATION"] != 0) & (out["MARRIAGE"] != 0)]
    
    # Agrupar educación > 4 en categoría 4 (others)
    out.loc[out["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return out


# ============================================================================
# PASO 2 Y 3: CREACIÓN DEL PIPELINE DE CLASIFICACIÓN
# ============================================================================

def _armar_busqueda() -> GridSearchCV:
    """
    Construye un pipeline completo de clasificación (Paso 3) y configura
    la búsqueda de hiperparámetros (Paso 4).
    
    Pipeline:
    1. Preprocesamiento (ColumnTransformer):
       - OneHotEncoding para variables categóricas (SEX, EDUCATION, MARRIAGE)
       - StandardScaler para variables numéricas
    2. PCA: Descomposición de matriz con todas las componentes
    3. SelectKBest: Selección de K características más relevantes
    4. SVC: Máquina de Vectores de Soporte
    
    Validación cruzada:
    - 10 splits para cross-validation
    - Scoring: balanced_accuracy (métrica adecuada para datos desbalanceados)
    
    Returns:
        GridSearchCV configurado con el pipeline
    """
    
    # Definir columnas categóricas para one-hot encoding
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Definir columnas numéricas para estandarización
    num_cols = [
        "LIMIT_BAL", "AGE",  # Datos demográficos
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",  # Historial de pagos
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",  # Montos facturados
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",  # Montos pagados
    ]

    # Crear transformador de preprocesamiento (capas 1-3)
    preprocess = ColumnTransformer(
        transformers=[
            # Aplicar OneHotEncoder a variables categóricas
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            # Aplicar StandardScaler a variables numéricas
            ("std", StandardScaler(), num_cols),
        ],
        remainder="passthrough",  # Mantener otras columnas sin cambios
    )

    # Construir el pipeline completo (Paso 3)
    pipe = Pipeline(steps=[
        ("prep", preprocess),  # Preprocesamiento
        ("pca", PCA()),  # PCA con todas las componentes (se optimiza en grid)
        ("kbest", SelectKBest(score_func=f_classif)),  # Selección de K mejores características
        ("svc", SVC(kernel="rbf", random_state=42)),  # SVM con kernel RBF
    ])

    # Definir grid de hiperparámetros para optimizar (Paso 4)
    grid = {
        "pca__n_components": [20, 21],  # Número de componentes PCA
        "kbest__k": [12],  # Número de características a seleccionar
        "svc__kernel": ["rbf"],  # Kernel de SVM
        "svc__gamma": [0.099],  # Parámetro gamma para kernel RBF
    }

    # Retornar GridSearchCV con validación cruzada de 10 splits
    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=10,  # 10 splits para cross-validation
        refit=True,  # Reentrena con mejor configuración
        verbose=1,  # Mostrar progreso
        return_train_score=False,
        scoring="balanced_accuracy",  # Métrica para datos desbalanceados
    )


# ============================================================================
# PASO 6: FUNCIONES PARA CALCULAR MÉTRICAS
# ============================================================================

def _metricas(nombre: str, y_true, y_pred) -> dict:
    """
    Calcula métricas de precisión del modelo (Paso 6).
    
    Métricas calculadas:
    - precision: (TP) / (TP + FP)
    - balanced_accuracy: media de recall para cada clase
    - recall: (TP) / (TP + FN)
    - f1_score: media armónica entre precisión y recall
    
    Args:
        nombre: Identificador del conjunto (train/test)
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        
    Returns:
        Diccionario con métricas calculadas
    """
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


# ============================================================================
# PASO 7: FUNCIONES PARA CALCULAR MATRIZ DE CONFUSIÓN
# ============================================================================

def _cm(nombre: str, y_true, y_pred) -> dict:
    """
    Calcula la matriz de confusión (Paso 7).
    
    Estructura:
    - true_0: Registros reales con clase 0
      * predicted_0: Verdaderos negativos (TN)
      * predicted_1: Falsos positivos (FP)
    - true_1: Registros reales con clase 1
      * predicted_0: Falsos negativos (FN)
      * predicted_1: Verdaderos positivos (TP)
    
    Args:
        nombre: Identificador del conjunto (train/test)
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        
    Returns:
        Diccionario con matriz de confusión formateada
    """
    # Desempacar matriz de confusión: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {
            "predicted_0": int(tn),  # Verdaderos negativos
            "predicted_1": int(fp),  # Falsos positivos
        },
        "true_1": {
            "predicted_0": int(fn),  # Falsos negativos
            "predicted_1": int(tp),  # Verdaderos positivos
        },
    }


# ============================================================================
# PASO 5: FUNCIONES PARA GUARDAR MODELO Y RESULTADOS
# ============================================================================

def _guardar_modelo(objeto) -> None:
    """
    Guarda el modelo entrenado comprimido con gzip (Paso 5).
    
    Args:
        objeto: Objeto del modelo (GridSearchCV) a guardar
    """
    # Crear directorio si no existe
    Path("files/models").mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo comprimido con gzip
    with gzip.open("files/models/model.pkl.gz", "wb") as fh:
        pickle.dump(objeto, fh)


def _guardar_jsonl(registros: list[dict]) -> None:
    """
    Guarda métricas en formato JSONL (JSON Lines) (Paso 6 y 7).
    
    Cada línea es un diccionario JSON independiente con:
    - Métricas de precisión
    - Matriz de confusión
    
    Args:
        registros: Lista de diccionarios con métricas y matrices
    """
    # Crear directorio si no existe
    Path("files/output").mkdir(parents=True, exist_ok=True)
    
    # Escribir cada registro como una línea JSON
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r) + "\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Definir rutas de entrada
    test_zip = "files/input/test_data.csv.zip"
    train_zip = "files/input/train_data.csv.zip"
    interno_test = "test_default_of_credit_card_clients.csv"
    interno_train = "train_default_of_credit_card_clients.csv"

    # PASO 1: Cargar y limpiar datos
    print("Cargando y limpiando datos...")
    df_test = _depurar(_leer_zip_csv(test_zip, interno_test))
    df_train = _depurar(_leer_zip_csv(train_zip, interno_train))
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # PASO 2: Dividir features y target
    X_tr, y_tr = df_train.drop("default", axis=1), df_train["default"]
    X_te, y_te = df_test.drop("default", axis=1), df_test["default"]

    # PASO 3 y 4: Crear pipeline y optimizar hiperparámetros
    print("Entrenando modelo con GridSearchCV...")
    search = _armar_busqueda()
    search.fit(X_tr, y_tr)
    print(f"Mejor score CV: {search.best_score_:.4f}")
    print(f"Mejor parámetros: {search.best_params_}")

    # PASO 5: Guardar modelo
    print("Guardando modelo...")
    _guardar_modelo(search)

    # Realizar predicciones en train y test
    print("Generando predicciones...")
    y_tr_pred = search.predict(X_tr)
    y_te_pred = search.predict(X_te)

    # PASO 6: Calcular métricas
    print("Calculando métricas...")
    train_metrics = _metricas("train", y_tr, y_tr_pred)
    test_metrics = _metricas("test", y_te, y_te_pred)

    # PASO 7: Calcular matrices de confusión
    print("Calculando matrices de confusión...")
    train_cm = _cm("train", y_tr, y_tr_pred)
    test_cm = _cm("test", y_te, y_te_pred)

    # Guardar resultados
    print("Guardando resultados...")
    _guardar_jsonl([train_metrics, test_metrics, train_cm, test_cm])
    print("¡Proceso completado exitosamente!")
