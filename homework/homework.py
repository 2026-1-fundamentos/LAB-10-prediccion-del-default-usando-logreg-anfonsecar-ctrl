# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
# Este script prepara los datos, construye un pipeline con regresión logística,
# optimiza sus hiperparámetros con validación cruzada y guarda el modelo junto
# con las métricas y matrices de confusión resultantes.

import gzip
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "files" / "input"
MODELS_DIR = ROOT_DIR / "files" / "models"
OUTPUT_DIR = ROOT_DIR / "files" / "output"


def main():
    # 1) Carga de los datos de entrenamiento y prueba.
    train_data = pd.read_csv(INPUT_DIR / "train_data.csv.zip", compression="zip")
    test_data = pd.read_csv(INPUT_DIR / "test_data.csv.zip", compression="zip")

    # 2) Limpieza inicial de las columnas y manejo de valores no disponibles.
    train_data = train_data.rename(columns={"default payment next month": "default"})
    test_data = test_data.rename(columns={"default payment next month": "default"})
    train_data.drop(columns=["ID"], inplace=True, errors="ignore")
    test_data.drop(columns=["ID"], inplace=True, errors="ignore")
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # Se eliminan registros con categorías no válidas para EDUCATION y MARRIAGE.
    train_data = train_data[(train_data["EDUCATION"] != 0) & (train_data["MARRIAGE"] != 0)]
    test_data = test_data[(test_data["EDUCATION"] != 0) & (test_data["MARRIAGE"] != 0)]

    # Se agrupan los valores mayores a 4 de EDUCATION en la categoría "others".
    train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    # 3) Separación de variables explicativas y variable objetivo.
    x_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]
    x_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    # 4) Construcción del pipeline de preprocesamiento y modelado.
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", MinMaxScaler()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )

    params = {
        "feature_selection__k": range(1, 11),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__max_iter": [100, 200],
    }

    # 5) Optimización de hiperparámetros con validación cruzada.
    grid_search = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(x_train, y_train)

    # 6) Guardado del modelo entrenado.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(MODELS_DIR / "model.pkl.gz", "wb") as f:
        pickle.dump(grid_search, f)

    # 7) Funciones auxiliares para evaluar el rendimiento del modelo.

    def compute_metrics(y_true, y_pred, dataset):
        return {
            "type": "metrics",
            "dataset": dataset,
            "precision": precision_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
        }

    def confusion_matrix_data(y_true, y_pred, dataset):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
        }

    # 8) Cálculo y guardado de métricas y matrices de confusión.
    train_predictions = grid_search.predict(x_train)
    test_predictions = grid_search.predict(x_test)

    metrics = [
        compute_metrics(y_train, train_predictions, "train"),
        compute_metrics(y_test, test_predictions, "test"),
        confusion_matrix_data(y_train, train_predictions, "train"),
        confusion_matrix_data(y_test, test_predictions, "test"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


if __name__ == "__main__":
    main()
