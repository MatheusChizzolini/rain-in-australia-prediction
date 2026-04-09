from __future__ import annotations

import json
import time
from typing import Any

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import ARTIFACTS_DIR, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from .data import load_dataset, prepare_prediction_input, prepare_training_data


def create_model(input_dim: int) -> Any:
    # Replica exatamente a arquitetura MLP do notebook.
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save_model(csv_path: str) -> dict[str, Any]:
    # Replica o fluxo do notebook: prepara a base, calcula class weights,
    # faz validacao cruzada com 3 folds e treina o modelo final.
    keras.utils.set_random_seed(RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_dataframe = load_dataset(csv_path)
    prepared = prepare_training_data(raw_dataframe)

    classes = np.unique(prepared["y_train"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=prepared["y_train"],
    )
    # O class_weight ajuda a compensar o desbalanceamento entre dias com e sem chuva.
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    x_train_array = prepared["x_train_scaled"].to_numpy()
    y_train_array = prepared["y_train"].to_numpy()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics_val = []
    fold_history_rows = []

    # A validacao cruzada reproduz a analise do notebook antes do treino final.
    for fold, (train_index, val_index) in enumerate(skf.split(x_train_array, y_train_array), start=1):
        print(f"\n{'=' * 30}")
        print(f"Fold {fold}/3")
        print(f"{'=' * 30}")

        x_fold_train = x_train_array[train_index]
        x_fold_val = x_train_array[val_index]
        y_fold_train = y_train_array[train_index]
        y_fold_val = y_train_array[val_index]

        fold_model = create_model(x_fold_train.shape[1])
        fold_history = fold_model.fit(
            x_fold_train,
            y_fold_train,
            validation_data=(x_fold_val, y_fold_val),
            epochs=30,
            batch_size=128,
            class_weight=class_weight_dict,
            verbose=1,
        )

        best_val_accuracy = float(max(fold_history.history["val_accuracy"]))
        fold_metrics_val.append(best_val_accuracy)
        fold_history_rows.append(
            {
                "fold": fold,
                "best_val_accuracy": best_val_accuracy,
            }
        )
        print(f"Melhor val_accuracy: {best_val_accuracy:.4f}")

    cv_summary = {
        "fold_best_val_accuracy": fold_metrics_val,
        "mean_val_accuracy": float(np.mean(fold_metrics_val)),
        "std_val_accuracy": float(np.std(fold_metrics_val)),
    }

    print(f"\n{'=' * 30}")
    print(f"Val Accuracy media: {cv_summary['mean_val_accuracy']:.4f} +- {cv_summary['std_val_accuracy']:.4f}")
    print(f"{'=' * 30}")

    # Depois da validacao cruzada, o modelo final e treinado sobre o conjunto de treino completo.
    model = create_model(prepared["x_train_scaled"].shape[1])
    start_time = time.time()
    history = model.fit(
        x_train_array,
        y_train_array,
        validation_split=0.1,
        epochs=30,
        batch_size=128,
        class_weight=class_weight_dict,
        verbose=1,
    )
    elapsed_time = time.time() - start_time

    probabilities = model.predict(prepared["x_test_scaled"].to_numpy(), verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(prepared["y_test"], predictions)),
        "precision": float(precision_score(prepared["y_test"], predictions, zero_division=0)),
        "recall": float(recall_score(prepared["y_test"], predictions, zero_division=0)),
        "f1_score": float(f1_score(prepared["y_test"], predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(prepared["y_test"], predictions).tolist(),
        "classification_report": classification_report(
            prepared["y_test"],
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    history_frame = pd.DataFrame(history.history)
    cv_history_frame = pd.DataFrame(fold_history_rows)

    # Metadata concentra tudo que a interface precisa para exibir resultados e preparar novas entradas.
    metadata = {
        "target_column": TARGET_COLUMN,
        "dataset_shape": list(raw_dataframe.shape),
        "prepared_shape": list(prepared["dataframe"].shape),
        "train_shape": list(prepared["x_train"].shape),
        "test_shape": list(prepared["x_test"].shape),
        "input_dim": int(prepared["x_train_scaled"].shape[1]),
        "raw_feature_columns": [column for column in prepared["dataframe"].columns if column != TARGET_COLUMN],
        "numeric_columns": prepared["numeric_columns"],
        "categorical_columns": prepared["categorical_columns"],
        "dummy_columns": prepared["dummy_columns"],
        "final_columns": prepared["final_columns"],
        "category_options": {
            column: sorted(raw_dataframe[column].dropna().astype(str).unique().tolist())
            for column in prepared["categorical_columns"]
            if column in raw_dataframe.columns
        },
        "numeric_defaults": {
            column: float(prepared["dataframe"][column].median())
            for column in prepared["numeric_columns"]
        },
        "feature_summary": [
            {"feature": column, "type": "numerica", "treatment": "mediana + StandardScaler"}
            for column in prepared["numeric_columns"]
        ]
        + [
            {
                "feature": column,
                "type": "categorica",
                "treatment": "moda + LabelEncoder/get_dummies",
            }
            for column in prepared["categorical_columns"]
        ],
        "rain_today_mapping": {"No": 0, "Yes": 1},
        "class_weight_dict": {str(key): float(value) for key, value in class_weight_dict.items()},
        "cross_validation": cv_summary,
        "best_epoch": int(np.argmax(history_frame["val_accuracy"]) + 1),
        "epochs_ran": int(len(history_frame)),
        "training_time_seconds": float(elapsed_time),
        "parameters": {
            "epochs": 30,
            "batch_size": 128,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "validation_split": 0.1,
        },
        "model_architecture": {
            "hidden_layers": [128, 64, 32],
            "dropout": [0.3, 0.2, 0.2],
            "output_activation": "sigmoid",
        },
    }

    model_path = MODELS_DIR / "rain_mlp.keras"
    scaler_path = MODELS_DIR / "scaler.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    history_path = ARTIFACTS_DIR / "history.csv"
    cv_history_path = ARTIFACTS_DIR / "cross_validation.csv"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    # Os artefatos salvos aqui serao reutilizados pela interface sem precisar treinar novamente.
    model.save(model_path)
    joblib.dump(prepared["scaler"], scaler_path)
    history_frame.to_csv(history_path, index=False)
    cv_history_frame.to_csv(cv_history_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "scaler": prepared["scaler"],
        "metrics": metrics,
        "history": history_frame,
        "metadata": metadata,
        "paths": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metrics": str(metrics_path),
            "history": str(history_path),
            "cross_validation": str(cv_history_path),
            "metadata": str(metadata_path),
        },
    }


def load_saved_artifacts() -> dict[str, Any]:
    # Recarrega o ultimo conjunto de artefatos gerado via CLI.
    model_path = MODELS_DIR / "rain_mlp.keras"
    scaler_path = MODELS_DIR / "scaler.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    history_path = ARTIFACTS_DIR / "history.csv"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    if not all(path.exists() for path in [model_path, scaler_path, metrics_path, history_path, metadata_path]):
        return {}

    return {
        "model": keras.models.load_model(model_path),
        "scaler": joblib.load(scaler_path),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "history": pd.read_csv(history_path),
    }


def predict_rain_tomorrow(input_frame: pd.DataFrame) -> dict[str, Any]:
    # Usa o mesmo fluxo de transformacao do treino antes de prever.
    artifacts = load_saved_artifacts()
    if not artifacts:
        raise FileNotFoundError("Modelo treinado nao encontrado.")

    prepared_input = prepare_prediction_input(input_frame, artifacts["metadata"])
    scaled_input = artifacts["scaler"].transform(prepared_input)
    scaled_input = np.asarray(scaled_input).astype("float32")

    probability = float(artifacts["model"].predict(scaled_input, verbose=0).ravel()[0])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "label": "Yes" if prediction == 1 else "No",
        "probability_rain": probability,
        "probability_no_rain": 1 - probability,
    }
