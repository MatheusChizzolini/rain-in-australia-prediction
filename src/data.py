from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DATE_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Le o arquivo CSV com a base meteorologica.
    return pd.read_csv(csv_path)


def prepare_training_data(dataframe: pd.DataFrame) -> dict[str, Any]:
    # Replica o fluxo de preparacao usado no notebook de referencia.
    df = dataframe.copy()

    if DATE_COLUMN in df.columns:
        df = df.drop(columns=[DATE_COLUMN])

    # O alvo nao pode ficar nulo, porque ele e a classe que a rede precisa aprender.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = df.select_dtypes(exclude=["object"]).columns.tolist()

    # O SimpleImputer devolve array; aqui voltamos para DataFrame para manter colunas e indice.
    num_imputer = SimpleImputer(strategy="median")
    numeric_frame = pd.DataFrame(
        num_imputer.fit_transform(df[numeric_columns]),
        columns=numeric_columns,
        index=df.index,
    )
    df[numeric_columns] = numeric_frame

    cat_imputer = SimpleImputer(strategy="most_frequent")
    categorical_frame = pd.DataFrame(
        cat_imputer.fit_transform(df[categorical_columns]),
        columns=categorical_columns,
        index=df.index,
    )
    df[categorical_columns] = categorical_frame

    label_encoder = LabelEncoder()
    # RainToday e RainTomorrow viram 0/1 antes do treino, como no notebook.
    if "RainToday" in df.columns:
        df["RainToday"] = pd.Series(
            label_encoder.fit_transform(df["RainToday"]),
            index=df.index,
        )
    df[TARGET_COLUMN] = pd.Series(
        label_encoder.fit_transform(df[TARGET_COLUMN]),
        index=df.index,
    )

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    # As demais categoricas entram em one-hot encoding para a MLP receber apenas valores numericos.
    dummy_columns = [column for column in categorical_columns if column not in ["RainToday", TARGET_COLUMN]]
    x = pd.get_dummies(x, columns=dummy_columns, drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    final_columns = x_train.columns.tolist()
    scaler = StandardScaler()

    # O scaler e ajustado no treino e reaplicado no teste para evitar vazamento de dados.
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=final_columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=final_columns, index=x_test.index)

    raw_feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    raw_numeric_columns = [column for column in raw_feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    raw_categorical_columns = [column for column in raw_feature_columns if column not in raw_numeric_columns]

    return {
        "dataframe": df,
        "x": x,
        "y": y,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "scaler": scaler,
        "numeric_columns": raw_numeric_columns,
        "categorical_columns": raw_categorical_columns,
        "dummy_columns": dummy_columns,
        "final_columns": final_columns,
    }


def prepare_prediction_input(input_frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    # Aplica na entrada manual da interface o mesmo formato usado no treino.
    frame = input_frame.copy()

    if "RainToday" in frame.columns:
        rain_today_mapping = metadata.get("rain_today_mapping", {"No": 0, "Yes": 1})
        frame["RainToday"] = frame["RainToday"].map(rain_today_mapping).fillna(0).astype(int)

    dummy_columns = metadata.get("dummy_columns", [])
    # Reindex garante que a entrada final tenha exatamente as mesmas colunas do treinamento.
    frame = pd.get_dummies(frame, columns=dummy_columns, drop_first=True)
    frame = frame.reindex(columns=metadata["final_columns"], fill_value=0)
    return frame


def summarize_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    # Gera um resumo simples da base para exibir na interface.
    summary = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": dataframe.isna().sum().sort_values(ascending=False).to_dict(),
    }

    if TARGET_COLUMN in dataframe.columns:
        summary["target_distribution"] = (
            dataframe[TARGET_COLUMN].fillna("Missing").value_counts(dropna=False).to_dict()
        )

    return summary
