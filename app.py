from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from src.config import DATA_PATH
from src.data import load_dataset, summarize_dataset
from src.train import load_saved_artifacts, predict_rain_tomorrow


st.set_page_config(
    page_title="Previsao de Chuva com MLP",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset(csv_path: str) -> pd.DataFrame:
    return load_dataset(csv_path)


def render_metric_cards(metrics: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acuracia", f"{metrics['accuracy']:.4f}")
    col2.metric("Precisao", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1-score", f"{metrics['f1_score']:.4f}")


def render_history_plot(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.index + 1, history["loss"], label="Treino")
    axes[0].plot(history.index + 1, history["val_loss"], label="Validacao")
    axes[0].set_title("Curva de perda")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.index + 1, history["accuracy"], label="Treino")
    axes[1].plot(history.index + 1, history["val_accuracy"], label="Validacao")
    axes[1].set_title("Curva de acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    st.pyplot(fig)


def render_confusion_matrix(metrics: dict) -> None:
    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao")
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"], rotation=0)
    st.pyplot(fig)


FIELD_LABELS = {
    "Location": "Localidade",
    "MinTemp": "Temperatura minima (C)",
    "MaxTemp": "Temperatura maxima (C)",
    "Rainfall": "Chuva hoje (mm)",
    "Evaporation": "Evaporacao",
    "Sunshine": "Horas de sol",
    "WindGustDir": "Direcao da rajada de vento",
    "WindGustSpeed": "Velocidade da rajada (km/h)",
    "WindDir9am": "Direcao do vento as 9h",
    "WindDir3pm": "Direcao do vento as 15h",
    "WindSpeed9am": "Velocidade do vento as 9h (km/h)",
    "WindSpeed3pm": "Velocidade do vento as 15h (km/h)",
    "Humidity9am": "Umidade as 9h (%)",
    "Humidity3pm": "Umidade as 15h (%)",
    "Pressure9am": "Pressao as 9h",
    "Pressure3pm": "Pressao as 15h",
    "Cloud9am": "Nebulosidade as 9h",
    "Cloud3pm": "Nebulosidade as 15h",
    "Temp9am": "Temperatura as 9h (C)",
    "Temp3pm": "Temperatura as 15h (C)",
    "RainToday": "Choveu hoje?",
}


def format_field_name(column: str) -> str:
    return FIELD_LABELS.get(column, column)


def get_input_schema(metadata: dict) -> tuple[list[str], list[str], dict, dict]:
    # RainToday e tratado como selecao Yes/No na interface, mesmo tendo virado 0/1 no treino.
    numeric_columns = [column for column in metadata["numeric_columns"] if column != "RainToday"]
    categorical_columns = list(metadata["categorical_columns"])
    if "RainToday" not in categorical_columns:
        categorical_columns.append("RainToday")

    category_options = dict(metadata["category_options"])
    category_options.setdefault("RainToday", ["No", "Yes"])

    numeric_defaults = dict(metadata["numeric_defaults"])
    numeric_defaults.pop("RainToday", None)
    return numeric_columns, categorical_columns, category_options, numeric_defaults


def initialize_prediction_state(metadata: dict) -> None:
    numeric_columns, categorical_columns, category_options, numeric_defaults = get_input_schema(metadata)

    # Session state preserva os valores entre reruns do Streamlit, inclusive no botao aleatorio.
    for column in numeric_columns:
        state_key = f"prediction_{column}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(numeric_defaults.get(column, 0.0))

    for column in categorical_columns:
        state_key = f"prediction_{column}"
        if state_key not in st.session_state:
            options = category_options.get(column, [""])
            if column == "RainToday" and "No" in options:
                st.session_state[state_key] = "No"
            else:
                st.session_state[state_key] = options[0] if options else ""


def fill_random_prediction_state(dataframe: pd.DataFrame, metadata: dict) -> None:
    numeric_columns, categorical_columns, _, _ = get_input_schema(metadata)
    # Usa uma linha real da base para gerar um exemplo consistente de teste.
    sample_row = dataframe[metadata["raw_feature_columns"]].sample(n=1).iloc[0]

    for column in numeric_columns:
        value = sample_row[column]
        st.session_state[f"prediction_{column}"] = float(value) if pd.notna(value) else 0.0

    for column in categorical_columns:
        value = sample_row[column]
        if pd.isna(value):
            value = "No" if column == "RainToday" else ""
        st.session_state[f"prediction_{column}"] = str(value)


def build_prediction_input(metadata: dict) -> pd.DataFrame:
    st.subheader("Teste o modelo")
    st.caption("Preencha os dados observados hoje para estimar se vai chover amanha.")

    initialize_prediction_state(metadata)

    numeric_columns, categorical_columns, category_options, numeric_defaults = get_input_schema(metadata)

    values: dict[str, object] = {}

    st.markdown("**Variaveis numericas**")
    numeric_grid = st.columns(3)
    for index, column in enumerate(numeric_columns):
        with numeric_grid[index % 3]:
            default_value = float(numeric_defaults.get(column, 0.0))
            values[column] = st.number_input(
                format_field_name(column),
                value=float(st.session_state.get(f"prediction_{column}", default_value)),
                format="%.2f",
                key=f"prediction_{column}",
            )

    st.markdown("**Variaveis categoricas**")
    categorical_grid = st.columns(3)
    for index, column in enumerate(categorical_columns):
        with categorical_grid[index % 3]:
            options = category_options.get(column, [])
            if not options:
                options = [""]
            current_value = st.session_state.get(f"prediction_{column}")
            if current_value not in options:
                current_value = "No" if column == "RainToday" and "No" in options else options[0]
                st.session_state[f"prediction_{column}"] = current_value
            default_index = options.index(current_value)
            values[column] = st.selectbox(
                format_field_name(column),
                options=options,
                index=default_index,
                key=f"prediction_{column}",
            )

    return pd.DataFrame([values], columns=metadata["raw_feature_columns"])


def main() -> None:
    st.title("Projeto Bimestral - Previsao de Chuva com Rede Neural MLP")
    st.write(
        "Aplicacao em Streamlit para visualizar os resultados do ultimo treinamento "
        "e realizar testes finais com a MLP baseada no notebook de referencia."
    )

    default_path = str(DATA_PATH)
    csv_path = st.sidebar.text_input("Caminho do dataset CSV", value=default_path)
    st.sidebar.info(
        "O treinamento agora e executado pelo terminal com `python run_training.py`. "
        "A interface serve para visualizar os resultados salvos e testar previsoes."
    )

    if not Path(csv_path).exists():
        st.error("O arquivo CSV informado nao foi encontrado.")
        st.stop()

    dataframe = get_dataset(csv_path)
    summary = summarize_dataset(dataframe)

    st.subheader("Visao geral da base")
    info1, info2, info3 = st.columns(3)
    info1.metric("Linhas", summary["rows"])
    info2.metric("Colunas", summary["columns"])
    info3.metric("Alvo", "RainTomorrow")

    preview_col, missing_col = st.columns([1.6, 1])
    with preview_col:
        st.markdown("**Amostra dos dados**")
        st.dataframe(dataframe.head(10), width="stretch")

    with missing_col:
        st.markdown("**Valores nulos por coluna**")
        missing_frame = (
            pd.Series(summary["missing_values"], name="faltantes")
            .reset_index()
            .rename(columns={"index": "coluna"})
        )
        st.dataframe(missing_frame, width="stretch", height=320)

    if "target_distribution" in summary:
        st.markdown("**Distribuicao do alvo**")
        target_frame = (
            pd.Series(summary["target_distribution"], name="quantidade")
            .reset_index()
            .rename(columns={"index": "classe"})
        )
        st.dataframe(target_frame, width="stretch")

    saved_artifacts = load_saved_artifacts()
    if not saved_artifacts:
        st.info(
            "Nenhum modelo treinado foi encontrado. Execute `python run_training.py` no terminal "
            "para gerar os artefatos e depois recarregue esta pagina."
        )
        st.stop()

    metadata = saved_artifacts["metadata"]
    metrics = saved_artifacts["metrics"]
    history = saved_artifacts["history"]

    st.subheader("Configuracao do treinamento salvo")
    split1, split2 = st.columns(2)
    split1.metric("Treino", f"{metadata['train_shape'][0]} amostras")
    split2.metric("Teste", f"{metadata['test_shape'][0]} amostras")
    st.caption(
        f"Epocas: {metadata['parameters']['epochs']} | "
        f"Batch size: {metadata['parameters']['batch_size']} | "
        f"Validation split: {metadata['parameters']['validation_split']} | "
        f"Melhor epoca: {metadata['best_epoch']} | "
        f"Tempo de treino: {metadata['training_time_seconds']:.2f}s"
    )
    st.markdown("**Como a base foi preparada para o treino**")
    st.dataframe(pd.DataFrame(metadata["feature_summary"]), width="stretch", height=320)

    st.subheader("Resultados do modelo")
    render_metric_cards(metrics)

    plot_col, matrix_col = st.columns([1.8, 1])
    with plot_col:
        render_history_plot(history)
    with matrix_col:
        render_confusion_matrix(metrics)

    report_frame = pd.DataFrame(metrics["classification_report"]).transpose()
    st.markdown("**Relatorio de classificacao**")
    st.dataframe(report_frame, width="stretch")

    random_col, _ = st.columns([1, 3])
    with random_col:
        if st.button("Preencher aleatoriamente", width="stretch"):
            fill_random_prediction_state(dataframe, metadata)
            # Rerun faz o formulario redesenhar ja com os novos valores.
            st.rerun()

    input_frame = build_prediction_input(metadata)
    if st.button("Executar previsao", width="stretch"):
        prediction = predict_rain_tomorrow(input_frame)
        st.subheader("Resultado da previsao")
        result1, result2, result3 = st.columns(3)
        result1.metric("Classe prevista", prediction["label"])
        result2.metric("Probabilidade de chuva", f"{prediction['probability_rain']:.2%}")
        result3.metric("Probabilidade de nao chover", f"{prediction['probability_no_rain']:.2%}")


if __name__ == "__main__":
    main()
