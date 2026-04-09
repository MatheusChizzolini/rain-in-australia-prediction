from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_PATH
from src.train import train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o treinamento da MLP de previsao de chuva pelo terminal."
    )
    parser.add_argument(
        "--csv",
        default=str(DATA_PATH),
        help="Caminho para o arquivo weatherAUS.csv. Usa o dataset padrao do projeto se omitido.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV nao encontrado: {csv_path}")

    print("Iniciando treinamento da MLP com o pipeline alinhado ao notebook...")
    print(f"Dataset: {csv_path}")
    print("Hiperparametros fixos do notebook: epochs=30, batch_size=128, validation_split=0.1")
    print("Tambem sera executada a validacao cruzada com 3 folds e class_weight balanceado.")

    result = train_and_save_model(str(csv_path))

    print("\nTreinamento concluido com sucesso.")
    print("Artefatos gerados:")
    for label, path in result["paths"].items():
        print(f"- {label}: {path}")

    print("\nMetricas de teste:")
    metrics = result["metrics"]
    print(f"- accuracy: {metrics['accuracy']:.4f}")
    print(f"- precision: {metrics['precision']:.4f}")
    print(f"- recall: {metrics['recall']:.4f}")
    print(f"- f1_score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
