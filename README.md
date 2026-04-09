# Projeto Bimestral - Previsao de Chuva com MLP

Aplicacao em Python com `Streamlit` para visualizar resultados e testar uma rede neural `MLP` treinada com o dataset `weatherAUS`. O preparo da base e o treinamento seguem a mesma logica do arquivo [rain_in_australia.ipynb](C:\Users\luizg\Downloads\rain_in_australia.ipynb), mas organizados em scripts para facilitar execucao e demonstracao.

## Objetivo do projeto

Prever a variavel `RainTomorrow` a partir dos dados meteorologicos da base `weatherAUS.csv`, usando uma rede neural do tipo `Multilayer Perceptron`.

## O que o projeto entrega

- leitura e exibicao da base `weatherAUS.csv`
- resumo inicial com preview e contagem de valores nulos
- treinamento da MLP fora da interface, pelo terminal
- salvamento do modelo e dos artefatos do experimento
- exibicao de metricas, matriz de confusao e curvas de aprendizado na interface
- formulario para teste manual de novas previsoes

## Estrutura do projeto

- `app.py`: interface Streamlit para visualizacao e testes finais
- `run_app.py`: inicializador da interface Streamlit
- `run_training.py`: script CLI para treinamento
- `src/data.py`: carga e preparo da base
- `src/train.py`: treinamento, persistencia e inferencia da MLP
- `data/weatherAUS.csv`: base utilizada no projeto
- `models/`: modelo treinado e scaler salvos
- `artifacts/`: metricas, historico e metadados do treinamento

## Requisitos

- Python 3.10 ou superior
- `pip`

## Instalacao recomendada com ambiente virtual

No CMD, dentro da pasta do projeto:

```cmd
python -m venv .venv
```

Ative o ambiente virtual:

```cmd
.venv\Scripts\Activate.ps1
```

Depois instale as dependencias:

```cmd
python -m pip install -r requirements.txt
```

Se o VS Code continuar mostrando aviso de importacao, selecione o interpretador da pasta `.venv`.

## Como executar o treinamento

O treinamento nao acontece pela interface. Ele deve ser executado pelo terminal:

```cmd
python run_training.py
```

Se quiser informar outro arquivo CSV:

```cmd
python run_training.py --csv caminho\para\weatherAUS.csv
```

Durante a execucao, o script:

- prepara a base
- calcula `class_weight`
- executa validacao cruzada com 3 folds
- treina o modelo final
- salva os arquivos necessarios para a interface

## Como abrir a interface

Depois que o treinamento terminar:

```cmd
python run_app.py
```

Ou, se preferir:

```cmd
streamlit run app.py
```

No VS Code, tambem e possivel abrir [run_app.py](C:\Users\luizg\Downloads\rain-in-australia-prediction\run_app.py) e usar o botao `Run Python File`.

## Fluxo da aplicacao

1. Execute `python run_training.py`.
2. O projeto salva:
- `models/rain_mlp.keras`
- `models/scaler.joblib`
- `artifacts/metrics.json`
- `artifacts/history.csv`
- `artifacts/cross_validation.csv`
- `artifacts/metadata.json`
3. Abra a interface com `python run_app.py`.
4. A interface carrega os artefatos salvos e fica responsavel apenas por:
- mostrar os resultados do experimento
- exibir informacoes da base
- permitir teste manual de previsoes

## Metodologia usada

As etapas abaixo seguem o notebook de referencia:

- leitura do arquivo `weatherAUS.csv`
- remocao da coluna `Date`
- remocao das linhas com `RainTomorrow` nulo
- separacao entre colunas numericas e categoricas
- imputacao dos valores ausentes numericos com `median`
- imputacao dos valores ausentes categoricos com `most_frequent`
- codificacao de `RainToday` e `RainTomorrow` com `LabelEncoder`
- transformacao das demais categoricas com `pd.get_dummies(..., drop_first=True)`
- divisao dos dados com `train_test_split(test_size=0.2, random_state=42)`
- padronizacao com `StandardScaler`
- calculo de `class_weight` com `compute_class_weight(..., class_weight="balanced")`
- validacao cruzada com `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`

## Arquitetura da MLP

- camada de entrada com dimensao igual ao numero de features
- `Dense(128, activation="relu")`
- `Dropout(0.3)`
- `Dense(64, activation="relu")`
- `Dropout(0.2)`
- `Dense(32, activation="relu")`
- `Dropout(0.2)`
- `Dense(1, activation="sigmoid")`

Compilacao e treinamento final:

- `optimizer="adam"`
- `loss="binary_crossentropy"`
- `metrics=["accuracy"]`
- `epochs=30`
- `batch_size=128`
- `validation_split=0.1`
- `class_weight=class_weight_dict`

## Resultado esperado de uso

- primeiro, treinar o modelo pelo terminal
- depois, abrir a interface
- por fim, preencher os campos e testar a previsao final

## Observacao importante

Sempre que houver mudanca no pipeline de treino, rode novamente:

```cmd
python run_training.py
```

Isso garante que os arquivos em `models/` e `artifacts/` fiquem atualizados com a versao atual do codigo.
