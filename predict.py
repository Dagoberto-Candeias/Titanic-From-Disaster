"""
Script de Predição para o Projeto Titanic

Este script carrega um pipeline de modelo treinado e o utiliza para fazer
predições em um novo conjunto de dados.

Uso:
    python predict.py --model output/models/best_model_pipeline.pkl \
                      --input data/unseen_data.csv \
                      --output output/predictions.csv

Autor: Dagoberto Candeias de Moraes (baseado no projeto principal)
"""

import pandas as pd
import pickle
import argparse
import logging
import sys
from datetime import datetime

# Importa a classe de engenharia de features do script principal para garantir consistência
try:
    from features import AdvancedFeatureEngineer
except ImportError:
    print("ERRO: O arquivo 'features.py' deve estar no mesmo diretório.")
    sys.exit(1)

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def make_predictions(model_path: str, input_path: str, output_path: str):
    """
    Carrega um modelo, faz predições em novos dados e salva os resultados.

    Args:
        model_path (str): Caminho para o arquivo .pkl do pipeline do modelo.
        input_path (str): Caminho para o arquivo CSV com os dados de entrada.
        output_path (str): Caminho para salvar o arquivo CSV com as predições.
    """
    start_time = datetime.now()
    logger.info(f"Iniciando processo de predição em '{input_path}'...")

    # 1. Carregar o pipeline do modelo
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"✅ Pipeline do modelo carregado de '{model_path}'.")
    except FileNotFoundError:
        logger.error(f"❌ ERRO: Arquivo do modelo não encontrado em '{model_path}'.")
        return
    except Exception as e:
        logger.error(f"❌ ERRO: Falha ao carregar o modelo: {e}")
        return

    # 2. Carregar e preparar os dados de entrada
    try:
        unseen_data = pd.read_csv(input_path)
        passenger_ids = unseen_data["PassengerId"]
        logger.info(
            f"✅ Dados de entrada carregados: {unseen_data.shape[0]} registros."
        )

        # Aplica a mesma engenharia de features do treinamento
        logger.info("🛠️  Aplicando engenharia de features...")
        feature_engineer = AdvancedFeatureEngineer()
        # is_training=False para não usar target encoding
        data_to_predict = feature_engineer.create_advanced_features(
            unseen_data, is_training=False
        )
        data_to_predict = feature_engineer.advanced_missing_imputation(data_to_predict)

    except FileNotFoundError:
        logger.error(f"❌ ERRO: Arquivo de entrada não encontrado em '{input_path}'.")
        return
    except Exception as e:
        logger.error(f"❌ ERRO: Falha ao processar os dados de entrada: {e}")
        return

    # 3. Fazer predições
    logger.info("🧠 Realizando predições...")
    predictions = pipeline.predict(data_to_predict)

    # 4. Salvar os resultados
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    submission.to_csv(output_path, index=False)
    logger.info(f"✅ Predições salvas com sucesso em '{output_path}'.")

    elapsed = datetime.now() - start_time
    logger.info(
        f"Processo de predição concluído em {elapsed.total_seconds():.2f} segundos."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script para fazer predições com o modelo Titanic treinado."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Caminho para o arquivo best_model_pipeline.pkl",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para o arquivo CSV de entrada para predição.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Caminho para salvar o arquivo CSV de saída com as predições.",
    )
    args = parser.parse_args()

    make_predictions(
        model_path=args.model, input_path=args.input, output_path=args.output
    )
