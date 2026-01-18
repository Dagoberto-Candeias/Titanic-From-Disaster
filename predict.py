# =============================================================================
# TITANIC - SCRIPT DE PREDIÇÃO
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 5.0 (Modular)
# =============================================================================

import pandas as pd
import pickle
import logging
import os
import sys
from datetime import datetime

# Adiciona o diretório raiz ao path para permitir importações do pacote
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from titanic_pipeline.utils import optimize_memory_usage, CacheManager
from titanic_pipeline.core.modeling import load_and_predict
from titanic_pipeline.config import ConfigManager
from titanic_pipeline.features import FeatureEngineer

def make_predictions(
    input_data_path: str = "data/raw/test.csv",
    model_pipeline_path: str = "output/models/best_model_pipeline.pkl",
    submission_path: str = "output/submission.csv"
):
    """
    Carrega o pipeline treinado, faz predições nos dados de teste e salva o arquivo de submissão.

    Args:
        input_data_path (str): Caminho para o arquivo de teste (test.csv).
        model_pipeline_path (str): Caminho para o pipeline de modelo salvo (.pkl).
        submission_path (str): Caminho para salvar o arquivo de submissão.
    """
    start_time = datetime.now()
    logger.info("🚀 INICIANDO PROCESSO DE PREDIÇÃO...")

    # 1. Validar caminhos
    if not os.path.exists(input_data_path):
        logger.error(f"❌ Arquivo de entrada não encontrado em: {input_data_path}")
        return
    if not os.path.exists(model_pipeline_path):
        logger.error(f"❌ Pipeline de modelo não encontrado em: {model_pipeline_path}")
        return

    # 2. Carregar dados e modelo
    logger.info(f"📖 Carregando dados de '{input_data_path}' e modelo de '{model_pipeline_path}'...")
    test_data = pd.read_csv(input_data_path)
    passenger_ids = test_data['PassengerId'].copy()

    # 3. Engenharia de Features
    logger.info("⚙️ Aplicando engenharia de features...")
    config_manager = ConfigManager()
    config = config_manager.load_config()["config"]
    cache_manager = CacheManager(config.get("cache_dir", "output/cache"), enabled=config.get("cache_enabled", True))
    
    fe = FeatureEngineer(config, cache_manager)
    X_test, _, feature_cols = fe.engineer_features(test_data, is_training=False)
    test_data_processed = pd.DataFrame(X_test, columns=feature_cols)

    # 4. Fazer predições usando a função encapsulada
    try:
        predictions = load_and_predict(model_pipeline_path, test_data_processed)
    except (FileNotFoundError, Exception) as e:
        logger.error(f"❌ Falha no processo de predição: {e}")
        return

    # 5. Gerar arquivo de submissão
    logger.info(f"📤 Gerando arquivo de submissão em '{submission_path}'...")
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions.astype(int)})
    submission_df.to_csv(submission_path, index=False)

    elapsed = datetime.now() - start_time
    logger.info(f"✅ Processo de predição concluído em {elapsed.total_seconds():.2f} segundos.")

if __name__ == "__main__":
    make_predictions()