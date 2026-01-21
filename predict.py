# =============================================================================
# TITANIC - SCRIPT DE PREDIÇÃO
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 5.0 (Modular)
# =============================================================================

import argparse
import pandas as pd
import os
import logging
import sys

# Adiciona o diretório raiz ao path para permitir importações do pacote
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from titanic_pipeline.config import ConfigManager
from titanic_pipeline.features import FeatureEngineer
from titanic_pipeline.utils import CacheManager
from titanic_pipeline.core.preprocessing import preprocess_data as core_preprocess_data
from titanic_pipeline.core.modeling import load_and_predict

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def predict(input_path, model_path, output_path):
    """
    Gera previsões usando o modelo treinado.
    """
    # 1. Validações iniciais
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado em: {model_path}")
        logger.error("   Execute 'python train.py' primeiro para treinar e salvar o modelo.")
        return

    if not os.path.exists(input_path):
        logger.error(f"❌ Arquivo de entrada não encontrado: {input_path}")
        return

    try:
        # 2. Carregar dados
        logger.info(f"📖 Carregando dados de: {input_path}")
        df = pd.read_csv(input_path)
        passenger_ids = df["PassengerId"].copy()

        # 3. Inicializar Engenharia de Features
        logger.info("⚙️  Inicializando engenharia de features...")
        config_manager = ConfigManager()
        config = config_manager.load_config()["config"]

        # Desabilitar cache para predição para garantir processamento fresco ou usar cache se preferir
        cache_manager = CacheManager(
            cache_dir=config.get("cache_dir", "output/cache"),
            enabled=config.get("cache_enabled", True)
        )

        fe = FeatureEngineer(config, cache_manager)

        # 4. Aplicar transformações
        logger.info("   Aplicando transformações nos dados...")
        # is_training=False é crucial para não tentar usar a coluna 'Survived'
        X_test, _, feature_cols = fe.engineer_features(df, is_training=False)

        # Converter array numpy de volta para DataFrame com nomes de colunas
        # Isso é necessário porque o pipeline do scikit-learn valida os nomes das features
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)

        # 5. Carregar modelo e prever
        logger.info(f"🔮 Carregando modelo de: {model_path}")
        predictions = load_and_predict(model_path, X_test_df)

        # 6. Salvar resultados
        output_df = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions.astype(int)
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"✅ Previsões salvas com sucesso em: {output_path}")

    except ValueError as e:
        if "Feature names" in str(e):
            logger.error("❌ Erro de compatibilidade de features.")
            logger.error("   O modelo salvo espera features diferentes das geradas pelo código atual.")
            logger.error("   SOLUÇÃO: Execute 'python train.py' para re-treinar o modelo com o código atual.")
        else:
            logger.error(f"❌ Erro de valor durante a predição: {e}")
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar previsões com o modelo Titanic")
    parser.add_argument("--input", default="data/raw/test.csv", help="Caminho para o CSV de entrada")
    parser.add_argument("--model", default="output/models/best_model_pipeline.pkl", help="Caminho para o modelo salvo (.pkl)")
    parser.add_argument("--output", default="output/submission.csv", help="Caminho para salvar o CSV de saída")

    args = parser.parse_args()

    predict(args.input, args.model, args.output)
