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

from titanic_pipeline.utils import optimize_memory_usage
from titanic_pipeline.core.modeling import load_and_predict

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

    logger.info("🧠 Otimizando uso de memória dos dados de teste...")
    test_data = optimize_memory_usage(test_data)

    # 3. Fazer predições usando a função encapsulada
    try:
        predictions = load_and_predict(model_pipeline_path, test_data)
    except (FileNotFoundError, Exception) as e:
        logger.error(f"❌ Falha no processo de predição: {e}")
        return

    # 4. Gerar arquivo de submissão
    logger.info(f"📤 Gerando arquivo de submissão em '{submission_path}'...")
    submission_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions.astype(int)})
    submission_df.to_csv(submission_path, index=False)

    elapsed = datetime.now() - start_time
    logger.info(f"✅ Processo de predição concluído em {elapsed.total_seconds():.2f} segundos.")

if __name__ == "__main__":
    import argparse
    
    # CLI support
    parser = argparse.ArgumentParser(
        description="Titanic ML Pipeline - Prediction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --input test.csv --output predictions.csv
  python predict.py --input data/test.csv --output output/my_predictions.csv
        """
    )
    
    parser.add_argument("--input", type=str, default="test.csv", help="Path to input CSV (default: test.csv)")
    parser.add_argument("--output", type=str, default="output/submission.csv", help="Path to save predictions (default: output/submission.csv)")
    parser.add_argument("--model", type=str, default="output/models/best_model_pipeline.pkl", help="Path to trained pipeline")
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Run prediction
    make_predictions(
        input_data_path=args.input,
        model_pipeline_path=args.model,
        submission_path=args.output
    )