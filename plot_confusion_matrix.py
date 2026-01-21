import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
import sys
import os
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from titanic_pipeline.config import ConfigManager
from titanic_pipeline.features import FeatureEngineer
from titanic_pipeline.utils import CacheManager

def load_model(model_path):
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)

def plot_confusion_matrix(model_path="output/models/random_forest_pipeline.pkl"):
    """
    Gera e plota a matriz de confusão para um modelo salvo usando validação cruzada.
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado em: {model_path}")
        return

    logger.info("1. Carregando dados de treino...")
    df = pd.read_csv("data/raw/train.csv")
    
    # Carregar configurações
    config_manager = ConfigManager()
    config = config_manager.load_config()["config"]
    cache_manager = CacheManager(config.get("cache_dir", "output/cache"), enabled=False)

    logger.info("2. Aplicando engenharia de features...")
    fe = FeatureEngineer(config, cache_manager)
    X_np, y, feature_cols = fe.engineer_features(df, is_training=True)
    
    # Converter para DataFrame para compatibilidade com o pipeline salvo
    X = pd.DataFrame(X_np, columns=feature_cols)

    logger.info(f"3. Carregando pipeline: {model_path}")
    try:
        pipeline = load_model(model_path)
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return

    logger.info("4. Gerando previsões via Cross-Validation (5 folds)...")
    # Usamos cross_val_predict para ter previsões "limpas" para todo o dataset de treino
    y_pred = cross_val_predict(pipeline, X, y, cv=5)

    # Gerar Matriz
    cm = confusion_matrix(y, y_pred)
    
    # Métricas rápidas
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n--- Análise de Erros ({os.path.basename(model_path)}) ---")
    print(f"Falsos Positivos (Previu Sobrevivente, mas Morreu): {fp}")
    print(f"Falsos Negativos (Previu Morto, mas Sobreviveu): {fn}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "output/models/random_forest_pipeline.pkl"
    plot_confusion_matrix(model_file)