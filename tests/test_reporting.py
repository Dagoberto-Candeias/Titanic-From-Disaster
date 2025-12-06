import os
import sys
import pandas as pd
import pytest

# Adiciona o diretório raiz ao path para permitir importações do pacote
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from titanic_pipeline.core.reporting import log_model_performance_to_csv

@pytest.fixture
def mock_model_results() -> dict:
    """Cria um dicionário de resultados de modelo para os testes."""
    return {
        "RandomForest": {
            "model_name": "RandomForest",
            "mean_score": 0.85,
            "std_score": 0.02,
            "mean_auc": 0.90,
            "mean_precision": 0.84,
            "mean_recall": 0.86,
            "mean_f1": 0.85,
        },
        "XGBoost": {
            "model_name": "XGBoost",
            "mean_score": 0.82,
            "std_score": 0.03,
            "mean_auc": 0.88,
            "mean_precision": 0.81,
            "mean_recall": 0.83,
            "mean_f1": 0.82,
        },
        "FailedModel": {
            "model_name": "FailedModel",
            "error": "Model training failed"
        }
    }

def test_log_model_performance_to_csv(mock_model_results, tmp_path):
    """Testa se a função log_model_performance_to_csv gera o arquivo CSV corretamente."""
    # Define o caminho do arquivo de saída dentro do diretório temporário
    output_file = tmp_path / "model_performance_log.csv"

    # Executa a função
    log_model_performance_to_csv(mock_model_results, output_path=str(output_file))

    # Verifica se o arquivo foi criado
    assert output_file.exists()

    # Lê o CSV e verifica seu conteúdo
    df = pd.read_csv(output_file)
    assert len(df) == 3  # Verifica o número de modelos (linhas)
    assert "mean_accuracy" in df.columns
    assert df.loc[df['model_name'] == 'RandomForest', 'mean_accuracy'].iloc[0] == 0.85
    assert pd.isna(df.loc[df['model_name'] == 'FailedModel', 'mean_auc'].iloc[0])