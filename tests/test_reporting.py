import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from titanic_pipeline.core.reporting import (
    generate_roc_curves,
    generate_feature_correlation_heatmap,
    generate_shap_comparison_plot,
)


def teardown_module(module):
    # Clean up any generated plots to avoid test side effects
    out_dir = os.path.join("output", "graficos")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)


def _prepare_sample():
    # Minimal sample dataset
    df = pd.DataFrame({
        "Age": [22, 38, 26, 35, 28],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Survived": [0, 1, 1, 1, 0],
    })
    return df


def test_generate_roc_curves_creates_file(tmp_path):
    df = _prepare_sample()
    X = df[["Age", "Fare", "SibSp", "Parch"]].fillna(0)
    y = df["Survived"]

    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, y)

    model_results = {
        "rf": {"trained_model": model, "cv_scores": [0.8], "mean_score": 0.8}
    }

    # Run and assert file created
    generate_roc_curves(model_results, X, y, feature_cols=["Age", "Fare"])
    files = glob.glob(str(tmp_path) + "/*")  # no files expected in tmp_path, but check output

    # check for expected output in output/graficos
    out = glob.glob(os.path.join("output", "graficos", "**", "04_roc_curve.png"), recursive=True)
    assert out, "ROC curve image was not produced"


def test_generate_feature_correlation_heatmap_skips_non_numeric():
    # Prepare a dataframe with only object columns for the provided feature list
    df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "v", "u"]})

    # Should not raise and should log warning internally; file should not be created
    generate_feature_correlation_heatmap(df, ["a", "b"])
    out = glob.glob(os.path.join("output", "graficos", "**", "09_feature_correlation_heatmap.png"), recursive=True)
    assert not out, "Correlation heatmap was generated despite no numeric columns"


def test_generate_feature_correlation_heatmap_does_not_create_dir_when_no_numeric():
    # Ensure the correlation directory is not created when there are no numeric columns
    corr_dir = os.path.join("output", "graficos", "correlation")
    # Remove if exists from previous runs to avoid false positives
    if os.path.exists(corr_dir):
        shutil.rmtree(corr_dir)

    df = pd.DataFrame({"a": ["x", "y", "x"], "b": ["u", "v", "u"]})
    generate_feature_correlation_heatmap(df, ["a", "b"])

    assert not os.path.exists(corr_dir), "Correlation directory was created despite no numeric columns"


def test_generate_shap_comparison_plot_handles_missing_shap():
    # If shap is not installed or raises, function should not raise an exception
    df = _prepare_sample()
    X = df[["Age", "Fare", "SibSp", "Parch"]].fillna(0)
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, df["Survived"])
    top_models = [("rf", {"trained_model": model})]

    # Call should not raise even if shap is not available
    generate_shap_comparison_plot(top_models, X.values, feature_names_out=["Age", "Fare", "SibSp", "Parch"])
    # SHAP output optional; test passes if no exception is raised
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