"""
Testes para o pipeline principal.
"""

import pytest
from titanic_pipeline.core.pipeline import TitanicPipeline
import os


def test_pipeline_initialization():
    """Testa se o pipeline inicializa corretamente."""
    try:
        pipeline = TitanicPipeline()
    except Exception as e:
        assert False, f"Erro ao inicializar o pipeline: {e}"
    assert True, "Pipeline inicializado sem erros."


def test_pipeline_run_with_dummy_data():
    """Testa se o pipeline roda sem erros com dados dummy."""
    # Assume que existem arquivos dummy em data/raw
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    if os.path.exists(train_path) and os.path.exists(test_path):
        pipeline = TitanicPipeline()
        assert pipeline.run_pipeline(train_path, test_path) is not None
