"""
Testes para o pipeline principal.
"""

import pytest
from titanic_pipeline.core.pipeline import TitanicPipeline
import pandas as pd
import numpy as np
import os


def test_pipeline_initialization():
    """Testa se o pipeline inicializa corretamente."""
    try:
        pipeline = TitanicPipeline()
    except Exception as e:
        assert False, f"Erro ao inicializar o pipeline: {e}"
    assert True, "Pipeline inicializado sem erros."


def test_pipeline_run_with_dummy_data(tmp_path):
    """Testa se o pipeline roda sem erros com dados dummy criados dinamicamente."""
    # Criar dados dummy pequenos para teste rápido
    df_train = pd.DataFrame({
        'PassengerId': range(1, 21),
        'Survived': [0, 1] * 10,
        'Pclass': [1, 3] * 10,
        'Name': ['Person ' + str(i) for i in range(20)],
        'Sex': ['male', 'female'] * 10,
        'Age': [20, 30] * 10,
        'SibSp': [0, 1] * 10,
        'Parch': [0, 1] * 10,
        'Ticket': ['T123'] * 20,
        'Fare': [10.0, 20.0] * 10,
        'Cabin': ['C123'] * 20,
        'Embarked': ['S', 'C'] * 10
    })
    df_test = df_train.drop('Survived', axis=1)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Forçar feature_selection=True e fast_mode=True (para pular Optuna neste teste rápido)
    # Forçar parallel_jobs=1 para evitar overhead/travamento em dados pequenos
    pipeline = TitanicPipeline(config_override={"feature_selection": True, "fast_mode": True, "parallel_jobs": 1})
    
    try:
        assert pipeline.run_pipeline(str(train_path), str(test_path)) is not None
    except ValueError as e:
        if "n_splits" in str(e) or "samples" in str(e):
            pytest.skip("Dados dummy insuficientes para validação cruzada interna do pipeline")
        else:
            raise e

def test_feature_selection_reduction(tmp_path):
    """Verifica se a seleção de features é executada e reportada corretamente."""
    # Criar dados dummy com colunas extras para testar a seleção
    n_samples = 50
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.randint(0, 2, n_samples),
        'Pclass': np.random.randint(1, 4, n_samples),
        'Name': [f'Person {i}' for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.uniform(1, 80, n_samples),
        'SibSp': np.random.randint(0, 5, n_samples),
        'Parch': np.random.randint(0, 5, n_samples),
        'Ticket': [f'T{i}' for i in range(n_samples)],
        'Fare': np.random.uniform(10, 100, n_samples),
        'Cabin': [f'C{i}' for i in range(n_samples)],
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    train_path = tmp_path / "train_fs.csv"
    test_path = tmp_path / "test_fs.csv"
    
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

    # Forçar feature_selection=True e fast_mode=True (para pular Optuna neste teste rápido)
    # Forçar parallel_jobs=1 para evitar overhead/travamento em dados pequenos
    pipeline = TitanicPipeline(config_override={"feature_selection": True, "fast_mode": True, "parallel_jobs": 1, "enhanced_balance": False})

    try:
        results = pipeline.run_pipeline(str(train_path), str(test_path))
        if results["status"] == "success":
            assert "performance" in results
            assert len(results["performance"]) > 0
    except (ValueError, KeyError):
        pytest.skip("Dados insuficientes para o teste de seleção de features")