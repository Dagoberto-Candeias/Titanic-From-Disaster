import os
import sys
import pickle
import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Adiciona o diretório raiz ao path para permitir importações do pacote
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from titanic_pipeline.core.modeling import load_and_predict

@pytest.fixture
def mock_pipeline(tmp_path):
    """
    Cria e salva um pipeline de modelo scikit-learn simples em um arquivo temporário.
    Retorna o caminho para o arquivo do pipeline salvo.
    """
    # Criar dados de treino simples
    X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6]})
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # Criar um pipeline simples
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    # Salvar o pipeline em um arquivo temporário
    model_path = tmp_path / "test_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return str(model_path)

@pytest.fixture
def mock_prediction_data() -> pd.DataFrame:
    """Cria um DataFrame de exemplo para predição."""
    return pd.DataFrame({'feature1': [1.5, 4.5]})

def test_load_and_predict_success(mock_pipeline, mock_prediction_data):
    """
    Testa se a função load_and_predict carrega o modelo, faz predições
    e retorna o formato esperado em um cenário de sucesso.
    """
    model_path = mock_pipeline
    new_data = mock_prediction_data

    # Executa a função
    predictions = load_and_predict(model_path, new_data)

    # Verifica os resultados
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(new_data)
    np.testing.assert_array_equal(predictions, np.array([0, 1]))

def test_load_and_predict_file_not_found():
    """Testa se a função levanta FileNotFoundError quando o arquivo do modelo não existe."""
    with pytest.raises(FileNotFoundError):
        load_and_predict("non_existent_model.pkl", pd.DataFrame())