import pytest
from unittest.mock import MagicMock, mock_open
import pandas as pd
import numpy as np
import sys
import os

# Adiciona o diretório 'src' ao path para importar os módulos do projeto
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC_DIR)

import gerar_relatorio_titanic

@pytest.fixture
def sample_df():
    """Fixture para criar um DataFrame de exemplo para os testes."""
    df = pd.DataFrame({
        'Survived': [0, 1, 0, 1, 0] * 10,
        'Pclass': [1, 2, 3, 1, 2] * 10,
        'Sex': ['male', 'female', 'male', 'female', 'male'] * 10,
        'Age': [22, 38, 26, 35, 35] * 10,
        'SibSp': [1, 1, 0, 1, 0] * 10,
        'Parch': [0, 0, 0, 0, 0] * 10,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05] * 10,
        'Embarked': ['S', 'C', 'Q', 'S', 'C'] * 10,
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 'Allen, Mr. William Henry'] * 10,
        'Cabin': [None, 'C85', None, 'C123', None] * 10,
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'] * 10
    })
    # Adiciona colunas derivadas como no carregar_dados
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].fillna('Mr')  # Fallback
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['NameLength'] = df['Name'].apply(len)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=['Criança', 'Adulto', 'Idoso'])
    return df

def test_treinar_modelo_baseline_pytest_style(mocker, sample_df):
    """Testa o treinamento do modelo baseline usando o estilo pytest com a fixture mocker."""
    # 1. Mock de todas as dependências externas usando mocker
    mocker.patch('gerar_relatorio_titanic.os.makedirs')
    mocker.patch('gerar_relatorio_titanic.plt.savefig')
    mocker.patch('gerar_relatorio_titanic.plt.close')
    mock_pickle_dump = mocker.patch('gerar_relatorio_titanic.pickle.dump')
    mocker.patch('gerar_relatorio_titanic.shutil.copy2')
    mocker.patch('builtins.open', mock_open())

    # 2. Configuração dos mocks de ML
    mock_grid_search = mocker.patch('gerar_relatorio_titanic.GridSearchCV')
    mock_estimator = MagicMock()
    mock_estimator.predict.side_effect = lambda *args, **kwargs: np.array([0, 1, 0, 1, 0, 1])
    mock_estimator.predict_proba.side_effect = lambda *args, **kwargs: np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9]])
    mock_estimator.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05])
    
    from sklearn.tree import DecisionTreeClassifier
    mock_tree = MagicMock(spec=DecisionTreeClassifier)
    mock_tree.tree_ = MagicMock()
    mock_tree.criterion = 'gini'
    mock_estimator.estimators_ = [mock_tree]
    
    mock_grid_search.return_value.best_estimator_ = mock_estimator
    mock_grid_search.return_value.best_params_ = {'n_estimators': 100, 'max_depth': 10}

    mock_perm_imp = mocker.patch('gerar_relatorio_titanic.permutation_importance')
    mock_perm_imp.return_value.importances_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05])

    mocker.patch('gerar_relatorio_titanic.cross_validate', return_value={'test_score': np.array([0.8, 0.85, 0.82, 0.87, 0.83])})
    mocker.patch('gerar_relatorio_titanic.plot_tree')

    # 3. Execução da função a ser testada
    results = gerar_relatorio_titanic.treinar_modelo_baseline(sample_df.head(20).copy())

    # 4. Asserções no estilo pytest
    assert isinstance(results, dict)
    assert 'ml' in results and 'imgs' in results
    assert 'acc' in results['ml'] and 'best_params' in results['ml']
    assert mock_pickle_dump.called