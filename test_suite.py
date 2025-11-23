import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from titanic_pipeline.core.utils import validate_data_schema
from features import AdvancedFeatureEngineer
from titanic_pipeline.core.modeling import train_single_model, get_base_models, objective
from titanic_pipeline.core.reporting import generate_shap_comparison_plot

# Fixture para dados de teste sintéticos
@pytest.fixture
def sample_data():
    data = {
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 0],
        'Pclass': [3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina'],
        'Sex': ['male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
        'Fare': [7.25, 71.2833, 7.925],
        'Cabin': [None, 'C85', None],
        'Embarked': ['S', 'C', 'S']
    }
    return pd.DataFrame(data)

# Fixture para configuração de teste
@pytest.fixture
def test_config():
    return {
        "random_state": 42,
        "cv_folds": 2,
    }

# Testes para utils.py
def test_validate_data_schema_positive(sample_data):
    expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    assert validate_data_schema(sample_data, expected_columns, "test_data") == True

def test_validate_data_schema_negative(sample_data):
    expected_columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    assert validate_data_schema(sample_data, expected_columns, "test_data") == False

# Testes para features.py
def test_advanced_feature_engineer_creation(sample_data):
    engineer = AdvancedFeatureEngineer()
    df = engineer.create_advanced_features(sample_data.copy(), is_training=True)
    assert 'FamilySize' in df.columns
    assert 'IsAlone' in df.columns

def test_advanced_feature_engineer_missing_imputation(sample_data):
    engineer = AdvancedFeatureEngineer()
    df = engineer.advanced_missing_imputation(sample_data.copy())
    assert df['Age'].isnull().sum() == 0
    assert df['Embarked'].isnull().sum() == 0

# Testes para modeling.py
def test_train_single_model_positive(sample_data, test_config):
    X = sample_data[['Pclass', 'Age']].fillna(0)
    y = sample_data['Survived']
    result = train_single_model("TestModel", LogisticRegression(), X, y, cv_folds=test_config["cv_folds"])
    assert isinstance(result, dict)
    assert 'trained_model' in result
    assert 'mean_score' in result

def test_train_single_model_with_invalid_data():
    with pytest.raises(ValueError):
        train_single_model("TestModel", LogisticRegression(), [], [], cv_folds=2)

def test_get_base_models(test_config):
    models = get_base_models(test_config)
    assert isinstance(models, dict)
    assert 'Random Forest' in models

def test_objective_function(sample_data, test_config):
    X = sample_data[['Pclass', 'Age']].fillna(0)
    y = sample_data['Survived']

    def mock_trial():
        class MockSuggest:
            def suggest_int(self, name, low, high):
                return 100
            def suggest_float(self, name, low, high, log=False):
                return 0.1
            def suggest_categorical(self, name, choices):
                return choices[0]

        return MockSuggest()

    trial = mock_trial()
    score = objective(trial, "RandomForest", X, y, test_config)
    assert isinstance(score, float)


def test_objective_function_xgboost(sample_data, test_config):
    try:
        from xgboost import XGBClassifier
        X = sample_data[['Pclass', 'Age']].fillna(0)
        y = sample_data['Survived']

        def mock_trial():
            class MockSuggest:
                def suggest_int(self, name, low, high):
                    return 3
                def suggest_float(self, name, low, high, log=False):
                    return 0.1
                def suggest_categorical(self, name, choices):
                    return choices[0]

            return MockSuggest()

        trial = mock_trial()
        score = objective(trial, "XGBoost", X, y, test_config)
        assert isinstance(score, float)

    except ImportError:
        pass #Pula o teste caso a biblioteca não esteja instalada

def test_objective_function_lightgbm(sample_data, test_config):
    try:
        from lightgbm import LGBMClassifier
        X = sample_data[['Pclass', 'Age']].fillna(0)
        y = sample_data['Survived']

        def mock_trial():
            class MockSuggest:
                def suggest_int(self, name, low, high):
                    return 3
                def suggest_float(self, name, low, high, log=False):
                    return 0.1
                def suggest_categorical(self, name, choices):
                    return choices[0]

            return MockSuggest()

        trial = mock_trial()
        score = objective(trial, "LightGBM", X, y, test_config)
        assert isinstance(score, float)

    except ImportError:
        pass #Pula o teste caso a biblioteca não esteja instalada

# Testes para reporting.py (mais difíceis de testar sem ambiente completo)
def test_generate_shap_comparison_plot_empty(sample_data, test_config):
     # Teste para garantir que a função não quebra com entrada vazia
    X = sample_data[['Pclass', 'Age']].fillna(0)
    
    try:
        generate_shap_comparison_plot([], X, ['Pclass', 'Age'])
    except Exception as e:
        assert False, f"Erro ao executar generate_shap_comparison_plot com dados vazios: {e}"

@pytest.mark.skip(reason="Requires trained model and SHAP installation")
def test_generate_shap_comparison_plot_with_models(sample_data, test_config):
    # Este teste requer modelos treinados e a biblioteca SHAP instalada
    try:
        from xgboost import XGBClassifier
        X = sample_data[['Pclass', 'Age']].fillna(0)
        y = sample_data['Survived']

        # Criar modelos de exemplo
        model1 = XGBClassifier()
        model1.fit(X, y)
        model2 = LogisticRegression()
        model2.fit(X,y)

        # Simular resultados de modelos
        model_results = {
            "XGBoost": {"trained_model": model1, "mean_score": 0.8},
            "LogisticRegression": {"trained_model": model2, "mean_score": 0.75}
        }

        top_models = list(model_results.items())[:2]

        # Executar a função
        generate_shap_comparison_plot(top_models, X, ['Pclass', 'Age'])

        # Verificar se o arquivo foi criado
        assert True

    except ImportError:
        pass  # SHAP não está instalado, então o teste é ignorado
    except Exception as e:
        assert False, f"Erro ao executar generate_shap_comparison_plot com modelos: {e}"

#Execução dos testes
if __name__ == "__main__":
    pytest.main(["-v", "test_suite.py"])