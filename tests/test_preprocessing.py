import pandas as pd
import numpy as np
import pytest

from titanic_pipeline.preprocessing import (
    create_family_features,
    extract_title,
    extract_deck,
    extract_ticket_prefix,
    create_interactions,
    create_bins,
    create_missing_indicators,
)

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Cria um DataFrame de exemplo para os testes."""
    data = {
        'Name': [
            'Braund, Mr. Owen Harris', 
            'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 
            'Heikkinen, Miss. Laina',
            'Allen, Mr. William Henry',
            'Palsson, Master. Gosta Leonard'
        ],
        'Sex': ['male', 'female', 'female', 'male', 'male'],
        'Age': [22, 38, 26, 35, 2],
        'SibSp': [1, 1, 0, 0, 3],
        'Parch': [0, 0, 0, 0, 1],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '373450', '349909'],
        'Fare': [7.25, 71.2833, 7.925, 8.05, 21.075],
        'Cabin': [np.nan, 'C85', 'E46', np.nan, 'G6'],
        'Embarked': ['S', 'C', 'S', 'S', 'S'],
        'Pclass': [3, 1, 1, 3, 3]
    }
    return pd.DataFrame(data)

def test_create_family_features(sample_dataframe):
    """Testa a criação das features 'FamilySize' e 'IsAlone'."""
    df = create_family_features(sample_dataframe)
    assert 'FamilySize' in df.columns
    assert 'IsAlone' in df.columns
    # Teste de cálculo: SibSp=1, Parch=0 -> FamilySize=2
    assert df.loc[0, 'FamilySize'] == 2
    assert df.loc[0, 'IsAlone'] == 0
    # Teste de cálculo: SibSp=0, Parch=0 -> FamilySize=1
    assert df.loc[2, 'FamilySize'] == 1
    assert df.loc[2, 'IsAlone'] == 1
    # Teste de cálculo: SibSp=3, Parch=1 -> FamilySize=5
    assert df.loc[4, 'FamilySize'] == 5
    assert df.loc[4, 'IsAlone'] == 0

def test_extract_title(sample_dataframe):
    """Testa a extração de títulos dos nomes."""
    df = extract_title(sample_dataframe)
    assert 'Title' in df.columns
    assert df.loc[0, 'Title'] == 'Mr'
    assert df.loc[1, 'Title'] == 'Mrs'
    assert df.loc[2, 'Title'] == 'Miss'
    assert df.loc[4, 'Title'] == 'Master'

def test_extract_deck(sample_dataframe):
    """Testa a extração do deck a partir da cabine."""
    df = extract_deck(sample_dataframe)
    assert 'Deck' in df.columns
    assert df.loc[0, 'Deck'] == 'U'  # NaN deve virar 'U'
    assert df.loc[1, 'Deck'] == 'C'
    assert df.loc[2, 'Deck'] == 'E'

def test_extract_ticket_prefix(sample_dataframe):
    """Testa a extração do prefixo do ticket."""
    df = extract_ticket_prefix(sample_dataframe)
    assert 'TicketPrefix' in df.columns
    assert df.loc[0, 'TicketPrefix'] == 'A/5'
    assert df.loc[1, 'TicketPrefix'] == 'PC'
    assert df.loc[3, 'TicketPrefix'] == 'UNKNOWN' # Sem prefixo

def test_create_interactions(sample_dataframe):
    """Testa a criação de features de interação."""
    df = extract_title(sample_dataframe) # Depende do título
    df = create_interactions(df)
    assert 'AgeClass' in df.columns
    assert 'FarePerPerson' in df.columns
    assert 'Title_Interactions' in df.columns
    assert df.loc[0, 'AgeClass'] == 22 * 3
    assert pytest.approx(df.loc[0, 'FarePerPerson'], 0.01) == 7.25 / 2
    assert df.loc[1, 'Title_Interactions'] == 'Mrs_female'

def test_create_bins(sample_dataframe):
    """Testa a criação de features baseadas em bins."""
    df = create_bins(sample_dataframe)
    assert 'feat_AgeBin' in df.columns
    assert 'feat_FareBin' in df.columns
    assert df.loc[0, 'feat_AgeBin'] == 'Young' # Age 22
    assert df.loc[4, 'feat_AgeBin'] == 'Child' # Age 2
    assert df.loc[0, 'feat_FareBin'] == 'Low' # Fare 7.25
    assert df.loc[1, 'feat_FareBin'] == 'Luxury' # Fare 71.28

def test_create_missing_indicators(sample_dataframe):
    """Testa a criação de indicadores de valores ausentes."""
    df = sample_dataframe.copy()
    # Adicionar mais um NaN para teste
    df.loc[0, 'Age'] = np.nan
    df = create_missing_indicators(df)
    assert 'feat_Age_missing' in df.columns
    assert 'feat_Cabin_missing' in df.columns
    assert df.loc[0, 'feat_Age_missing'] == 1
    assert df.loc[1, 'feat_Age_missing'] == 0
    assert df.loc[0, 'feat_Cabin_missing'] == 1
    assert df.loc[1, 'feat_Cabin_missing'] == 0