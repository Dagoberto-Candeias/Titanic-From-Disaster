import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# 1. Carregar o modelo
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'models', 'baseline_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Erro: Modelo não encontrado em: {MODEL_PATH}")
    exit(1)

# 2. Dados do passageiro (VOCÊ DEVE MUDAR ISSO)
passageiro = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 25,
    'Fare': 7.25,
    'FamilySize': 1,
    'Deck': 'U',
    'Title': 'Mr',
    'Embarked': 'S',
    'NameLength': 20,
    'IsAlone': 1,
    'AgeGroup': 'Adulto'
}

# 3. Pré-processamento
df_passageiro = pd.DataFrame([passageiro])

def preprocess(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title'] = le.fit_transform(df['Title'])
    df['Deck'] = le.fit_transform(df['Deck'])
    df['AgeGroup'] = le.fit_transform(df['AgeGroup'])
    return df

df_processado = preprocess(df_passageiro.copy())

# 4. Selecionar features (manter as mesmas do treinamento)
X = df_processado[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Deck', 'Title', 'Embarked', 'NameLength', 'IsAlone', 'AgeGroup']]

# 5. Predição
prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]

# 6. Exibir resultado
if prediction[0] == 1:
    resultado = "Sobreviveu"
else:
    resultado = "Não sobreviveu"

probabilidade_percentual = probability[0] * 100

print("\n--- Resultado da Predição ---")
print(f"O passageiro tem {probabilidade_percentual:.2f}% de chance de ter {resultado}.")
print("-----------------------------")
print("\nAVISO: Este resultado é baseado em um modelo baseline e pode não ser totalmente preciso.")
print("Para previsões mais robustas, utilize o pipeline avançado.")

# Exemplo de uso (para facilitar a vida do usuário)
print("\n💡 Dica: Para testar outro passageiro, altere os dados no dicionário 'passageiro' e execute este script novamente.")
print("Lembre-se de usar os mesmos valores (ex: 'male', 'female', 'Adulto', etc.) para garantir a compatibilidade.")

"""
O script acima carrega o modelo, define os dados de um passageiro,
pré-processa esses dados e usa o modelo para prever a sobrevivência.

IMPORTANTE:
Você precisa editar o dicionário 'passageiro' com as características
do passageiro que você quer prever.
"""