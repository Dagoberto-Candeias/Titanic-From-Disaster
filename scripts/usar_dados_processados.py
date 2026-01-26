import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def usar_dados_processados():
    """
    Exemplo de como carregar os dados processados, treinar um modelo
    e fazer uma previsão.
    """
    caminho_dados = 'output/processed_data/processed_titanic_data.csv'

    # 1. Verificar se o arquivo de dados processados existe
    if not os.path.exists(caminho_dados):
        print(f"❌ ERRO: Arquivo '{caminho_dados}' não encontrado.")
        print("   Por favor, execute 'gerar_relatorio_titanic.py' primeiro para criar este arquivo.")
        return

    print(f"✅ Carregando dados de '{caminho_dados}'...")
    df = pd.read_csv(caminho_dados)

    # 2. Separar as features (X) do alvo (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    print("\nDados carregados com sucesso:")
    print(f"   - Features (X): {X.shape[1]} colunas")
    print(f"   - Alvo (y): {y.shape[0]} amostras")

    # 3. Treinar um modelo com os dados já processados
    print("\n🤖 Treinando um modelo RandomForestClassifier...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    print("   Modelo treinado com sucesso!")

    # 4. Fazer uma previsão para um passageiro de exemplo
    # Exemplo: Mulher (Sex=0), 1ª Classe (Pclass=1), 30 anos (Age=30), pagou 100 (Fare=100)
    passageiro_exemplo = [[1, 0, 30, 100]]
    previsao = modelo.predict(passageiro_exemplo)
    resultado = "Sobreviveria" if previsao[0] == 1 else "Não Sobreviveria"
    print(f"\n🔮 Previsão para o passageiro de exemplo: {resultado}")

if __name__ == "__main__":
    usar_dados_processados()