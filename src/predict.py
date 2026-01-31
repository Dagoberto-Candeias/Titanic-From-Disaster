import argparse
import os
import sys
import pandas as pd

# Adiciona o diretório raiz ao path para permitir a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from titanic_pipeline.core.modeling import load_and_predict

def main():
    parser = argparse.ArgumentParser(description="Fazer previsões com o modelo treinado do Titanic.")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o arquivo CSV de entrada (ex: data/raw/test.csv)")
    parser.add_argument("--model", type=str, default="output/models/best_model_pipeline.pkl", help="Caminho para o modelo treinado")
    parser.add_argument("--output", type=str, default="output/predictions.csv", help="Caminho para salvar as previsões")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Erro: Arquivo de entrada não encontrado: {args.input}")
        return

    if not os.path.exists(args.model):
        print(f"❌ Erro: Modelo não encontrado: {args.model}. Execute src/train.py primeiro.")
        return

    print(f"=== Iniciando Previsão ===")
    print(f"📂 Dados: {args.input}")
    print(f"🧠 Modelo: {args.model}")
    
    try:
        df = pd.read_csv(args.input)
        predictions = load_and_predict(args.model, df)
        
        # Salvar previsões
        output_df = pd.DataFrame({
            "PassengerId": df["PassengerId"] if "PassengerId" in df.columns else range(len(predictions)),
            "Survived": predictions
        })
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"✅ Sucesso! Previsões salvas em: {args.output}")
        
    except Exception as e:
        print(f"❌ Erro ao gerar previsões: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()