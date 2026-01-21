import pandas as pd
import os
import sys

def compare_models(target_model="RandomForest"):
    """
    Compara um modelo específico com o melhor modelo encontrado durante o treinamento.
    """
    results_path = "output/relatorios/resultados_modelos.csv"
    
    if not os.path.exists(results_path):
        print(f"❌ Arquivo de resultados não encontrado: {results_path}")
        print("   Execute 'python train.py' primeiro para treinar e salvar o modelo.")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo CSV: {e}")
        return

    # 1. Encontrar o melhor modelo (maior acurácia média)
    if 'mean_accuracy' not in df.columns:
        print("❌ Coluna 'mean_accuracy' não encontrada no CSV.")
        return

    best_model_idx = df['mean_accuracy'].idxmax()
    best_model = df.iloc[best_model_idx]
    best_model_name = best_model['model_name']

    # 2. Encontrar o modelo alvo (RandomForest)
    target_rows = df[df['model_name'] == target_model]
    if target_rows.empty:
        print(f"❌ Modelo '{target_model}' não encontrado na lista de resultados.")
        print(f"   Modelos disponíveis: {', '.join(df['model_name'].tolist())}")
        return
    
    target_row = target_rows.iloc[0]

    # 3. Exibir Comparação
    print(f"\n📊 COMPARAÇÃO: {target_model} vs MELHOR MODELO ({best_model_name})")
    print("=" * 80)
    print(f"{'Métrica':<20} | {target_model:<15} | {best_model_name:<15} | {'Diferença':<10}")
    print("-" * 80)

    metrics = [
        ('Acurácia', 'mean_accuracy'),
        ('Desvio Padrão', 'std_score'),
        ('AUC-ROC', 'mean_auc'),
        ('Precisão', 'mean_precision'),
        ('Recall', 'mean_recall'),
        ('F1-Score', 'mean_f1')
    ]

    for label, col in metrics:
        if col in df.columns:
            val_target = target_row[col]
            val_best = best_model[col]
            
            # Calcular diferença
            try:
                diff = val_target - val_best
                diff_str = f"{diff:+.4f}"
                
                # Formatar valores
                val_target_str = f"{val_target:.4f}"
                val_best_str = f"{val_best:.4f}"
                
                # Destacar melhoria ou piora
                if diff > 0 and col != 'std_score': diff_str = f"🟢 {diff_str}"
                elif diff < 0 and col != 'std_score': diff_str = f"🔴 {diff_str}"
                
                print(f"{label:<20} | {val_target_str:<15} | {val_best_str:<15} | {diff_str:<10}")
            except:
                print(f"{label:<20} | {str(val_target):<15} | {str(val_best):<15} | -")

    print("=" * 80)
    
    if target_model == best_model_name:
        print(f"🎉 O {target_model} já é o melhor modelo do pipeline!")
    else:
        acc_diff = best_model['mean_accuracy'] - target_row['mean_accuracy']
        print(f"ℹ️  O {target_model} está {acc_diff:.4f} pontos ({(acc_diff*100):.2f}%) atrás do líder.")

if __name__ == "__main__":
    # Permite passar o nome do modelo como argumento via linha de comando
    model = sys.argv[1] if len(sys.argv) > 1 else "RandomForest"
    compare_models(model)