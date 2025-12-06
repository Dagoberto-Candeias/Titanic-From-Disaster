# =============================================================================
# TITANIC - SCRIPT DE COMPARAÇÃO DE EXECUÇÕES
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 1.0
# =============================================================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_runs(
    history_dir: str = "output/run_history",
    output_dir: str = "output/reports"
):
    """
    Compara os logs de performance de múltiplos modelos de diferentes execuções.

    Esta função lê todos os arquivos 'model_performance_log.csv' de um diretório
    de histórico, compara a performance do melhor modelo de cada execução e
    gera um gráfico de evolução e um arquivo CSV de resumo.

    Args:
        history_dir (str): Diretório contendo os CSVs de log de performance.
        output_dir (str): Diretório para salvar o relatório de comparação e o gráfico.
    """
    logger.info(f"🔎 Procurando por logs de performance em '{history_dir}'...")

    # Encontra todos os arquivos CSV no diretório de histórico
    log_files = glob.glob(os.path.join(history_dir, "*.csv"))

    if not log_files:
        logger.warning(f"Nenhum arquivo de log encontrado em '{history_dir}'. "
                       "Certifique-se de copiar os arquivos 'model_performance_log.csv' para este diretório.")
        return

    all_runs_data = []
    for file_path in log_files:
        try:
            # Extrai um ID da execução a partir do nome do arquivo
            run_id = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path)
            df['run_id'] = run_id
            all_runs_data.append(df)
        except Exception as e:
            logger.error(f"❌ Falha ao ler ou processar o arquivo {file_path}: {e}")

    if not all_runs_data:
        logger.error("Nenhum dado de log pôde ser carregado. Abortando.")
        return

    # Combina os dados de todas as execuções
    combined_df = pd.concat(all_runs_data, ignore_index=True)

    # Encontra o melhor modelo para cada execução
    best_models_per_run = combined_df.loc[combined_df.groupby('run_id')['mean_accuracy'].idxmax()]
    best_models_per_run = best_models_per_run.sort_values('run_id')

    # Garante que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # 1. Salva o resumo da comparação em um CSV
    summary_path = os.path.join(output_dir, "runs_comparison_summary.csv")
    best_models_per_run.to_csv(summary_path, index=False)
    logger.info(f"✅ Resumo da comparação salvo em: {summary_path}")
    print("\n--- Resumo da Comparação ---")
    print(best_models_per_run[['run_id', 'model_name', 'mean_accuracy', 'mean_auc']].to_string(index=False))
    print("--------------------------\n")

    # 2. Gera o gráfico de evolução da performance
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plot = sns.lineplot(
        data=best_models_per_run,
        x='run_id',
        y='mean_accuracy',
        marker='o',
        markersize=8,
        label='Melhor Acurácia'
    )
    plt.title('Evolução da Performance do Melhor Modelo a Cada Execução', fontsize=16, fontweight='bold')
    plt.xlabel('ID da Execução', fontsize=12)
    plt.ylabel('Acurácia Média (CV)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(bottom=max(0, best_models_per_run['mean_accuracy'].min() - 0.02))
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "performance_evolution.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Gráfico de evolução da performance salvo em: {plot_path}")

if __name__ == "__main__":
    compare_runs()