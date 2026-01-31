import os
import sys
import logging
import warnings

# Ignorar avisos de depreciação de bibliotecas externas para manter o log limpo
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Adiciona o diretório raiz do projeto ao sys.path para permitir a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurar filtro para suprimir aviso de FeatureSelector (funcionalidade opcional)
class FeatureSelectorFilter(logging.Filter):
    def filter(self, record):
        return "FeatureSelector.select_features not found" not in record.getMessage()

logging.getLogger("titanic_pipeline.core.pipeline").addFilter(FeatureSelectorFilter())

# Configurar filtro para suprimir avisos de Reporting (mismatch de features em modelos complexos)
class ReportingFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(s in msg for s in [
            "Feature mismatch",
            "Shape mismatch",
            "Feature names count",
            "TreeExplainer failed"
        ])

logging.getLogger("titanic_pipeline.core.reporting").addFilter(ReportingFilter())

from titanic_pipeline.core.pipeline import TitanicPipeline

def run_executive_report():
    """Tenta executar a geração do relatório executivo consolidado."""
    print("\n=== Gerando Relatório Executivo Consolidado ===")
    try:
        # Tenta importar assumindo execução da raiz (python src/train.py)
        from src.gerar_relatorio_titanic import gerar_relatorio_completo
        gerar_relatorio_completo()
    except ImportError:
        try:
            # Fallback para execução de dentro da pasta src
            from gerar_relatorio_titanic import gerar_relatorio_completo
            gerar_relatorio_completo()
        except ImportError:
            print("⚠️ Aviso: Não foi possível carregar o módulo de relatório. O PDF/DOCX final não será atualizado.")

def main():
    # Define os caminhos absolutos para os arquivos de dados
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_path = os.path.join(base_path, 'data', 'raw', 'train.csv')
    test_path = os.path.join(base_path, 'data', 'raw', 'test.csv')

    print("=== Iniciando Pipeline de Treinamento ===")
    print(f"Lendo dados de: {base_path}")

    if os.path.exists(train_path) and os.path.exists(test_path):
        try:
            pipeline = TitanicPipeline()
            # Executa o pipeline completo (carregamento, pré-processamento, treino, avaliação)
            pipeline.run_pipeline(train_path, test_path)
            
            # Gera o relatório final consolidando os resultados do pipeline
            run_executive_report()
            
            print("\n=== Sucesso! ===")
            print("Os artefatos (modelos, relatórios, gráficos) foram salvos na pasta 'output/'.")
            print(f"📄 Relatório final disponível em: {os.path.abspath('output/relatorios/Relatorio_Executivo_Titanic.pdf')}")
        except Exception as e:
            print(f"\n❌ Erro durante a execução do pipeline: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Erro: Arquivos não encontrados.\nVerifique se existem:\n - {train_path}\n - {test_path}")

if __name__ == "__main__":
    main()