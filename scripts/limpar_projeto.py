import os
import shutil

def limpar_projeto():
    """
    Remove arquivos obsoletos, redundantes ou não relacionados ao projeto Titanic atual,
    garantindo uma estrutura limpa após falhas do editor ou iterações antigas.
    """
    # Lista de arquivos para remover (caminhos relativos à raiz do projeto)
    arquivos_para_remover = [
        "Script_semana1(Original Titanic).py",  # Obsoleto, substituído pelo pipeline
        "omega_prothean_arc_engine.py",         # Não relacionado ao projeto
    ]

    # Lista de diretórios para remover
    diretorios_para_remover = [
        "graficos",  # A saída correta agora é 'output/graficos'
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("🧹 Iniciando limpeza do projeto...")

    for arquivo in arquivos_para_remover:
        caminho = os.path.join(base_dir, arquivo)
        if os.path.exists(caminho):
            try:
                os.remove(caminho)
                print(f"✅ Arquivo removido: {arquivo}")
            except Exception as e:
                print(f"❌ Erro ao remover {arquivo}: {e}")

    for diretorio in diretorios_para_remover:
        caminho = os.path.join(base_dir, diretorio)
        if os.path.exists(caminho):
            try:
                shutil.rmtree(caminho)
                print(f"✅ Diretório removido: {diretorio}/ (Conteúdo antigo)")
            except Exception as e:
                print(f"❌ Erro ao remover diretório {diretorio}: {e}")

    print("\n✨ Limpeza concluída! O projeto está organizado.")
    print("   - Relatórios e Gráficos atuais estão em: output/")
    print("   - Código principal: gerar_relatorio_titanic.py e train.py")

if __name__ == "__main__":
    limpar_projeto()