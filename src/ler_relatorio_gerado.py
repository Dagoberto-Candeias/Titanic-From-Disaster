import os
import sys

# Determina o diretório onde o script está localizado para montar o caminho absoluto
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
relatorio_path = os.path.join(project_root, "output", "relatorios", "Relatorio_Executivo_Titanic.md")

if os.path.exists(relatorio_path):
    print(f"--- Conteúdo de {relatorio_path} ---\n")
    with open(relatorio_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("\n--- Fim do Relatório ---")
else:
    print(f"❌ O arquivo não foi encontrado em:\n   {relatorio_path}")
    print("\n⚠️  AÇÃO NECESSÁRIA: Execute 'python src/gerar_relatorio_titanic.py' para gerar este arquivo.")