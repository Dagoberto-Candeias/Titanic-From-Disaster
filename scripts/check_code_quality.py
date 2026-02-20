import subprocess
import sys
import os
from importlib.util import find_spec

def check_code_quality():
    """
    Executa o flake8 para verificar a qualidade do código na pasta src/.
    """
    print("🔍 Executando verificação de qualidade de código com flake8...")
    print("-" * 70)

    # Define o diretório raiz do projeto e o diretório do código-fonte
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, 'src')

    # Verifica se o diretório src existe
    if not os.path.isdir(src_dir):
        print(f"❌ ERRO: O diretório de código-fonte '{src_dir}' não foi encontrado.")
        return False

    # Verifica se deve usar configuração do pyproject.toml ou padrões hardcoded
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    has_pyproject = os.path.exists(pyproject_path)
    has_plugin = find_spec("flake8_pyproject") is not None

    # Comando base para executar o flake8
    command = [
        sys.executable,
        '-m',
        'flake8',
        src_dir,
        '--count',
        '--show-source',
        '--statistics'
    ]

    if has_pyproject and has_plugin:
        print("ℹ️  Plugin 'flake8-pyproject' detectado. Usando configurações do pyproject.toml.")
    else:
        print("ℹ️  Usando configurações padrão (plugin não encontrado ou sem pyproject.toml).")
        # Adiciona argumentos padrão para alinhar com Black se não puder ler do toml
        command.extend(['--max-line-length=88', '--ignore=E203,W503'])

    # Executa o comando
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')

    # Imprime a saída do flake8
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("--- ERROS NA EXECUÇÃO DO FLAKE8 ---")
        print(result.stderr)
        print("-" * 70)

    # Verifica o resultado
    if result.returncode == 0:
        print("🎉 Nenhum problema de qualidade de código encontrado. Excelente!")
        return True
    else:
        print(f"❗ Foram encontrados problemas de linting. Verifique o log acima.")
        return False

if __name__ == "__main__":
    if not check_code_quality():
        sys.exit(1)