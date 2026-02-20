import os
import sys
import re
from importlib.metadata import version, PackageNotFoundError
from packaging.specifiers import SpecifierSet
from packaging.version import parse as parse_version

# A biblioteca PyYAML é necessária para ler o arquivo de ambiente.
# Ela geralmente já faz parte das distribuições Conda.
try:
    import yaml
except ImportError:
    print("❌ ERRO: A biblioteca 'PyYAML' é necessária. Instale-a com 'conda install pyyaml' ou 'pip install pyyaml'")
    sys.exit(1)

# --- Constantes ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, 'environment.yml')

def parse_dependency(dep_string: str):
    """
    Analisa uma string de dependência (ex: 'pandas>=2.1,<3') e extrai
    o nome do pacote e os especificadores de versão.
    """
    # Regex para encontrar o nome do pacote (letras, números, -, _)
    match = re.match(r"^[a-zA-Z0-9-_]+", dep_string)
    if not match:
        return None, None

    name = match.group(0)
    specifiers = dep_string[len(name):].strip()

    # Trata o caso de '=' que o SpecifierSet não entende, convertendo para '=='
    if specifiers.startswith('=') and not specifiers.startswith('=='):
        specifiers = '==' + specifiers[1:]

    return name, specifiers

def validate_environment():
    """
    Verifica se o ambiente Python atual corresponde às especificações
    do arquivo environment.yml.
    """
    if not os.path.exists(ENV_FILE):
        print(f"❌ ERRO: Arquivo de ambiente '{ENV_FILE}' não encontrado.")
        return False

    print(f"🔬 Validando ambiente contra '{os.path.basename(ENV_FILE)}'...")
    print("-" * 70)

    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        env_data = yaml.safe_load(f)

    dependencies = env_data.get('dependencies', [])
    pip_dependencies = env_data.get('pip', [])

    all_deps = dependencies + pip_dependencies
    
    compliant_count = 0
    mismatch_count = 0
    missing_count = 0

    for dep_string in all_deps:
        if not isinstance(dep_string, str):
            continue

        name, spec_str = parse_dependency(dep_string)
        if not name or name == 'python':  # Ignora a validação da versão do python
            continue

        try:
            installed_version_str = version(name)
            installed_version = parse_version(installed_version_str)
            
            if not spec_str: # Se não há especificador, qualquer versão instalada é válida
                print(f"✅ {name:<20} (Instalado: {installed_version_str}, Requerido: qualquer)")
                compliant_count += 1
                continue

            spec = SpecifierSet(spec_str)
            if installed_version in spec:
                print(f"✅ {name:<20} (Instalado: {installed_version_str}, Requerido: {spec_str})")
                compliant_count += 1
            else:
                print(f"⚠️ {name:<20} (Instalado: {installed_version_str}, Requerido: {spec_str}) - INCOMPATÍVEL")
                mismatch_count += 1

        except PackageNotFoundError:
            print(f"❌ {name:<20} (Requerido: {spec_str or 'qualquer'}) - NÃO ENCONTRADO")
            missing_count += 1

    print("-" * 70)
    print("Resumo da Validação:")
    print(f"  - {compliant_count} pacotes compatíveis.")
    print(f"  - {mismatch_count} pacotes com versões incompatíveis.")
    print(f"  - {missing_count} pacotes não encontrados.")
    print("-" * 70)

    is_valid = mismatch_count == 0 and missing_count == 0
    print("🎉 Ambiente validado com sucesso!" if is_valid else "❗ O ambiente possui inconsistências.")
    return is_valid

if __name__ == "__main__":
    if not validate_environment():
        sys.exit(1)