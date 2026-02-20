# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-02-20

### Added
- **Estrutura de Pacote (`setup.py`):** Formalizada a instalação do projeto como um pacote Python, permitindo a instalação via `pip install -e .`.
- **Validação de Ambiente (`scripts/validate_environment.py`):** Adicionado script para verificar programaticamente se as dependências instaladas correspondem ao `environment.yml`.
- **Qualidade de Código (`scripts/check_code_quality.py`):** Adicionado script para análise de qualidade de código com `flake8`.
- **Configuração Centralizada (`pyproject.toml`):** Criado arquivo para centralizar as configurações de `black`, `flake8` e `pytest`.
- **Automação de Commits (`.pre-commit-config.yaml`):** Implementados hooks de pre-commit para garantir a formatação (black), qualidade (flake8) e passagem dos testes (pytest) antes de cada commit.
- **Scripts de Automação (Windows):**
    - `run_all.bat`: Orquestra a execução completa do projeto (limpeza, validação, linting, testes, pipeline).
    - `format_code.bat`: Formata todo o código do projeto com `black`.
    - `run_tests.bat`: Executa a suíte de testes com cobertura e paralelização.
    - `update_environment.bat`: Sincroniza o ambiente conda com o `environment.yml`.
- **Documentação de Contribuição (`CONTRIBUTING.md`):** Adicionado guia para novos desenvolvedores sobre como configurar o ambiente e seguir os padrões do projeto.

### Changed
- **Integração Contínua (`.github/workflows/ci.yml`):**
    - O pipeline agora instala o projeto com `pip install -e .`, refletindo a estrutura de pacote.
    - Adicionados passos para validar o ambiente, checar a formatação do código (`black --check`) e executar os hooks de `pre-commit` em todos os arquivos.
- **Testes:** A execução de testes (local e pre-commit) foi otimizada para rodar em paralelo usando `pytest-xdist`.
- **Dependências (`environment.yml`, `setup.py`):** Adicionados pacotes de desenvolvimento como `pre-commit`, `flake8-pyproject` e `pytest-xdist` para suportar o novo ferramental de qualidade.
- **Scripts de Automação:** Scripts `.bat` foram aprimorados com verificações de diretório e mensagens de erro mais claras.

## [1.1.0] - 2026-02-19

### Changed
- **CI Workflow (`.github/workflows/ci.yml`):** O workflow de Integração Contínua foi atualizado para utilizar `conda` e o arquivo `environment.yml` para a configuração do ambiente. Isso garante que os testes sejam executados em um ambiente idêntico ao de desenvolvimento local, aumentando a confiabilidade e prevenindo falhas de incompatibilidade de pacotes que só ocorreriam na CI.
- **Script de Setup (`setup_environment.bat`):** A lógica de fallback do script de configuração foi aprimorada. Caso a criação do ambiente a partir do `environment.yml` falhe, o script agora tenta criar um ambiente base com `conda` e o canal `conda-forge` antes de recorrer ao `pip`, alinhando-se com as boas práticas de gerenciamento de dependências do projeto.

### Fixed
- **Script de Limpeza (`scripts/limpar_projeto.py`):** Corrigido um bug no cálculo do caminho do diretório raiz. O script agora localiza e remove corretamente os arquivos e pastas obsoletos na raiz do projeto, garantindo que a limpeza funcione como esperado.

## [1.0.0] - 2026-01-31
### Added
- Versão inicial do projeto com pipeline de Machine Learning completo.
- Análise exploratória de dados (EDA) com geração de gráficos.
- Treinamento de modelo baseline (Random Forest) com otimização de hiperparâmetros.
- Geração automática de relatórios em formatos PDF, DOCX e Markdown.
- Scripts de automação para configuração de ambiente e limpeza do projeto.
- Testes unitários e de integração.
- Documentação detalhada (`README.md`, `SETUP.md`, etc.).