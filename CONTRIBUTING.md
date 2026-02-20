# Guia de Contribuição - Titanic ML Project

Obrigado pelo interesse em contribuir para o projeto Titanic ML! Este documento fornece as diretrizes para configurar o ambiente de desenvolvimento, manter a qualidade do código e submeter alterações.

## 1. Pré-requisitos

Antes de começar, certifique-se de ter instalado:

*   [Anaconda](https://www.anaconda.com/) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Gerenciamento de ambientes)
*   [Git](https://git-scm.com/) (Controle de versão)

## 2. Configuração do Ambiente

Siga os passos abaixo para configurar o ambiente localmente:

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd "Titanic From Disaster"
    ```

2.  **Crie o ambiente Conda:**
    Utilizamos o arquivo `environment.yml` para garantir que todas as dependências (incluindo compiladores para bibliotecas científicas) estejam corretas.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Ative o ambiente:**
    ```bash
    conda activate titanic_ml
    ```

4.  **Instale o projeto em modo editável:**
    Isso permite que você edite o código em `src/` e as alterações sejam refletidas imediatamente, além de instalar as ferramentas de desenvolvimento definidas no `setup.py`.
    ```bash
    pip install -e .
    ```

5.  **Instale os Hooks de Pre-commit (Importante):**
    Este passo configura o Git para rodar verificações automáticas antes de cada commit.
    ```bash
    pre-commit install
    ```

## 3. Padrões de Código e Qualidade

Este projeto segue padrões rigorosos para garantir a legibilidade e a estabilidade:

*   **Formatação:** Utilizamos Black com limite de 88 caracteres.
*   **Linting:** Utilizamos Flake8 para encontrar erros de estilo e lógica.
*   **Testes:** Utilizamos Pytest para testes unitários.

As configurações dessas ferramentas estão centralizadas no arquivo `pyproject.toml`.

### Usando o Pre-commit

Se você executou o passo 5 da configuração, o `pre-commit` rodará automaticamente ao fazer um `git commit`. Ele verificará:
*   Espaços em branco desnecessários.
*   Sintaxe de arquivos YAML/JSON.
*   Formatação do código (Black).
*   Erros de Linting (Flake8).
*   Execução dos testes unitários (Pytest).

Se algum passo falhar, o commit será bloqueado. Corrija os erros e tente novamente.

Para rodar os hooks manualmente em todos os arquivos a qualquer momento:
```bash
pre-commit run --all-files
```

## 4. Scripts Auxiliares (Windows)

Para facilitar o desenvolvimento no Windows, fornecemos vários scripts `.bat` na raiz do projeto:

*   `run_all.bat`: **(Recomendado)** Executa limpeza, validação, linting, testes e o pipeline completo em sequência.
*   `format_code.bat`: Formata automaticamente todo o código usando Black.
*   `run_tests.bat`: Executa apenas a suíte de testes com relatórios de cobertura.
*   `update_environment.bat`: Atualiza seu ambiente Conda se o `environment.yml` for modificado.
*   `setup_environment.bat`: Script inicial para criação do ambiente (caso não queira usar os comandos manuais).

## 5. Fluxo de Desenvolvimento

1.  Crie uma nova branch para sua feature ou correção:
    ```bash
    git checkout -b feature/minha-nova-feature
    ```
2.  Faça suas alterações.
3.  Garanta que o código está formatado (`format_code.bat`).
4.  Rode os testes para garantir que nada quebrou (`run_tests.bat`).
5.  Faça o commit das alterações (o pre-commit fará a verificação final).
6.  Envie para o repositório remoto e abra um Pull Request.

## 6. Estrutura de Pastas

O código fonte principal reside em `src/`. Scripts de automação e utilitários estão em `scripts/`. Testes devem ser adicionados em `tests/` seguindo o padrão `test_*.py`.