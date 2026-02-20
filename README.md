# Projeto Titanic - Análise e Predição

[![CI Pipeline](https://github.com/dagoberto-moraes/titanic-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/dagoberto-moraes/titanic-ml-pipeline/actions)

Este projeto contém um pipeline completo de Machine Learning para o dataset do Titanic, incluindo análise exploratória, treinamento de modelos, otimização e geração de relatórios automáticos.

## Visão Geral

O objetivo é analisar os fatores que influenciaram a sobrevivência dos passageiros e construir um modelo preditivo. O projeto está estruturado para ser robusto, reprodutível e fácil de manter, utilizando as melhores práticas de engenharia de software em um contexto de ciência de dados.

## Documentação do Projeto

*   **[Guia de Contribuição (`CONTRIBUTING.md`)](CONTRIBUTING.md):** Instruções para novos desenvolvedores sobre como configurar o ambiente, padrões de código e fluxo de trabalho.
*   **Histórico de Mudanças (`CHANGELOG.md`):** Registro de todas as alterações, correções e novas funcionalidades em cada versão.
*   **Código de Conduta (`CODE_OF_CONDUCT.md`):** Diretrizes para garantir um ambiente de colaboração respeitoso e inclusivo.

## Começando (Getting Started)

### Pré-requisitos

*   Anaconda ou Miniconda
*   Git

### Instalação

1.  **Clone o repositório:**
    ```sh
    git clone <URL_DO_REPOSITORIO>
    cd titanic-ml-pipeline
    ```

2.  **Crie e ative o ambiente (Windows):**
    Use o script de automação para criar o ambiente Conda e instalar todas as dependências.
    ```bat
    setup_environment.bat
    ```

3.  **Instale o projeto e os hooks de pré-commit:**
    Após a criação do ambiente, ative-o e finalize a configuração.
    ```bat
    conda activate titanic_ml
    pip install -e .
    pre-commit install
    ```

Para instruções detalhadas ou setup manual, consulte o `CONTRIBUTING.md`.

## Uso (Usage)

### Execução Completa (Recomendado)
Para executar todo o pipeline (limpeza, validação, testes e geração de relatórios), use o script principal:
```bat
run_all.bat
```
Os resultados (relatórios, modelos, gráficos) serão salvos na pasta `output/`.

### Execução de Passos Individuais
*   **Gerar o relatório principal:** `python src/gerar_relatorio_titanic.py`
*   **Rodar os testes:** `run_tests.bat`
*   **Formatar o código:** `format_code.bat`
*   **Atualizar o ambiente:** `update_environment.bat`

## Estrutura de Pastas

*   `src/`: Código fonte principal (relatórios, utilitários, configurações).
*   `scripts/`: Scripts auxiliares (predição, treinamento, limpeza).
*   `tests/`: Testes unitários e de integração.
*   `.github/`: Workflows de Integração Contínua (CI).
*   `data/`: Dados do projeto (raw, processed).
*   `output/`: Artefatos gerados (modelos, gráficos, relatórios).
*   `output/graficos`: Gráficos gerados (PNG).
*   `output/relatorios`: Relatórios finais (PDF, DOCX, MD) e logs.
*   `output/models`: Modelos treinados (.pkl).
*   `titanic_pipeline/`: Pipeline modular avançado.
