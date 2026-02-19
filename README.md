# Projeto Titanic - Análise e Predição

Este projeto contém um pipeline completo de Machine Learning para o dataset do Titanic, incluindo análise exploratória, treinamento de modelos, otimização e geração de relatórios automáticos.

## Ordem de Execução

Para garantir que todos os artefatos sejam gerados corretamente, siga a ordem abaixo:

1.  **Geração do Relatório Executivo (Obrigatório)**
    Gera a análise exploratória, treina um modelo baseline e consolida os resultados em um relatório executivo em PDF/DOCX.
    **Este passo é obrigatório para criar o arquivo que o `ler_relatorio_gerado.py` procura.**
    ```bash
    python src/gerar_relatorio_titanic.py
    ```

2.  **Verificação do Conteúdo (Opcional)**
    Lê o relatório Markdown gerado no terminal.
    ```bash
    python src/ler_relatorio_gerado.py
    ```

3.  **Predição de Novos Passageiros (Opcional)**
    Usa o modelo treinado para prever a sobrevivência de passageiros específicos.
    ```bash
    python scripts/prever_passageiro.py
    ```

Recomenda-se criar e ativar um ambiente Conda com `conda-forge` para garantir compatibilidade binária entre `numpy`, `scipy`, `scikit-learn` e pacotes que dependem de extensões compiladas (ex.: `shap`). Use o arquivo `environment.yml` para reproduzir o ambiente completo:

```bash
conda env create -f environment.yml
conda activate titanic_ml
pip install -r requirements.txt
pytest -q
```

## Estrutura de Pastas

*   `src/`: Código fonte principal (relatórios, utilitários, configurações).
*   `scripts/`: Scripts auxiliares (predição, treinamento, limpeza).
*   `tests/`: Testes unitários e de integração.
*   `notebooks/`: Notebooks Jupyter para exploração.
*   `docs/`: Documentação e relatórios finais (PDF, DOCX).
*   `templates/`: Templates para relatórios.
*   `batch_scripts/`: Scripts de automação (bat, sh).
*   `misc/`: Arquivos diversos (logs, dados de exemplo).
*   `output/graficos`: Gráficos gerados (PNG).
*   `output/relatorios`: Relatórios finais (PDF, DOCX, MD) e logs.
*   `output/models`: Modelos treinados (.pkl).
*   `data/`: Dados do projeto (raw, processed).
*   `titanic_pipeline/`: Pipeline modular avançado.
