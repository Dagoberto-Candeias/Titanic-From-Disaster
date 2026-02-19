Título: fix(preprocessing): evitar efeitos colaterais no import e robustez em optimize_memory_usage

Resumo:
- Remove efeitos colaterais de import (criação de pastas no topo de módulos).
- Ajusta `optimize_memory_usage` (em `titanic_pipeline/utils.py`) para detectar corretamente colunas string/object e converter colunas de baixa cardinalidade (mesmo com NaNs e `StringDtype`) para `category`.
- Mantém imports pesados (CatBoost, SHAP, etc.) como opcionais/lazy para evitar falhas em ambientes sem dependências binárias.

Motivação:
- Importações que executam ações (como criar pastas) causam efeitos colaterais quando módulos são importados em testes ou ferramentas de análise. Isso quebra princípios de design de módulos e dificulta testes e reuso.
- Alguns módulos compilados apresentam incompatibilidades ABI com versões do NumPy; recomendação: usar `environment.yml` (conda-forge) e um ambiente fixo (`titanic_ml`) para execução de testes e CI.

Testes realizados:
- Suíte completa executada no ambiente conda `titanic_ml`: 34 passed, 306 warnings.

Instruções para rodar localmente:
1. Criar/atualizar o ambiente conda:
```bash
conda env update -f environment.yml -n titanic_ml
conda activate titanic_ml
python -m pip install -e .
```
2. Rodar testes:
```bash
pytest -q
```

Notas adicionais:
- Warnings (depreciações) foram observados e podem ser abordados em PRs separados.
- Se preferir CI sem conda, recomendo gerar um `constraints.txt` com rodas compatíveis ou usar `pip` + `manylinux`-built wheels; porém, conda-forge foi a solução mais simples e estável para este projeto.

