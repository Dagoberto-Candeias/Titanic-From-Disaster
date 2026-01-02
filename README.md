# Titanic ML Pipeline - ELT579 UFV

## 📋 Descrição
Este projeto implementa um pipeline completo e avançado de machine learning para o problema clássico do Titanic, utilizando dados do Kaggle. O pipeline inclui:

- **Pré-processamento avançado**: Imputação KNN/Iterative, encoding categórico, normalização
- **Engenharia de features**: Target encoding, features polinomiais, interações, bins e missing indicators
- **Balanceamento de classes**: SMOTE + undersampling (opcional)
- **Treinamento de múltiplos modelos**: RandomForest, XGBoost, LightGBM, SVM, Logistic Regression, etc. (18+ modelos)
- **Otimização de hiperparâmetros**: Optuna (configurável, desabilitado por padrão para performance)
- **Avaliação robusta**: Cross-validation estratificada (5 folds) + RepeatedStratifiedKFold (5x3=15 folds para validação estendida)
- **Explicabilidade**: SHAP plots (bar, beeswarm), curvas de calibração
- **Geração de relatórios**: Métricas JSON, logs estruturados, visualizações (ROC, confusion matrix, correlation heatmap)
- **Testes abrangentes**: Smoke tests, unit tests (pytest), validação de schema

O pipeline foi corrigido para remover erros críticos (typos, imports circulares, chamadas incorretas), implementadas melhorias (cache versionado, ensemble robusto, tratamento granular de erros, validação de features) e estendido com validação robusta via Repeated CV, SHAP e calibração. Todos os commits foram mesclados via PR no GitHub.

Status atual: Funcional, testado e pronto para submissão Kaggle (acurácia ~82-85%).

## 🏗️ Arquitetura do Projeto

### Estrutura de Diretórios
```
titanic-from-disaster/
├── train.py                                      # Script principal de treinamento
├── predict.py                                    # Previsão em dados novos
├── config.py                                     # Configurações globais

├── titanic_pipeline/                             # Módulos organizados
│   ├── core/
│   │   ├── modeling.py                           # Treinamento de modelos e ensembles
│   │   ├── pipeline.py                           # Pipeline principal
│   │   ├── preprocessing.py                      # Pré-processamento e imputação
│   │   ├── reporting.py                          # Geração de relatórios e métricas
│   │   └── utils.py                              # Utilitários (cache, validação)
│   ├── features/
│   │   ├── engineer.py                           # Engenharia de features avançada
│   │   ├── preprocessing.py                      # Pré-processamento de features
│   │   └── selectors.py                          # Seleção de features
│   └── utils/
│       ├── cache.py                              # Sistema de cache versionado
│       ├── helpers.py                            # Helpers gerais
│       ├── memory.py                             # Otimização de memória
│       ├── parallel.py                           # Processamento paralelo
│       └── validation.py                         # Validação de dados e schema
├── tests/                                        # Testes
│   ├── conftest.py                               # Configuração pytest
│   ├── test_modeling.py                          # Testes de modelagem
│   ├── test_optimize_memory_additional.py        # Testes de otimização de memória
│   ├── test_pipeline.py                          # Testes do pipeline
│   ├── test_preprocessing.py                     # Testes de pré-processamento
│   ├── test_reporting.py                         # Testes de relatórios
│   └── test_utils.py                             # Testes de utilitários
├── scripts/                                      # Scripts auxiliares
│   ├── retrain_and_explain.py                    # Retreinamento com validação estendida e SHAP
│   └── generate_shap.py                          # Geração de plots SHAP
├── output/                                       # Resultados gerados
│   ├── models/                                   # Modelos salvos (.pkl)
│   ├── relatorios/                               # Relatórios (metrics.json, timing_report.json)
│   ├── graficos/                                 # Gráficos (ROC, SHAP, calibração, etc.)
│   ├── cache/                                    # Cache versionado (v1.0.0+)
│   └── submission.csv                            # Predições para Kaggle
├── data/raw/                                     # Dados originais
│   ├── train.csv                                 # 891 amostras
│   └── test.csv                                  # 418 amostras
└── arquivo/                                      # Documentação adicional (guias, relatórios anotados)
```

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip
- Git (para versionamento)

### Instalação
```bash
# Clone o repositório
git clone https://github.com/Dagoberto-Candeias/UFV.git  # Ajuste URL se necessário
cd "c:/Projetos/UFV/ELT 579/Scripts e Datasets/Titanic From Disaster"

# Instale as dependências
pip install -r requirements.txt

# Para desenvolvimento e testes (opcional)
pip install pytest pytest-cov
```

### Dependências Principais
- **ML/Core**: scikit-learn, xgboost, lightgbm, optuna
- **Dados/Pré-processamento**: pandas, numpy
- **Visualização/Explicabilidade**: matplotlib, seaborn, shap
- **Utilitários**: joblib (paralelismo), python-docx (relatórios opcionais)
- **Testes**: pytest

Verifique instalação com: `python test_imports.py`

## 📊 Uso

### Execução Completa do Pipeline
```bash
# Treinamento completo (gera models, submission.csv, relatórios)
python train.py

# Tempo estimado: 5-15 min (com cache: <1 min em execuções subsequentes)
# Outputs: output/submission.csv (pronto para Kaggle), metrics.json, plots
```

### Validação Estendida e Explicabilidade
```bash
# Retreinamento com Repeated CV (15 folds), SHAP e calibração
python scripts/retrain_and_explain.py

# Geração dedicada de SHAP plots
python scripts/generate_shap.py
```

### Testes
```bash
# Smoke tests e validação básica (integrado no pipeline)
python train.py  # Executa smoke tests automaticamente

# Testes unitários completos
pytest tests/ -v

# Cobertura
pytest --cov=titanic_pipeline tests/
```

### Configuração Customizada
Edite `config.py` ou passe via variáveis:
- `CONFIG["fast_mode"] = True`: Desabilita Optuna para execução rápida
- `CONFIG["use_optuna"] = True`: Ativa otimização (aumenta tempo)
- `CONFIG["run_smoke_tests"] = True`: Executa testes de integração
- `CONFIG["cache_enabled"] = True`: Usa cache versionado (padrão)

Exemplo via código:
```python
from titanic_pipeline.core.pipeline import run_pipeline
from config import CONFIG

CONFIG["optuna_trials"] = 50  # Reduz trials para teste
results = run_pipeline()  # Retorna dict com métricas
```

## 📈 Resultados e Outputs

- **Acurácia Esperada**: 82-85% (ensemble); validação estendida com Repeated CV para robustez
- **Features**: 25-35 após engenharia (inclui interações, bins, missing indicators)
- **Modelos**: 18+ (base + otimizados); ensemble stacking/voting com pesos normalizados
- **Explicabilidade**: SHAP (importância global/local), curvas de calibração
- **Outputs Principais**:
  - `output/submission.csv`: Predições (418 linhas, formato Kaggle)
  - `output/relatorios/metrics.json`: Métricas estruturadas (acurácia, F1, tempo, best model)
  - `output/graficos/`: Plots (ROC, confusion matrix, SHAP bar/beeswarm, calibração)
  - `titanic_ml.log`: Logs detalhados (nível INFO por padrão)
  - `output/models/best_model_pipeline.pkl`: Pipeline completo salvo

Submissão Kaggle: Faça upload de `output/submission.csv` em https://www.kaggle.com/c/titanic/submit

## 🔧 Desenvolvimento e Contribuição

### Adicionando Novos Modelos/Features
1. Edite `titanic_pipeline/core/modeling.py` > `get_base_models()` para novos modelos
2. Atualize `config.py` com hiperparâmetros
3. Incremente `FEATURE_SCHEMA_VERSION` em `train.py` para invalidar cache
4. Rode `python train.py` e verifique métricas

Exemplo (novo modelo):
```python
# Em get_base_models():
models["NewModel"] = NewModelClassifier(random_state=42)
```

### Contribuindo
1. Fork o projeto
2. Crie branch: `git checkout -b feature/nova-melhoria`
3. Commit: `git commit -m "feat: adiciona nova feature"`
4. Push: `git push origin feature/nova-melhoria`
5. Abra PR (prefixo `blackboxai/` para branches automáticas)

Histórico recente: PR "Extended Validation with Repeated CV, SHAP and Calibration" mesclado (branch `blackboxai/extended-validation`).

### Troubleshooting
- **Imports falham**: Rode `python test_imports.py`; instale dependências faltantes
- **Cache inválido**: Defina `CONFIG["cache_enabled"] = False` ou incremente versão
- **Memória alta**: Reduza `CONFIG["parallel_jobs"] = 2`; ative otimização em `utils/memory.py`
- **SHAP erro**: Verifique `SHAP_AVAILABLE`; fallback para feature importance básica
- **Dados ausentes**: Certifique-se de `data/raw/train.csv` e `test.csv`

Performance: Use cache para execuções <1 min; desabilite Optuna para testes rápidos.

## 📚 Referências
- [Kaggle Titanic](https://www.kaggle.com/c/titanic)
- [Scikit-learn](https://scikit-learn.org/)
- [Optuna](https://optuna.org/)
- [SHAP](https://shap.readthedocs.io/)

## 👤 Autor
**Dagoberto Candeias de Moraes (118550)**  
*ELT579 - Aprendizado de Máquina, UFV*

## 📄 Licença
Projeto educacional (ELT579/UFV); uso livre para fins acadêmicos.
