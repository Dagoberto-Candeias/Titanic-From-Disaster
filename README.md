# Gerador de Relatórios de Análise do Titanic

Este projeto contém um script automatizado para análise exploratória de dados e modelagem preditiva sobre o desastre do Titanic. Ele gera relatórios detalhados em múltiplos formatos (Markdown, Word e PDF) contendo gráficos, estatísticas e interpretações.

## 📋 Funcionalidades

- **Análise de Dados**: Processamento e limpeza do dataset do Titanic.
- **Visualização**: Geração automática de gráficos (sobrevivência por sexo, classe, idade, correlação, etc.).
- **Machine Learning**: Treinamento de um modelo Random Forest para prever sobrevivência e analisar a importância das variáveis.
- **Relatórios**: Geração de arquivos finais em:
  - `output/relatorios/Relatorio_Final_Titanic.md`
  - `output/relatorios/Relatorio_Final_Titanic.docx`
  - `output/relatorios/Relatorio_Final_Titanic.pdf`

## 🛠️ Pré-requisitos

Certifique-se de ter o Python instalado (versão 3.8 ou superior recomendada).

### Bibliotecas Necessárias

As dependências do projeto estão listadas no arquivo `requirements.txt`. As principais bibliotecas utilizadas são:

- `pandas`: Manipulação de dados.
- `seaborn` & `matplotlib`: Visualização de dados.
- `scikit-learn`: Machine Learning.
- `python-docx`: Geração de documentos Word.
- `fpdf`: Geração de documentos PDF.
- `tqdm`: Barra de progresso.

## 🚀 Instalação e Execução

1. **Instale as dependências:**
```bash
pip install -r requirements.txt
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

Histórico recente: Todas as melhorias consolidadas na branch principal (main).

### Troubleshooting
- **Imports falham**: Rode `python test_imports.py`; instale dependências faltantes
- **Cache inválido**: Defina `CONFIG["cache_enabled"] = False` ou incremente versão
- **Memória alta**: Reduza `CONFIG["parallel_jobs"] = 2`; ative otimização em `utils/memory.py`
- **SHAP erro**: Verifique `SHAP_AVAILABLE`; fallback para feature importance básica
- **Dados ausentes**: Certifique-se de `data/raw/train.csv` e `test.csv`

Performance: Use cache para execuções <1 min; desabilite Optuna para testes rápidos.

## 📚 Documentação Adicional

Para documentação detalhada em português, consulte os arquivos na pasta `arquivo/`:
- `LEIA_ME_PRIMEIRO.txt`: Guia de início rápido completo
- `GUIA_RAPIDO.md`: Guia rápido de uso
- `README_TITANIC_DOCUMENTADO.md`: Documentação técnica detalhada
- `ELT579_118550_Relatorio_Anotado_Detalhado.docx`: Relatório anotado

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
