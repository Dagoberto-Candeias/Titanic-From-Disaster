# Titanic ML Pipeline - ELT579 UFV

## 📋 Descrição
Este projeto implementa um pipeline completo e avançado de machine learning para o problema clássico do Titanic, utilizando dados do Kaggle. O pipeline inclui:

- **Pré-processamento avançado**: Imputação KNN, encoding categórico, normalização
- **Engenharia de features**: Target encoding, features polinomiais, interações
- **Balanceamento de classes**: SMOTE + undersampling
- **Treinamento de múltiplos modelos**: RandomForest, XGBoost, LightGBM, CatBoost, MLP, etc.
- **Otimização de hiperparâmetros**: Optuna com 150 trials
- **Avaliação robusta**: Cross-validation com 15 folds
- **Geração de relatórios**: Markdown, DOCX, PDF com gráficos e tabelas
- **Testes abrangentes**: Unitários, integração e smoke tests

## 🏗️ Arquitetura do Projeto

### Estrutura de Diretórios
```
titanic-from-disaster/
├── train.py                                            # Script principal (orquestrador)
├── predict.py                                          # Script de inferência
├── config.py                                           # Configurações globais

├── titanic_pipeline/                                   # Módulos organizados
│   ├── core/
│   │   ├── modeling.py                                 # Treinamento de modelos
│   │   ├── preprocessing.py                            # Pré-processamento
│   │   ├── reporting.py                                # Geração de relatórios
│   │   └── utils.py                                    # Utilitários
│   ├── features/
│   │   ├── engineer.py                                 # Engenharia de features
│   │   └── selectors.py                                # Seleção de features
│   └── utils/
│       ├── cache.py                                    # Sistema de cache
│       ├── parallel.py                                 # Processamento paralelo
│       └── validation.py                               # Validação de dados
├── tests/                                              # Testes
│   ├── test_pipeline.py                                # Testes unitários
│   └── test_reporting.py                               # Testes de relatórios
├── output/                                             # Resultados gerados
│   ├── relatorios/                                     # Relatórios (MD, DOCX, PDF)
│   ├── graficos/                                       # Gráficos e plots
│   │   ├── calibration/                                # Plots de calibração
│   │   ├── correlation/                                # Heatmaps de correlação
│   │   ├── feature_importance/                         # Importância de features
│   │   ├── roc_curves/                                 # Curvas ROC
│   │   └── shap/                                       # Análises SHAP
│   ├── models/                                         # Modelos treinados
│   └── changelog/                                      # Logs e manifestos
└── arquivo/                                            # Documentação adicional
```

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip
- Git

### Instalação
```bash
# Clone o repositório
git clone <repository-url>
cd titanic-from-disaster

# Instale as dependências
pip install -r requirements.txt

# Para desenvolvimento (opcional)
pip install -r requirements-dev.txt
```

### Dependências Principais
- **ML**: scikit-learn, xgboost, lightgbm, catboost
- **Pré-processamento**: pandas, numpy, imbalanced-learn
- **Otimização**: optuna
- **Visualização**: matplotlib, seaborn, plotly
- **Relatórios**: python-docx, reportlab, shap
- **Testes**: pytest, pytest-cov

## 📊 Uso

### Execução Completa do Pipeline
```bash
# Pipeline completo com todos os modelos e relatórios
python train.py
```

### Execução com Configurações Customizadas
```python
from titanic_pipeline.core.pipeline import run_pipeline
from config import CONFIG

# Modificar configurações
CONFIG["fast_mode"] = True
CONFIG["optuna_trials"] = 50

# Executar pipeline
results = run_pipeline()
```

### Validação de Setup
```bash
# Validar importações e dependências
python test_imports.py
```

### Execução Rápida (Fast Mode)
```bash
# Execução rápida com menos modelos e trials
python train.py --fast-mode
```

### Inferência em Novos Dados
```bash
# Fazer predições com modelo treinado
python predict.py --input test.csv --output predictions.csv
```



### Testes
```bash
# Todos os testes
pytest tests/

# Testes específicos
pytest tests/test_pipeline.py -v
pytest tests/test_reporting.py -v

# Cobertura de testes
pytest --cov=titanic_pipeline tests/
```

## ⚙️ Configuração

O arquivo `config.py` contém todas as configurações do sistema:

- **Modelo**: Número de folds CV, random state, paralelização
- **Features**: Configurações de engenharia de features
- **Relatórios**: Tipos de relatório a gerar
- **Cache**: Configurações de cache para performance
- **Testes**: Configurações de testes

## 📈 Resultados Atuais (Última Execução)

- **Melhor Acurácia**: 83.61% (Ensemble Voting)
- **Features**: 25 features após engenharia completa
- **Modelos**: 20 algoritmos treinados e comparados
- **Relatórios**: Documentação completa em 3 formatos (MD, DOCX, PDF)
- **Plots**: 15+ visualizações geradas (ROC, calibração, feature importance, etc.)
- **Ensemble**: Voting e Stacking com calibração automática
- **Tempo de Execução**: ~17 minutos (completo) / ~2 minutos (rápido)

## 🔧 Desenvolvimento

### Adicionando Novos Modelos
```python
# Em titanic_pipeline/core/modeling.py, adicionar novo modelo
from sklearn.ensemble import NewModel

def get_base_models(config):
    models = {
        # ... modelos existentes
        "New Model": NewModel(param1=value1, param2=value2),
    }
    return models

# Ou modificar config.py para configurações customizadas
```

### Atualizações Recentes
- **Correções no Módulo Parallel:** Removida classe `ParallelProcessor` duplicada em `titanic_pipeline/utils/parallel.py`. Melhorada formatação para conformidade com Flake8 (linhas longas quebradas, indentação ajustada). Removidos imports não utilizados. O módulo agora está limpo e sem erros de linting.
- **Testes Validados:** Importações, execução do pipeline e geração de relatórios confirmados sem erros.
- **Performance:** Processamento paralelo otimizado com fallback sequencial em caso de falhas.

### Contribuindo
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Logs e Debugging

- **Arquivo de log**: `titanic_ml.log`
- **Níveis de log**: DEBUG, INFO, WARNING, ERROR
- **Configuração**: Modificar `CONFIG["log_level"]`

## 🐛 Troubleshooting

### Problemas Comuns
1. **Dependências faltando**: `pip install -r requirements.txt`
2. **Dados não encontrados**: Verificar arquivos `train.csv` e `test.csv`
3. **Memória insuficiente**: Reduzir `CONFIG["parallel_jobs"]`
4. **SHAP falhando**: Instalar `pip install shap` ou desabilitar
5. **Import errors**: Executar `python test_imports.py` para validar setup
6. **Arquivo antigo**: O script antigo foi movido para `.bak`, usar `train.py`

### Performance
- Use `CONFIG["fast_mode"] = True` para execuções rápidas
- Configure `CONFIG["cache_enabled"] = True` para reutilizar computações
- Ajuste `CONFIG["parallel_jobs"]` baseado no hardware

## 📚 Referências

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Optuna Documentation](https://optuna.org/)

## 👤 Autor
**Dagoberto Candeias de Moraes (118550)**  
*Curso ELT579 - Universidade Federal de Viçosa*

## 📄 Licença
Este projeto é parte do curso ELT579 da UFV e é destinado para fins educacionais.
