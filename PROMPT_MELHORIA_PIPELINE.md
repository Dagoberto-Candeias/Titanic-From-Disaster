# 🎯 PROMPT COMPLETO: MELHORIA E CORREÇÃO DO PIPELINE TITANIC ML

**Data**: Dezembro 2025  
**Projeto**: Titanic - Machine Learning from Disaster  
**Repositório**: https://github.com/Dagoberto-Candeias/UFV  
**Arquivo Principal**: `train.py` (475 linhas)

---

## 📋 CONTEXTO DO PROJETO

### Objetivo
Treinar um pipeline de Machine Learning robusto, otimizado e documentado para prever sobrevivência no Titanic. O projeto deve:
- Rodar sem erros
- Suportar caching inteligente
- Incluir múltiplos modelos e ensembles
- Gerar relatórios e visualizações
- Ser extensível para novos modelos/features

### Estrutura
```
titanic/
├── train.py                 # Script principal (ALVO desta tarefa)
├── predict.py               # Previsão em dados novos
├── config.py                # Configurações
├── requirements.txt         # Dependências
├── tests/                   # Testes automatizados
├── titanic_pipeline/        # Módulos de suporte
│   ├── core/
│   │   ├── modeling.py      # Treinamento de modelos
│   │   ├── utils.py         # Utilitários
│   │   └── reporting.py     # Geração de relatórios
│   ├── preprocessing.py     # Feature engineering
│   └── utils.py             # Helpers
├── data/
│   └── raw/                 # train.csv, test.csv
├── output/
│   ├── models/              # Modelos salvos
│   ├── graficos/            # Visualizações
│   └── relatorios/          # Relatórios
└── .github/workflows/       # CI/CD (GitHub Actions)
```

---

## 🐛 ERROS CRÍTICOS A CORRIGIR (OBRIGATÓRIO)

### Erro 1: Typo na Linha 58
**Localização**: Bloco de imports opcionais, declaração de `LGBM_AVAILABLE`

**Problema**:
```python
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False  # ❌ ERRADO: LGB_AVAILABLE em vez de LGBM_AVAILABLE
    LGBMClassifier = None
```

**Solução**: Renomear `LGB_AVAILABLE` para `LGBM_AVAILABLE` consistentemente
- Procurar todas as referências a `LGB_AVAILABLE` no arquivo
- Substituir por `LGBM_AVAILABLE`
- Verificar se há referências em `titanic_pipeline/core/modeling.py`

---

### Erro 2: Bloco Vazio na Otimização Optuna (~Linha 358)
**Localização**: Seção `if OPTUNA_AVAILABLE and not CONFIG.get("fast_mode", False):`

**Problema**:
```python
for model_name in models_to_optimize:
    if model_name in resultados and resultados[model_name].get("trained_model") is not None:
        logger.info(f"   Otimizando {model_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, model_name, X_train_opt, y_train, CONFIG),
            n_trials=CONFIG.get("optuna_trials", 30),
        )
        # ... (resto da lógica de otimização e salvamento de resultados)  ❌ VAZIO
```

**Solução**: Implementar a lógica após `study.optimize()`:
```python
        best_params = study.best_params
        best_trial = study.best_trial
        best_value = best_trial.value
        
        logger.info(f"   ✅ Best params para {model_name}: {best_params}")
        logger.info(f"   ✅ Best score: {best_value:.4f}")
        
        # Retreinar com best params
        if model_name == "Random Forest":
            optimized_model = RandomForestClassifier(**best_params, random_state=CONFIG["random_state"])
        elif model_name == "XGBoost":
            optimized_model = XGBClassifier(**best_params, random_state=CONFIG["random_state"], use_label_encoder=False, eval_metric='logloss')
        elif model_name == "LightGBM":
            optimized_model = LGBMClassifier(**best_params, random_state=CONFIG["random_state"])
        
        # CV com modelo otimizado
        opt_scores = cross_val_score(optimized_model, X_train_opt, y_train, cv=CONFIG["cv_folds"], scoring="accuracy")
        resultados[f"{model_name}_Optuna"] = {
            "model_name": f"{model_name}_Optuna",
            "mean_score": opt_scores.mean(),
            "std_score": opt_scores.std(),
            "trained_model": optimized_model,
            "best_params": best_params,
            "best_value": best_value,
        }
```

---

### Erro 3: Verificar Imports do Módulo `titanic_pipeline.core.modeling`
**Localização**: Linhas 89-103

**Problema**: O arquivo importa de `titanic_pipeline.core.modeling`, mas não está claro se o módulo existe ou se está completo.

**Solução**:
1. Verificar se `titanic_pipeline/core/modeling.py` existe
2. Confirmar que as funções importadas existem:
   - `train_single_model`
   - `build_stacking_ensemble`
   - `get_base_models`
   - `objective`
   - `load_and_predict`
   - `save_model_pipeline`
3. Se faltarem, adicionar funções stub com logging de erro
4. Se não existir, criar `titanic_pipeline/core/__init__.py` e `titanic_pipeline/core/modeling.py` com implementações básicas

---

## 🚀 MELHORIAS RECOMENDADAS (IMPLEMENTAR NESTA ORDEM)

### Melhoria 1: Cache com Versionamento (Prioridade: ALTA)

**Objetivo**: Evitar usar cache desatualizado quando features mudam

**Implementação**:

1. Adicionar versão de schema no topo do arquivo (`train.py`):
```python
# Logo após imports
FEATURE_SCHEMA_VERSION = "1.0.0"  # Incrementar quando features mudam
```

2. Modificar `get_cache_key()` (localizado em `titanic_pipeline.core.utils`):
```python
def get_cache_key(data_hash, operation, schema_version=FEATURE_SCHEMA_VERSION):
    """Gera chave de cache com versionamento de schema."""
    key_str = f"{data_hash}_{operation}_{schema_version}"
    return hashlib.md5(key_str.encode()).hexdigest()
```

3. Quando chamar cache, passar versão:
```python
cache_key_imputation = get_cache_key(data_hash, "imputation_train", FEATURE_SCHEMA_VERSION)
cache_key_features = get_cache_key(data_hash, "features_train", FEATURE_SCHEMA_VERSION)
cache_key_models = get_cache_key(data_hash, "model_results", FEATURE_SCHEMA_VERSION)
```

4. Adicionar função para invalidar cache manualmente:
```python
def invalidate_cache_version():
    """Incrementar versão para invalidar todos os caches."""
    global FEATURE_SCHEMA_VERSION
    major, minor, patch = FEATURE_SCHEMA_VERSION.split('.')
    FEATURE_SCHEMA_VERSION = f"{major}.{int(minor)+1}.0"
    logger.info(f"Cache versão atualizada para: {FEATURE_SCHEMA_VERSION}")
```

---

### Melhoria 2: Ensemble Robusto com Normalização de Pesos (Prioridade: ALTA)

**Objetivo**: Evitar desbalanço de pesos e garantir ensemble com ≥3 modelos válidos

**Localização**: ~Linha 360, seção "CRIANDO ENSEMBLE"

**Mudanças**:

1. Adicionar função de normalização e validação:
```python
def prepare_ensemble_models(valid_results, min_models=3, weight_bounds=(0.1, 1.0)):
    """
    Prepara modelos e pesos para ensemble com validações.
    
    Args:
        valid_results: dict com resultados dos modelos
        min_models: mínimo de modelos para criar ensemble
        weight_bounds: (min_weight, max_weight) normalizados
    
    Returns:
        tuple: (ensemble_models, normalized_weights) ou (None, None) se inválido
    """
    top_models = sorted(
        valid_results.items(),
        key=lambda x: x[1]["mean_score"],
        reverse=True
    )[:5]
    
    ensemble_models = []
    raw_weights = []
    
    for name, perf in top_models:
        model = perf.get("trained_model")
        if model is not None and hasattr(model, 'predict_proba'):
            ensemble_models.append((name, model))
            raw_weights.append(perf["mean_score"])
    
    if len(ensemble_models) < min_models:
        logger.warning(f"Apenas {len(ensemble_models)} modelos válidos. Mínimo: {min_models}")
        return None, None
    
    # Normalizar pesos entre weight_bounds
    raw_weights = np.array(raw_weights)
    min_raw = raw_weights.min()
    max_raw = raw_weights.max()
    
    if max_raw == min_raw:
        # Todos iguais
        normalized_weights = np.ones_like(raw_weights) * weight_bounds[0]
    else:
        # Normalizar entre bounds
        scaled = (raw_weights - min_raw) / (max_raw - min_raw)
        normalized_weights = weight_bounds[0] + scaled * (weight_bounds[1] - weight_bounds[0])
    
    # Log detalhado
    logger.info("📊 Pesos do Ensemble (normalizados):")
    for (name, _), weight in zip(ensemble_models, normalized_weights):
        logger.info(f"   {name}: {weight:.3f}")
    
    return ensemble_models, normalized_weights.tolist()
```

2. Usar a função no pipeline:
```python
ensemble_models, ensemble_weights = prepare_ensemble_models(valid_results, min_models=3)

if ensemble_models is None:
    logger.error("Não foi possível criar ensemble com ≥3 modelos. Usando melhor modelo único.")
    ensemble = None
else:
    # Criar VotingClassifier com pesos normalizados
    ensemble = VotingClassifier(
        estimators=ensemble_models,
        voting="soft",
        weights=ensemble_weights
    )
    ensemble.fit(X_train_processed, y_train)
    ensemble_scores = cross_val_score(ensemble, X_train_processed, y_train, cv=CONFIG["cv_folds"])
    resultados["Ensemble_Voting"] = {
        "model_name": "Ensemble_Voting",
        "mean_score": ensemble_scores.mean(),
        "std_score": ensemble_scores.std(),
        "trained_model": ensemble,
        "weights": ensemble_weights,
    }
    logger.info(f"✅ Ensemble Voting criado: Acc={ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
```

---

### Melhoria 3: Tratamento de Erros Granular (Prioridade: ALTA)

**Objetivo**: Capturar e logar erros específicos, não genéricos

**Localização**: ~Linha 303, seção de treinamento paralelo

**Mudanças**:

1. Adicionar definição de exceções customizadas no topo:
```python
class ModelTrainingError(Exception):
    """Erro durante treinamento de modelo."""
    pass

class DataValidationError(Exception):
    """Erro durante validação de dados."""
    pass

class EnsembleError(Exception):
    """Erro durante criação de ensemble."""
    pass
```

2. Substituir bloco genérico:
```python
# ❌ ANTES:
except Exception as e:
    logger.error(f"   ❌ {model_name} falhou: {e}")
    resultados[model_name] = {"model_name": model_name, "error": str(e), "mean_score": 0.0}

# ✅ DEPOIS:
except ModelTrainingError as e:
    logger.error(f"   ❌ {model_name} falhou (treinamento): {e}")
    resultados[model_name] = {
        "model_name": model_name,
        "error": f"ModelTrainingError: {str(e)}",
        "mean_score": 0.0,
        "trained_model": None,
    }
except MemoryError:
    logger.error(f"   ❌ {model_name} falhou: Memória insuficiente")
    resultados[model_name] = {
        "model_name": model_name,
        "error": "MemoryError",
        "mean_score": 0.0,
        "trained_model": None,
    }
except Exception as e:
    logger.error(f"   ❌ {model_name} falhou (erro inesperado): {type(e).__name__}: {e}")
    resultados[model_name] = {
        "model_name": model_name,
        "error": f"{type(e).__name__}: {str(e)}",
        "mean_score": 0.0,
        "trained_model": None,
    }
```

3. Adicionar retry logic para modelos que falharam:
```python
def retry_failed_models(resultados, X_train, y_train, cv_folds=5, max_retries=2):
    """Tenta retreinar modelos que falharam."""
    failed_models = {k: v for k, v in resultados.items() if v.get("trained_model") is None}
    
    if not failed_models:
        logger.info("✅ Nenhum modelo para retry")
        return
    
    logger.info(f"🔄 Tentando retry em {len(failed_models)} modelos...")
    
    for model_name, perf in failed_models.items():
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"   Tentativa {attempt}/{max_retries} para {model_name}...")
                model = get_base_models(CONFIG)[model_name]
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                resultados[model_name] = {
                    "model_name": model_name,
                    "mean_score": scores.mean(),
                    "std_score": scores.std(),
                    "trained_model": model,
                }
                logger.info(f"   ✅ {model_name} sucesso no retry! Acc={scores.mean():.4f}")
                break
            except Exception as e:
                logger.warning(f"   ❌ Retry {attempt} falhou: {e}")
                if attempt == max_retries:
                    logger.error(f"   ❌ {model_name} descartado após {max_retries} tentativas")
```

---

### Melhoria 4: Validação e Consistência de Features (Prioridade: ALTA)

**Objetivo**: Garantir que train e test usam exatamente as mesmas features

**Localização**: ~Linha 280, após processamento de test

**Mudanças**:

1. Adicionar função de validação:
```python
def validate_feature_consistency(X_train, X_test, feature_cols, logger):
    """
    Valida que train e test têm mesmas features.
    
    Returns:
        tuple: (features_válidas, n_removidas, n_adicionadas)
    """
    train_features = set(feature_cols)
    test_features = set(X_test.columns if hasattr(X_test, 'columns') else [])
    
    removed = train_features - test_features
    added = test_features - train_features
    
    if removed:
        logger.warning(f"⚠️  Colunas em TRAIN mas não em TEST ({len(removed)}): {removed}")
    
    if added:
        logger.warning(f"⚠️  Colunas em TEST mas não em TRAIN ({len(added)}): {added}")
    
    valid_features = sorted(train_features & test_features)
    
    logger.info(f"✅ Features válidas (intersecção): {len(valid_features)}")
    logger.info(f"   Shape esperado: Train {X_train.shape}, Test {(len(X_test), len(valid_features))}")
    
    return valid_features, len(removed), len(added)
```

2. Usar após preprocessor:
```python
X_train_processed = preprocessor.fit_transform(train[feature_cols])
X_test_processed = preprocessor.transform(test[feature_cols])

# Validação de features
if isinstance(X_train_processed, pd.DataFrame):
    valid_features, n_rem, n_add = validate_feature_consistency(
        X_train_processed, X_test_processed, feature_cols, logger
    )
else:
    logger.info(f"✅ Sparse matrix: shape={X_train_processed.shape}")

# Assertions
assert X_train_processed.shape[1] == X_test_processed.shape[1], \
    f"Mismatch de features: Train {X_train_processed.shape[1]} vs Test {X_test_processed.shape[1]}"
```

3. Adicionar logging de estatísticas:
```python
logger.info("📊 RESUMO DE FEATURES:")
logger.info(f"   Features iniciais: {len(feature_cols)}")
logger.info(f"   Features após filtragem: {X_train_processed.shape[1]}")
logger.info(f"   Amostras treino: {X_train_processed.shape[0]}")
logger.info(f"   Amostras teste: {X_test_processed.shape[0]}")
```

---

### Melhoria 5: Smoke Tests (Testes de Integração) (Prioridade: MÉDIA)

**Objetivo**: Validar pipeline com dados pequenos antes de rodar completo

**Localização**: Adicionar como função separada antes de `main()`

**Implementação**:

```python
def smoke_test():
    """
    Testes rápidos de integração - valida pipeline com dados pequenos.
    
    Returns:
        bool: True se todos os testes passarem
    """
    logger.info("🧪 Iniciando SMOKE TESTS...")
    
    try:
        # 1. Validar diretórios
        os.makedirs("output/graficos", exist_ok=True)
        os.makedirs("output/models", exist_ok=True)
        assert os.path.exists("data/raw/train.csv"), "train.csv não encontrado"
        assert os.path.exists("data/raw/test.csv"), "test.csv não encontrado"
        logger.info("   ✅ Diretórios e arquivos OK")
        
        # 2. Carregar dados (sample pequeño)
        train = pd.read_csv("data/raw/train.csv").head(50)
        test = pd.read_csv("data/raw/test.csv").head(30)
        assert train.shape[0] > 0, "Train vazio"
        assert test.shape[0] > 0, "Test vazio"
        logger.info(f"   ✅ Dados carregados: train={train.shape}, test={test.shape}")
        
        # 3. Validar schema
        validate_data_schema(train, EXPECTED_TRAIN_COLUMNS, "train.csv")
        validate_data_schema(test, EXPECTED_TEST_COLUMNS, "test.csv")
        logger.info("   ✅ Schema válido")
        
        # 4. Feature Engineering
        fe = AdvancedFeatureEngineer()
        train_fe = fe.advanced_missing_imputation(train)
        train_fe = fe.create_advanced_features(train_fe, is_training=True)
        test_fe = fe.advanced_missing_imputation(test)
        test_fe = fe.create_advanced_features(test_fe, is_training=False)
        logger.info(f"   ✅ Features geradas: train={train_fe.shape}, test={test_fe.shape}")
        
        # 5. Treinar modelo simples
        feature_cols = [c for c in train_fe.columns if c not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin"]]
        preprocessor = create_feature_pipeline(train_fe, feature_cols, CONFIG["random_state"])
        X = preprocessor.fit_transform(train_fe[feature_cols])
        y = train_fe["Survived"]
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        scores = cross_val_score(model, X, y, cv=2, scoring="accuracy")
        assert scores.mean() > 0.3, f"Score muito baixo: {scores.mean():.4f}"
        logger.info(f"   ✅ Modelo treinado: Acc={scores.mean():.4f}")
        
        # 6. Fazer predição
        model.fit(X, y)
        X_test = preprocessor.transform(test_fe[feature_cols])
        preds = model.predict(X_test)
        assert len(preds) == len(test_fe), "Mismatch predictions"
        logger.info(f"   ✅ Predições geradas: {len(preds)} amostras")
        
        # 7. Salvar e carregar modelo
        import pickle
        pipeline_dict = {"preprocessor": preprocessor, "model": model}
        with open("output/models/smoke_test_model.pkl", "wb") as f:
            pickle.dump(pipeline_dict, f)
        with open("output/models/smoke_test_model.pkl", "rb") as f:
            loaded = pickle.load(f)
        logger.info("   ✅ Modelo salvo e carregado OK")
        
        logger.info("✅ SMOKE TESTS PASSARAM!")
        return True
        
    except AssertionError as e:
        logger.error(f"❌ Assertion falhou: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro em smoke tests: {e}", exc_info=True)
        return False
```

**Usar no `main()`**:
```python
def main():
    if CONFIG.get("run_smoke_tests", False):
        logger.info("🚀 Rodando smoke tests antes do pipeline principal...")
        if not smoke_test():
            logger.warning("⚠️  Smoke tests falharam, mas continuando...")
    
    # ... resto do main()
```

---

### Melhoria 6: Documentação Completa (Prioridade: MÉDIA)

**Objetivo**: Adicionar docstrings e comentários explicativos

**Localização**: Topo do arquivo e funções principais

**Mudanças**:

1. Adicionar docstring do módulo (após imports):
```python
"""
Titanic ML Pipeline - Treinamento de Modelos Ensemble

Módulo principal para:
- Carregar e validar dados (train.csv, test.csv)
- Feature engineering e imputation avançado
- Treinamento paralelo de múltiplos modelos
- Otimização de hiperparâmetros com Optuna
- Criação de ensembles robusto
- Geração de relatórios e visualizações

Configuração:
    Editar 'config.py' ou passar CONFIG em main()

Uso:
    python train.py

Saída:
    - output/models/best_model_pipeline.pkl      (modelo final)
    - output/submission.csv                       (predições)
    - output/graficos/                            (visualizações)
    - output/relatorios/                          (relatórios)
    - titanic_ml.log                              (logs)

Autor: Dagoberto Candeias de Moraes (118550)
Versão: 5.0 (Modular)
"""
```

2. Adicionar docstrings em funções principais:
```python
def main():
    """
    Pipeline completo de treinamento para Titanic ML.
    
    Fluxo:
    1. Criar diretórios e carregar dados
    2. Validar schema e otimizar memória
    3. Feature engineering com cache
    4. Treinamento paralelo de modelos base
    5. Otimização de hiperparâmetros (opcional, Optuna)
    6. Criar ensembles (Voting + Stacking)
    7. Gerar visualizações e relatórios
    8. Salvar modelo e fazer submissão
    
    Returns:
        bool: True se pipeline completou com sucesso
        
    Raises:
        FileNotFoundError: Se dados não encontrados
        DataValidationError: Se schema inválido
    """
```

3. Adicionar exemplos de extensão:
```python
"""
COMO ADICIONAR NOVO MODELO:

1. Definir no get_base_models():
   ```python
   from sklearn.svm import SVC
   
   models["SVM"] = SVC(kernel="rbf", probability=True)
   ```

2. Se needed hiperparâmetros otimizáveis, adicionar ao objective():
   ```python
   if model_name == "SVM":
       C = trial.suggest_float("C", 0.1, 100)
       kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
       return objective_svm(trial, C, kernel, X, y)
   ```

3. Rerun pipeline:
   python train.py
"""
```

---

### Melhoria 7: Logging Granular e Métricas Estruturadas (Prioridade: MÉDIA)

**Objetivo**: Exportar métricas em JSON e criar relatório final estruturado

**Localização**: Seção de relatórios (~Linha 410)

**Implementação**:

1. Adicionar função de exportação de métricas:
```python
def export_metrics_json(resultados, script_total_time):
    """
    Exporta métricas de treinamento em formato JSON estruturado.
    
    Saída:
        {
            "timestamp": "2025-12-06T14:32:00",
            "total_time_seconds": 125.4,
            "models": {
                "Random Forest": {
                    "accuracy": 0.845,
                    "std": 0.025,
                    "time_seconds": 12.3,
                    "trained": true,
                    "best_params": {...}
                },
                ...
            },
            "ensemble": {
                "name": "Ensemble_Stacking",
                "accuracy": 0.87,
                "weights": [0.3, 0.4, 0.3]
            },
            "best_model": {
                "name": "Ensemble_Stacking",
                "accuracy": 0.87,
                "features_count": 45
            }
        }
    """
    from datetime import datetime
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": script_total_time.total_seconds(),
        "models": {},
        "ensemble": None,
        "best_model": None,
    }
    
    for model_name, perf in resultados.items():
        metrics["models"][model_name] = {
            "accuracy": float(perf.get("mean_score", 0)),
            "std": float(perf.get("std_score", 0)),
            "trained": perf.get("trained_model") is not None,
            "best_params": perf.get("best_params", {}),
        }
    
    ensemble_models = [m for m in resultados.keys() if "Ensemble" in m]
    if ensemble_models:
        best_ens = max(ensemble_models, key=lambda m: resultados[m]["mean_score"])
        metrics["ensemble"] = {
            "name": best_ens,
            "accuracy": float(resultados[best_ens]["mean_score"]),
            "weights": resultados[best_ens].get("weights", []),
        }
    
    best_model = max(resultados.keys(), key=lambda m: resultados[m].get("mean_score", 0))
    metrics["best_model"] = {
        "name": best_model,
        "accuracy": float(resultados[best_model]["mean_score"]),
    }
    
    # Salvar JSON
    output_path = "output/relatorios/metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Métricas exportadas: {output_path}")
    return metrics
```

2. Usar antes de retornar:
```python
metrics = export_metrics_json(resultados, script_total_time)

# Log resumo final
logger.info("=" * 80)
logger.info("📊 RESUMO FINAL DO PIPELINE")
logger.info("=" * 80)
logger.info(f"Tempo total: {script_total_time.total_seconds():.2f}s")
logger.info(f"Modelos treinados: {len([m for m in resultados.values() if m.get('trained_model')])}/{len(resultados)}")
logger.info(f"Melhor modelo: {metrics['best_model']['name']} (Acc: {metrics['best_model']['accuracy']:.4f})")
logger.info(f"Ensemble: {metrics['ensemble']['name'] if metrics['ensemble'] else 'Nenhum'}")
logger.info("=" * 80)
```

---

## 📋 CHECKLIST DE IMPLEMENTAÇÃO

### Fase 1: Correções Críticas (Obrigatório)
- [ ] Corrigir typo `LGB_AVAILABLE` → `LGBM_AVAILABLE`
- [ ] Implementar bloco vazio de Optuna
- [ ] Verificar imports de `titanic_pipeline.core.modeling`
- [ ] Testar que script roda sem erros: `python train.py`

### Fase 2: Melhorias Altas Prioridade (Recomendado)
- [ ] Implementar cache com versionamento
- [ ] Implementar ensemble robusto com normalização de pesos
- [ ] Implementar tratamento granular de erros
- [ ] Implementar validação de features
- [ ] Testar que pipeline completa sem erros

### Fase 3: Melhorias Médias Prioridade (Opcional)
- [ ] Implementar smoke tests
- [ ] Adicionar docstrings completas
- [ ] Implementar exportação de métricas JSON
- [ ] Adicionar exemplos de extensão

### Fase 4: Validação
- [ ] Rodar pipeline completo: `python train.py`
- [ ] Verificar que `output/submission.csv` foi gerado
- [ ] Verificar que `output/relatorios/metrics.json` foi criado
- [ ] Revisar logs em `titanic_ml.log`
- [ ] Validar relatórios em `output/relatorios/`

---

## 🧪 TESTES APÓS IMPLEMENTAÇÃO

### Teste 1: Execução Básica
```bash
python train.py
# Esperado: Pipeline completa sem exceções
# Arquivo output/submission.csv criado
# Arquivo titanic_ml.log populado
```

### Teste 2: Cache Funcionando
```bash
python train.py  # 1ª vez (sem cache)
python train.py  # 2ª vez (com cache) - DEVE ser mais rápido
```

### Teste 3: Validação de Schema
```bash
# Renomear ou corromper train.csv, rodar:
python train.py
# Esperado: Error loggado, pipeline encerrado gracefully
```

### Teste 4: Smoke Test
```python
# Em config.py:
CONFIG["run_smoke_tests"] = True
# python train.py
# Esperado: "SMOKE TESTS PASSARAM!" no log
```

---

## 📌 NOTAS IMPORTANTES

1. **Compatibilidade**: Código deve rodar em Python 3.8+
2. **Logging**: Sempre usar `logger.` em vez de `print()`
3. **Assertions**: Usar para validações críticas, não genéricas
4. **Type hints**: Adicionar quando possível (não é obrigatório)
5. **Versionamento**: Incrementar `FEATURE_SCHEMA_VERSION` quando features mudam
6. **Git**: Commit após cada fase com mensagens descritivas:
   - "fix: corrigir typo LGBM_AVAILABLE"
   - "feat: implementar cache com versionamento"
   - "refactor: adicionar tratamento granular de erros"

---

## 🎯 RESULTADO ESPERADO

Após implementar este prompt, o pipeline deve:

✅ Rodar sem erros críticos  
✅ Suportar cache inteligente com versionamento  
✅ Criar ensembles robusto com ≥3 modelos  
✅ Tratar erros granularmente e fazer retry  
✅ Validar consistência de features  
✅ Exportar métricas estruturadas em JSON  
✅ Gerar submission.csv e relatórios  
✅ Ser facilmente extensível com novos modelos  
✅ Documentado e testável  

---

**Autor**: Dagoberto Candeias de Moraes (118550)  
**Data Criação**: Dezembro 2025  
**Versão Prompt**: 1.0  
**Status**: Pronto para implementação
