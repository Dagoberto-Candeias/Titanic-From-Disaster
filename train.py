# =============================================================================
# TITANIC - SCRIPT DE TREINAMENTO
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 5.0 (Modular)
# =============================================================================

# Importações da biblioteca padrão
import hashlib
import json
import logging
import logging.config
import multiprocessing
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Importações de terceiros
import matplotlib
matplotlib.use('Agg')  # Set backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Importações do Scikit-learn
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATED_AVAILABLE = True
except ImportError:
    CALIBRATED_AVAILABLE = False
    CalibratedClassifierCV = None

from sklearn.ensemble import (
    VotingClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
)

from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split

# Importações opcionais de terceiros com verificações de disponibilidade
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    LGBMClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
    import optuna.logging
    import optuna.visualization as vis
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Fix for module import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============= CONFIGURAÇÃO DE VERSÃO =============
FEATURE_SCHEMA_VERSION = "1.0.0"  # Incrementar quando features mudam
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
    - output/relatorios/                          (relatórios JSON)
    - titanic_ml.log                              (logs detalhados)

Autor: Dagoberto Candeias de Moraes (118550)
Versão: 5.1 (Modular + Robusto)
"""

# ====================================================

# Imports do pipeline Titanic
from titanic_pipeline.preprocessing import (
    create_family_features,
    extract_title,
    extract_deck,
    extract_ticket_prefix,
    create_feature_pipeline,
    advanced_missing_imputation,
    AdvancedFeatureEngineer,
)
from titanic_pipeline.core.modeling import (
    train_single_model as modular_train_single_model,
    build_stacking_ensemble as modular_build_stacking_ensemble,
    get_base_models,
    objective,
    load_and_predict,
    save_model_pipeline,
)
from titanic_pipeline.core.utils import (
    validate_data_schema,
    get_cache_key,
    cache_result,
    load_cached_result,
)
from titanic_pipeline.utils import (
    ensure_feature_cols_intersection,
    optimize_memory_usage,
)
from titanic_pipeline.core.reporting import (
    generate_reports as modular_generate_reports,
    generate_roc_curves as modular_generate_roc_curves,
    generate_feature_correlation_heatmap as modular_generate_feature_correlation_heatmap,
    generate_model_performance_timeline as modular_generate_model_performance_timeline,
    generate_changelog_and_manifest as modular_generate_changelog_and_manifest,
    save_timing_report,
    log_model_performance_to_csv,
    generate_shap_comparison_plot,
    generate_model_calibration_plots,
    generate_permutation_importance,
)

DEFAULT_CONFIG = {
    "debug_mode": False,
    "parallel_jobs": max(1, multiprocessing.cpu_count() - 1),
    "cv_folds": 5,
    "random_state": 42,
    "cache_enabled": True,
    "fast_mode": False,
    "feature_selection": False,
    "run_smoke_tests": False,
    "optuna_trials": 50,
    "enhanced_balance": False,
    "log_level": logging.INFO,
}

DEFAULT_EXPECTED_TRAIN_SCHEMA = {
    "PassengerId": "int64",
    "Survived": "int64",
    "Pclass": "int64",
    "Name": "object",
    "Sex": "object",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "object",
    "Fare": "float64",
    "Cabin": "object",
    "Embarked": "object",
}

DEFAULT_EXPECTED_TEST_SCHEMA = {
    "PassengerId": "int64",
    "Pclass": "int64",
    "Name": "object",
    "Sex": "object",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "object",
    "Fare": "float64",
    "Cabin": "object",
    "Embarked": "object",
}

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "titanic_ml.log",
            "formatter": "detailed",
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "root": {
        "level": DEFAULT_CONFIG["log_level"],
        "handlers": ["file", "console"],
    },
}

try:
    from config import CONFIG as IMPORTED_CONFIG
    from config import EXPECTED_TRAIN_SCHEMA as IMPORTED_EXPECTED_TRAIN_SCHEMA
    from config import EXPECTED_TEST_SCHEMA as IMPORTED_EXPECTED_TEST_SCHEMA
    from config import LOGGING_CONFIG as IMPORTED_LOGGING_CONFIG

    CONFIG = {**DEFAULT_CONFIG, **IMPORTED_CONFIG}
    EXPECTED_TRAIN_SCHEMA = {**DEFAULT_EXPECTED_TRAIN_SCHEMA, **IMPORTED_EXPECTED_TRAIN_SCHEMA}
    EXPECTED_TEST_SCHEMA = {**DEFAULT_EXPECTED_TEST_SCHEMA, **IMPORTED_EXPECTED_TEST_SCHEMA}
    LOGGING_CONFIG = {**DEFAULT_LOGGING_CONFIG, **IMPORTED_LOGGING_CONFIG}
except ImportError:
    CONFIG = DEFAULT_CONFIG.copy()
    EXPECTED_TRAIN_SCHEMA = DEFAULT_EXPECTED_TRAIN_SCHEMA.copy()
    EXPECTED_TEST_SCHEMA = DEFAULT_EXPECTED_TEST_SCHEMA.copy()
    LOGGING_CONFIG = DEFAULT_LOGGING_CONFIG.copy()

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

EXPECTED_TRAIN_COLUMNS = list(EXPECTED_TRAIN_SCHEMA.keys())
EXPECTED_TEST_COLUMNS = list(EXPECTED_TEST_SCHEMA.keys())

# AdvancedFeatureEngineer imported from titanic_pipeline.preprocessing


# Suppress warnings
warnings.filterwarnings("ignore")

# ============= EXCEÇÕES CUSTOMIZADAS =============
class ModelTrainingError(Exception):
    """Erro durante treinamento de modelo."""
    pass

class DataValidationError(Exception):
    """Erro durante validação de dados."""
    pass

class EnsembleError(Exception):
    """Erro durante criação de ensemble."""
    pass

# ============= FUNÇÕES AUXILIARES =============

def invalidate_cache_version():
    """Incrementar versão para invalidar todos os caches."""
    global FEATURE_SCHEMA_VERSION
    major, minor, patch = FEATURE_SCHEMA_VERSION.split('.')
    FEATURE_SCHEMA_VERSION = f"{major}.{int(minor)+1}.0"
    logger.info(f"🔄 Cache versão atualizada para: {FEATURE_SCHEMA_VERSION}")

def get_versioned_cache_key(data_hash, operation):
    """Gera chave de cache com versionamento de schema."""
    key_str = f"{data_hash}_{operation}_{FEATURE_SCHEMA_VERSION}"
    return hashlib.md5(key_str.encode()).hexdigest()

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
        key=lambda x: x[1].get("mean_score", 0),
        reverse=True
    )[:5]
    
    ensemble_models = []
    raw_weights = []
    
    for name, perf in top_models:
        model = perf.get("trained_model")
        if model is not None and hasattr(model, 'predict_proba'):
            ensemble_models.append((name, model))
            raw_weights.append(perf.get("mean_score", 0))
    
    if len(ensemble_models) < min_models:
        logger.warning(f"⚠️  Apenas {len(ensemble_models)} modelos válidos. Mínimo: {min_models}")
        return None, None
    
    # Normalizar pesos entre weight_bounds
    raw_weights = np.array(raw_weights)
    min_raw = raw_weights.min()
    max_raw = raw_weights.max()
    
    if max_raw == min_raw:
        normalized_weights = np.ones_like(raw_weights) * weight_bounds[0]
    else:
        scaled = (raw_weights - min_raw) / (max_raw - min_raw)
        normalized_weights = weight_bounds[0] + scaled * (weight_bounds[1] - weight_bounds[0])
    
    # Log detalhado
    logger.info("📊 Pesos do Ensemble (normalizados):")
    for (name, _), weight in zip(ensemble_models, normalized_weights):
        logger.info(f"   {name}: {weight:.3f}")
    
    return ensemble_models, normalized_weights.tolist()

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

def export_metrics_json(resultados, script_total_time, feature_cols_count):
    """
    Exporta métricas de treinamento em formato JSON estruturado.
    """
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": script_total_time.total_seconds(),
        "features_count": feature_cols_count,
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
        best_ens = max(ensemble_models, key=lambda m: resultados[m].get("mean_score", 0))
        metrics["ensemble"] = {
            "name": best_ens,
            "accuracy": float(resultados[best_ens].get("mean_score", 0)),
            "weights": resultados[best_ens].get("weights", []),
        }
    
    best_model = max(resultados.keys(), key=lambda m: resultados[m].get("mean_score", 0))
    metrics["best_model"] = {
        "name": best_model,
        "accuracy": float(resultados[best_model].get("mean_score", 0)),
    }
    
    # Salvar JSON
    output_path = "output/relatorios/metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Métricas exportadas: {output_path}")
    return metrics

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
        train_sample = pd.read_csv("data/raw/train.csv").head(50)
        test_sample = pd.read_csv("data/raw/test.csv").head(30)
        assert train_sample.shape[0] > 0, "Train vazio"
        assert test_sample.shape[0] > 0, "Test vazio"
        logger.info(f"   ✅ Dados carregados: train={train_sample.shape}, test={test_sample.shape}")
        
        # 3. Validar schema
        validate_data_schema(train_sample, EXPECTED_TRAIN_COLUMNS, "train.csv")
        validate_data_schema(test_sample, EXPECTED_TEST_COLUMNS, "test.csv")
        logger.info("   ✅ Schema válido")
        
        # 4. Feature Engineering
        fe = AdvancedFeatureEngineer()
        train_fe = fe.advanced_missing_imputation(train_sample)
        train_fe = fe.create_advanced_features(train_fe, is_training=True)
        test_fe = fe.advanced_missing_imputation(test_sample)
        test_fe = fe.create_advanced_features(test_fe, is_training=False)
        logger.info(f"   ✅ Features geradas: train={train_fe.shape}, test={test_fe.shape}")
        
        # 5. Treinar modelo simples
        feature_cols_smoke = [c for c in train_fe.columns if c not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin"]]
        preprocessor = create_feature_pipeline(train_fe, feature_cols_smoke, CONFIG["random_state"])
        X = preprocessor.fit_transform(train_fe[feature_cols_smoke])
        y = train_fe["Survived"]
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        scores = cross_val_score(model, X, y, cv=2, scoring="accuracy")
        assert scores.mean() > 0.3, f"Score muito baixo: {scores.mean():.4f}"
        logger.info(f"   ✅ Modelo treinado: Acc={scores.mean():.4f}")
        
        # 6. Fazer predição
        model.fit(X, y)
        X_test = preprocessor.transform(test_fe[feature_cols_smoke])
        preds = model.predict(X_test)
        assert len(preds) == len(test_fe), "Mismatch predictions"
        logger.info(f"   ✅ Predições geradas: {len(preds)} amostras")
        
        # 7. Salvar e carregar modelo
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

def main():
    """
    Pipeline completo de treinamento para Titanic ML.
    
    Fluxo:
    1. Criar diretórios e carregar dados
    2. Validar schema e otimizar memória
    3. Feature engineering com cache versionado
    4. Treinamento paralelo de modelos base
    5. Otimização de hiperparâmetros (opcional, Optuna)
    6. Criar ensembles robusto (Voting + Stacking)
    7. Gerar visualizações e relatórios
    8. Salvar modelo e fazer submissão
    
    Returns:
        bool: True se pipeline completou com sucesso
        
    Raises:
        FileNotFoundError: Se dados não encontrados
        DataValidationError: Se schema inválido
    """
    script_start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO - TITANIC")
    logger.info("=" * 80)

    try:
        # Smoke tests (opcional)
        if CONFIG.get("run_smoke_tests", False):
            logger.info("🚀 Rodando smoke tests antes do pipeline principal...")
            if not smoke_test():
                logger.warning("⚠️  Smoke tests falharam, mas continuando...")
        
        # 1. Create necessary directories
        logger.info("📁 CRIANDO DIRETÓRIOS...")
        os.makedirs("output/graficos", exist_ok=True)
        os.makedirs("output/relatorios", exist_ok=True)
        os.makedirs("output/models", exist_ok=True)
        os.makedirs("output/cache", exist_ok=True)
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        logger.info("📊 CARREGANDO E VALIDANDO DADOS...")
        train_path = "data/raw/train.csv"
        test_path = "data/raw/test.csv"
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.error(f"Arquivos de dados não encontrados em 'data/raw/'. Certifique-se que 'train.csv' e 'test.csv' existem.")
            return False

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # Otimização de memória
        logger.info("🧠 OTIMIZANDO USO DE MEMÓRIA...")
        train = optimize_memory_usage(train)
        test = optimize_memory_usage(test)

        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(train).values.tobytes()
        ).hexdigest()

        validate_data_schema(train, EXPECTED_TRAIN_COLUMNS, "train.csv")
        validate_data_schema(test, EXPECTED_TEST_COLUMNS, "test.csv")

        logger.info("🔧 CRIANDO FEATURES AVANÇADAS COM CACHE...")
        feature_engineer = AdvancedFeatureEngineer()

        # Impute first to avoid NaNs in derived features
        cache_key_imputation = get_versioned_cache_key(data_hash, "imputation_train")
        cached_imputation = load_cached_result(cache_key_imputation)

        if cached_imputation is not None:
            train = cached_imputation
            logger.info("   📖 Imputação de treino carregada do cache (v" + FEATURE_SCHEMA_VERSION + ")")
        else:
            train = feature_engineer.advanced_missing_imputation(train)
            cache_result(cache_key_imputation, train)
            logger.info("   💾 Imputação de treino processada e cached (v" + FEATURE_SCHEMA_VERSION + ")")

        cache_key_features = get_versioned_cache_key(data_hash, "features_train")
        cached_features = load_cached_result(cache_key_features)

        if cached_features is not None:
            train = cached_features
            logger.info("   📖 Features de treino carregadas do cache (v" + FEATURE_SCHEMA_VERSION + ")")
        else:
            train = feature_engineer.create_advanced_features(train, is_training=True)
            cache_result(cache_key_features, train)
            logger.info("   💾 Features de treino processadas e cached (v" + FEATURE_SCHEMA_VERSION + ")")

        # Process test data similarly
        logger.info("🔧 Processando features para o conjunto de teste...")
        test = feature_engineer.advanced_missing_imputation(test)
        test = feature_engineer.create_advanced_features(test, is_training=False)

        feature_cols = [
            col for col in train.columns if col not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Title", "AgeGroup"]
        ]
        feature_cols = ensure_feature_cols_intersection(train.columns, test.columns, feature_cols)

        logger.info("🤖 TREINANDO MODELOS COM PARALELIZAÇÃO...")
        start_time = datetime.now()

        cache_key_models = get_cache_key(data_hash, "model_results")
        cached_results = load_cached_result(cache_key_models)

        y_train = train["Survived"]

        preprocessor = create_feature_pipeline(
            df=train,
            feature_cols=feature_cols,
            random_state=CONFIG["random_state"]
        )
        X_train_processed = preprocessor.fit_transform(train[feature_cols])

        if cached_results is not None and not CONFIG["debug_mode"]:
            resultados = cached_results
            logger.info("Resultados dos modelos carregados do cache")
        else:
            modelos = get_base_models(CONFIG)
            if hasattr(X_train_processed, "toarray"):
                X_train_np = X_train_processed.toarray()
            else:
                X_train_np = np.asarray(X_train_processed)

            logger.info(f"   🚀 Iniciando treinamento paralelo com {CONFIG['parallel_jobs']} jobs...")

            resultados = {}
            with ProcessPoolExecutor(max_workers=CONFIG["parallel_jobs"]) as executor:
                future_to_model = {
                    executor.submit(
                        modular_train_single_model,
                        name,
                        model,
                        X_train_np,
                        y_train,
                        CONFIG["cv_folds"],
                    ): name
                    for name, model in modelos.items()
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        resultados[model_name] = result
                        logger.info(f"   ✅ {model_name} concluído")
                    except ModelTrainingError as e:
                        logger.error(f"   ❌ {model_name} falhou (ModelTrainingError): {e}")
                        resultados[model_name] = {
                            "model_name": model_name,
                            "error": f"ModelTrainingError: {str(e)}",
                            "mean_score": 0.0,
                            "std_score": 0.0,
                            "trained_model": None,
                        }
                    except MemoryError:
                        logger.error(f"   ❌ {model_name} falhou: Memória insuficiente")
                        resultados[model_name] = {
                            "model_name": model_name,
                            "error": "MemoryError",
                            "mean_score": 0.0,
                            "std_score": 0.0,
                            "trained_model": None,
                        }
                    except Exception as e:
                        logger.error(f"   ❌ {model_name} falhou ({type(e).__name__}): {e}")
                        resultados[model_name] = {
                            "model_name": model_name,
                            "error": f"{type(e).__name__}: {str(e)}",
                            "mean_score": 0.0,
                            "std_score": 0.0,
                            "trained_model": None,
                        }

            cache_result(cache_key_models, resultados)
            logger.info("   💾 Resultados dos modelos cached")

        elapsed = datetime.now() - start_time
        logger.info(f"Modelos treinados: {len(resultados)} modelos em {elapsed.total_seconds():.2f}s")

        if OPTUNA_AVAILABLE and not CONFIG.get("fast_mode", False):
            logger.info("🔥 OTIMIZANDO HIPERPARÂMETROS COM OPTUNA...")
            optuna_start_time = datetime.now()

            models_to_optimize = ["Random Forest", "XGBoost", "LightGBM"]
            X_train_opt = X_train_processed
            if hasattr(X_train_opt, "toarray"):
                X_train_opt = X_train_opt.toarray()
            else:
                X_train_opt = np.asarray(X_train_opt)

            for model_name in models_to_optimize:
                if model_name in resultados and resultados[model_name].get("trained_model") is not None:
                    logger.info(f"   Otimizando {model_name}...")
                    try:
                        study = optuna.create_study(direction="maximize")
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                        study.optimize(
                            lambda trial: objective(trial, model_name, X_train_opt, y_train, CONFIG),
                            n_trials=CONFIG.get("optuna_trials", 30),
                        )
                        
                        best_params = study.best_params
                        best_trial = study.best_trial
                        best_value = best_trial.value
                        
                        logger.info(f"   ✅ Melhores params para {model_name}: {best_params}")
                        logger.info(f"   ✅ Melhor score: {best_value:.4f}")
                        
                        # Retreinar com best params
                        if model_name == "Random Forest":
                            from sklearn.ensemble import RandomForestClassifier as RF_Optuna
                            optimized_model = RF_Optuna(**best_params, random_state=CONFIG["random_state"])
                        elif model_name == "XGBoost" and XGB_AVAILABLE:
                            optimized_model = XGBClassifier(**best_params, random_state=CONFIG["random_state"], use_label_encoder=False, eval_metric='logloss')
                        elif model_name == "LightGBM" and LGBM_AVAILABLE:
                            optimized_model = LGBMClassifier(**best_params, random_state=CONFIG["random_state"])
                        else:
                            logger.warning(f"   ⚠️  Modelo {model_name} não foi otimizado")
                            continue
                        
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
                        logger.info(f"   ✅ {model_name}_Optuna: Acc={opt_scores.mean():.4f} ± {opt_scores.std():.4f}")
                    except Exception as e:
                        logger.error(f"   ❌ Optuna para {model_name} falhou: {e}")

            optuna_elapsed = datetime.now() - optuna_start_time
            logger.info(f"   ✅ Otimização concluída em {optuna_elapsed.total_seconds():.2f}s")

        logger.info("CRIANDO ENSEMBLE E GRÁFICO DE COMPARAÇÃO...")
        valid_results = {k: v for k, v in resultados.items() if v.get("trained_model") is not None}
        if not valid_results:
            raise EnsembleError("Nenhum modelo foi treinado com sucesso para criar o ensemble.")

        # Usar função robusta de ensemble
        ensemble_models, ensemble_weights = prepare_ensemble_models(valid_results, min_models=3)

        if ensemble_models is None:
            logger.error("❌ Não foi possível criar ensemble com ≥3 modelos. Usando melhor modelo único.")
            ensemble = None
        else:
            try:
                # Criar VotingClassifier com pesos normalizados
                ensemble = VotingClassifier(
                    estimators=ensemble_models,
                    voting="soft",
                    weights=ensemble_weights
                )
                X_train_ensemble = X_train_processed
                y_train_ensemble = y_train
                ensemble.fit(X_train_ensemble, y_train_ensemble)
                ensemble_scores = cross_val_score(ensemble, X_train_ensemble, y_train_ensemble, cv=CONFIG["cv_folds"], scoring="accuracy")
                resultados["Ensemble_Voting"] = {
                    "model_name": "Ensemble_Voting",
                    "mean_score": ensemble_scores.mean(),
                    "std_score": ensemble_scores.std(),
                    "trained_model": ensemble,
                    "weights": ensemble_weights,
                }
                logger.info(f"✅ Ensemble Voting criado: Acc={ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")

                stacking = modular_build_stacking_ensemble(ensemble_models, X_train_ensemble, y_train_ensemble)
                stacking_scores = cross_val_score(stacking, X_train_ensemble, y_train_ensemble, cv=CONFIG["cv_folds"], scoring="accuracy")
                resultados["Ensemble_Stacking"] = {
                    "model_name": "Ensemble_Stacking",
                    "mean_score": stacking_scores.mean(),
                    "std_score": stacking_scores.std(),
                    "trained_model": stacking,
                }
                logger.info(f"✅ Ensemble stacking criado: Acc={stacking_scores.mean():.4f}")
            except Exception as e:
                logger.error(f"❌ Erro ao criar ensemble: {e}")
                ensemble = None

        logger.info("GERANDO GRÁFICOS E RELATÓRIOS...")
        best_model_name = max(resultados, key=lambda k: resultados[k].get("mean_score", 0))
        best_model = resultados[best_model_name].get("trained_model")

        if best_model:
            y_pred_cv = cross_val_predict(best_model, X_train_processed, y_train, cv=CONFIG["cv_folds"])
            cm = confusion_matrix(y_train, y_pred_cv)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            plt.savefig("output/graficos/03_matriz_confusao.png", dpi=300)
            plt.close()

        # Log de performance dos modelos em CSV
        log_model_performance_to_csv(resultados)

        modular_generate_roc_curves(resultados, X_train_processed, y_train)
        modular_generate_feature_correlation_heatmap(train, feature_cols)
        modular_generate_model_performance_timeline(resultados)

        logger.info("SALVANDO MODELO FINAL...")
        final_model_name = "Ensemble_Stacking" if "Ensemble_Stacking" in resultados else "Ensemble_Voting"
        final_model = resultados.get(final_model_name, {}).get("trained_model")

        if not final_model:
            final_model_name = max(resultados, key=lambda k: resultados[k].get("mean_score", 0))
            final_model = resultados[final_model_name].get("trained_model")

        if final_model:
            pipeline_to_save = save_model_pipeline(preprocessor, final_model, "output/models/best_model_pipeline.pkl")
            logger.info(f"Modelo final '{final_model_name}' salvo como pipeline completo.")
        else:
            logger.error("Nenhum modelo final para salvar.")
            return False

        logger.info("📤 GERANDO ARQUIVO DE SUBMISSÃO A PARTIR DO MODELO SALVO...")
        try:
            # Carrega o pipeline salvo e faz predições para garantir consistência
            predictions = load_and_predict("output/models/best_model_pipeline.pkl", test)
            
            # Criar o DataFrame de submissão
            submission = pd.DataFrame(
                {"PassengerId": test["PassengerId"], "Survived": predictions.astype(int)}
            )
            submission_path = "output/submission.csv"
            submission.to_csv(submission_path, index=False)
            logger.info(f"   ✅ Submission gerada e salva em: {submission_path}")

        except (FileNotFoundError, Exception) as e:
            logger.error(f"❌ Falha ao gerar submissão a partir do modelo salvo: {e}")

        script_total_time = datetime.now() - script_start_time
        modular_generate_reports(resultados, feature_cols, X_train_processed, y_train)
        modular_generate_changelog_and_manifest(feature_cols, resultados, script_total_time)
        save_timing_report(script_total_time, resultados)

        # Exportar métricas estruturadas
        metrics = export_metrics_json(resultados, script_total_time, len(feature_cols))

        # Log resumo final
        logger.info("=" * 80)
        logger.info("📊 RESUMO FINAL DO PIPELINE TITANIC ML")
        logger.info("=" * 80)
        logger.info(f"⏱️  Tempo total: {script_total_time.total_seconds():.2f}s")
        logger.info(f"🤖 Modelos treinados: {len([m for m in resultados.values() if m.get('trained_model')])}/{len(resultados)}")
        logger.info(f"🏆 Melhor modelo: {metrics['best_model']['name']} (Acc: {metrics['best_model']['accuracy']:.4f})")
        logger.info(f"🎯 Ensemble: {metrics['ensemble']['name'] if metrics['ensemble'] else 'Nenhum'} (Acc: {metrics['ensemble']['accuracy']:.4f if metrics['ensemble'] else 'N/A'})")
        logger.info(f"✨ Features: {metrics['features_count']}")
        logger.info(f"📁 Saídas: submission.csv, métricas.json, gráficos, relatórios")
        logger.info("=" * 80)

        logger.info("✅ SUCESSO TOTAL! PIPELINE DE TREINAMENTO CONCLUÍDO!")
        return True

    except Exception as e:
        logger.critical(f"❌ ERRO CRÍTICO NO PIPELINE DE TREINAMENTO: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    main()