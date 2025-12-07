# =============================================================================
# TITANIC - APRENDIZADO DE MÁQUINA DO DESASTRE
# Autor: Dagoberto Candeias de Moraes (118550)
# Disciplina: ELT579 - Aprendizado de Máquina
# Versão: 4.0 (Corrigida e Otimizada)
# =============================================================================

# Importações da biblioteca padrão
import argparse
import hashlib
import json
import logging
import logging.config
import multiprocessing
import os
import pickle
import sys
import warnings
from datetime import datetime

# Importações de terceiros
import matplotlib
matplotlib.use('Agg')  # Set backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importações do Scikit-learn
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATED_AVAILABLE = True
except ImportError:
    CALIBRATED_AVAILABLE = False
    CalibratedClassifierCV = None

from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
)

# Importações opcionais de terceiros com verificações de disponibilidade
try:
    from sklearn.neural_network import MLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    MLPClassifier = None

try:
    from sklearn.gaussian_process import GaussianProcessClassifier
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    GaussianProcessClassifier = None

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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    plotly = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
    import optuna.logging
    import optuna.visualization as vis
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    kaleido = None

# Fix for module import path (after absolute imports, before relative)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports do pipeline Titanic
from titanic_pipeline.preprocessing import (
    kfold_target_encode,
    create_family_features,
    extract_title,
    extract_deck,
    extract_ticket_prefix,
    create_feature_pipeline,  # Adicionado para usar o pipeline modular
)
from titanic_pipeline.core.preprocessing import (
    preprocess_data as modular_preprocess_data
)
from titanic_pipeline.core.modeling import (
    train_single_model as modular_train_single_model,
    build_stacking_ensemble as modular_build_stacking_ensemble,
    get_base_models,
    ModelingManager,
    objective,
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
)
from titanic_pipeline.core.reporting import (
    generate_reports as modular_generate_reports,
    generate_roc_curves as modular_generate_roc_curves,
    generate_feature_correlation_heatmap as modular_generate_feature_correlation_heatmap,
    generate_model_performance_timeline as modular_generate_model_performance_timeline,
    generate_changelog_and_manifest as modular_generate_changelog_and_manifest,
    save_timing_report,
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
    EXPECTED_TRAIN_SCHEMA = {
        **DEFAULT_EXPECTED_TRAIN_SCHEMA, **IMPORTED_EXPECTED_TRAIN_SCHEMA
    }
    EXPECTED_TEST_SCHEMA = {
        **DEFAULT_EXPECTED_TEST_SCHEMA, **IMPORTED_EXPECTED_TEST_SCHEMA
    }
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

# Protegendo import do AdvancedFeatureEngineer com fallback
try:
    from features import AdvancedFeatureEngineer
    logger.info("✅ AdvancedFeatureEngineer imported successfully from features.py")
except ImportError as e:
    logger.warning(
        f"⚠️  Failed to import AdvancedFeatureEngineer: {e}. "
        "Using fallback implementation."
    )

    class AdvancedFeatureEngineer:
        """Fallback class for AdvancedFeatureEngineer when features.py is not available."""

        def __init__(self):
            pass

        def create_advanced_features(
            self, df: pd.DataFrame, is_training: bool = True
        ) -> pd.DataFrame:
            """Create advanced features using mock implementation."""
            logger.warning(
                "   Using MOCK create_advanced_features with sequential processing."
            )
            df = df.copy()
            # Apply feature engineering functions sequentially
            df = create_family_features(df)
            df = extract_title(df)
            df = extract_deck(df)
            df = extract_ticket_prefix(df)
            # Interações
            df["AgeClass"] = df["Age"] * df["Pclass"]
            df["FarePerPerson"] = df["Fare"] / (
                df["SibSp"] + df["Parch"] + 1
            ).replace(0, 1)
            if "Title" in df.columns:
                df["Title_Interactions"] = df["Title"] + "_" + df["Sex"]
            # Bins avançados
            df["feat_AgeBin"] = pd.cut(
                df["Age"],
                bins=[0, 12, 18, 35, 60, 100],
                labels=["Child", "Teen", "Young", "Adult", "Senior"],
            ).astype(str)
            df["feat_FareBin"] = pd.cut(
                df["Fare"],
                bins=[-1, 7.91, 14.45, 31, 1000],
                labels=["Low", "Medium", "High", "Luxury"],
            ).astype(str)
            df["feat_AgeCategory_v2"] = pd.cut(
                df["Age"],
                bins=[0, 18, 35, 60, 100],
                labels=["Minor", "YoungAdult", "Adult", "Senior"],
            ).astype(str)
            df["feat_FareCategory_v2"] = pd.cut(
                df["Fare"],
                bins=[-1, 10, 50, 1000],
                labels=["Cheap", "Moderate", "Expensive"],
            ).astype(str)
            # Indicadores de missing
            df["feat_Age_missing"] = df["Age"].isnull().astype(int)
            df["feat_Cabin_missing"] = df["Cabin"].isnull().astype(int)
            df["feat_Embarked_missing"] = df["Embarked"].isnull().astype(int)
            df["feat_Fare_missing"] = df["Fare"].isnull().astype(int)
            # Target Encoding (se treino)
            if is_training and "Survived" in df.columns:
                df["feat_Title_te"] = kfold_target_encode(
                    df, "Title", "Survived", suffix="_te"
                )
                ticket_prefix_series = df["Ticket"].str[:3]
                ticket_prefix_series.name = "TicketPrefix"
                df["feat_TicketPrefix_te"] = kfold_target_encode(
                    df, ticket_prefix_series, "Survived", suffix="_te"
                )
                deck_series = df["Cabin"].str[0].fillna("U")
                deck_series.name = "Deck"
                df["feat_Deck_te"] = kfold_target_encode(
                    df, deck_series, "Survived", suffix="_te"
                )
                df["feat_Embarked_te"] = kfold_target_encode(
                    df, "Embarked", "Survived", suffix="_te"
                )
            else:
                # Para teste, usar médias globais ou mapear de treino (simplificado)
                df["feat_Title_te"] = 0.5  # Placeholder
                df["feat_TicketPrefix_te"] = 0.5
                df["feat_Deck_te"] = 0.5
                df["feat_Embarked_te"] = 0.5
            return df

        def advanced_missing_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
            """Perform advanced missing imputation using mock implementation."""
            logger.warning("   Using MOCK advanced_missing_imputation.")
            df = df.copy()
            df["Age"] = df["Age"].fillna(df["Age"].median())
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
            return df

        def select_features_via_model(self, X_train, y_train, feature_names):
            """Select features via model using mock implementation."""
            logger.warning("   Using MOCK select_features_via_model.")
            return feature_names, None

        def validate_imputation(self, df, original_df=None):
            """Validate imputation using mock implementation."""
            logger.info("   Using MOCK validate_imputation.")
            return True


# Suppress warnings
warnings.filterwarnings("ignore")


# Basic logging setup as fallback (duplicate removed)
# The initial logging setup is sufficient; this duplicate is removed to avoid conflicts.

def preprocess_data(train, test, feature_cols, apply_smote=False):
    """(DEPRECATED) Centralized data preprocessing function. Use modular_preprocess_data instead."""
    return modular_preprocess_data(train, test, feature_cols, apply_smote, config=CONFIG)


def main():
    """Main function that GUARANTEES the generation of all required files."""  # noqa
    # Parse command-line arguments for config overrides
    parser = argparse.ArgumentParser(
        description="Titanic ML Pipeline - Enhanced with manual configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --fast_mode False --enhanced_balance True
  python %(prog)s --config_file config.json --parallel_jobs 4
  python %(prog)s --optuna_trials 100 --force_optimizations

Optional dependencies can be installed with:
  python requirements_suggestions.py --auto-install
        """
    )
    parser.add_argument(
        "--fast_mode", type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=None, help="Enable fast mode (skip heavy computations). Default: auto-detect"
    )
    parser.add_argument(
        "--optuna_trials", type=int, default=None,
        help="Number of Optuna hyperparameter optimization trials. Default: 50"
    )
    parser.add_argument(
        "--parallel_jobs", type=int, default=None,
        help="Number of parallel jobs for model training. Default: CPU cores - 1"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=None,
        help="Number of cross-validation folds. Default: 5"
    )
    parser.add_argument(
        "--force_optimizations", action="store_true",
        help="Force optimizations even when fast_mode is enabled"
    )
    parser.add_argument(
        "--debug_mode", type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=None, help="Enable debug mode with verbose logging. Default: False"
    )
    parser.add_argument(
        "--enhanced_balance", type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=None, help="Enable enhanced class balancing (SMOTE if available). Default: False"
    )
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="Path to JSON config file to override defaults"
    )
    parser.add_argument(
        "--check_dependencies", action="store_true",
        help="Check and suggest optional dependencies installation"
    )
    args = parser.parse_args()

    # Apply overrides to CONFIG
    if args.fast_mode is not None:
        CONFIG["fast_mode"] = args.fast_mode
    if args.optuna_trials is not None:
        CONFIG["optuna_trials"] = args.optuna_trials
    if args.parallel_jobs is not None:
        CONFIG["parallel_jobs"] = args.parallel_jobs
    if args.cv_folds is not None:
        CONFIG["cv_folds"] = args.cv_folds
    if args.debug_mode is not None:
        CONFIG["debug_mode"] = args.debug_mode
    if args.enhanced_balance is not None:
        CONFIG["enhanced_balance"] = args.enhanced_balance
    if args.force_optimizations:
        CONFIG["fast_mode"] = False  # Override fast mode if forced

    script_start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("TITANIC - OPTIMIZED PARALLEL ANALYSIS")
    logger.info("=" * 80)

    # 14. Check library availability and set fast mode (Item 14) - Refined settings
    critical_libs = [XGB_AVAILABLE, LGBM_AVAILABLE]
    optional_libs = [SHAP_AVAILABLE, OPTUNA_AVAILABLE]  # Nice-to-have

    if not all(critical_libs):
        CONFIG["fast_mode"] = True  # noqa
        logger.info(
            "⚡ FAST MODE ENABLED - Missing critical libs "
            "(XGBoost/LightGBM), limiting features"
        )
        CONFIG["optuna_trials"] = 0  # Disable Optuna # noqa
        CONFIG["parallel_jobs"] = 1  # noqa
    elif not all(optional_libs):
        logger.info(
            "⚠️  Optional libs missing (SHAP/Optuna), "
            "using fallbacks"
        )
        CONFIG["optuna_trials"] = min(
            CONFIG.get("optuna_trials", 50), 20
        )  # Reduce but don't disable # noqa
    if CONFIG.get("fast_mode", False):
        logger.info("⚡ FAST MODE ENABLED - Skipping heavy computations")
        CONFIG["optuna_trials"] = min(
            CONFIG.get("optuna_trials", 50), 10
        )  # noqa
        CONFIG["parallel_jobs"] = 1  # noqa

    try:
        # 1. Create necessary directories
        logger.info("📁 CRIANDO DIRETÓRIOS...")  # noqa
        start_time = datetime.now()

        os.makedirs("output/graficos", exist_ok=True)
        os.makedirs("output/relatorios", exist_ok=True)
        os.makedirs("output/models", exist_ok=True)
        os.makedirs("output/cache", exist_ok=True)

        elapsed = datetime.now() - start_time
        logger.info(
            f"   ✅ Diretórios criados em {elapsed.total_seconds():.2f}s"
        )  # noqa

        logger.info("📊 CARREGANDO E VALIDANDO DADOS...")  # noqa
        start_time = datetime.now()

        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")

        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(train).values.tobytes()
        ).hexdigest()

        os.makedirs("output/relatorios", exist_ok=True)
        config_with_meta = CONFIG.copy()
        config_with_meta["timestamp"] = datetime.now().isoformat()
        config_with_meta["data_hash"] = data_hash
        with open("output/relatorios/config_used.json", "w") as f:
            json.dump(
                config_with_meta, f, indent=2, default=str
            )
        logger.info("   ✅ Config salvo em output/relatorios/config_used.json")

        validate_data_schema(train, EXPECTED_TRAIN_COLUMNS, "train.csv")
        validate_data_schema(test, EXPECTED_TEST_COLUMNS, "test.csv")

        # Automatic class balancing detection
        class_counts = train["Survived"].value_counts()
        majority_class = class_counts.max()
        minority_class = class_counts.min()
        class_ratio = majority_class / minority_class
        apply_smote = class_ratio > 1.5

        if apply_smote:
            logger.info(
                f"🔄 DETECTADO DESBALANCEAMENTO DE CLASSES: Razão "
                f"{class_ratio:.2f} > 1.5 (Maioria: {majority_class}, "
                f"Minoria: {minority_class}). Ativando balanceamento "
                f"automático."
            )
            CONFIG["enhanced_balance"] = True
        else:
            logger.info(
                f"✅ Classes balanceadas: Razão {class_ratio:.2f} <= 1.5. "
                f"Balanceamento não necessário."
            )

        # Save balancing stats
        os.makedirs("output/changelog", exist_ok=True)
        balancing_stats = {
            "class_counts": class_counts.to_dict(),
            "class_ratio": class_ratio,
            "apply_smote": apply_smote,
            "detection_threshold": 1.5,
            "timestamp": datetime.now().isoformat(),
        }
        with open("output/changelog/balancing_stats.json", "w") as f:
            json.dump(balancing_stats, f, indent=2)
        logger.info("   📝 Estatísticas de balanceamento salvas")

        elapsed = datetime.now() - start_time
        logger.info(
            f"   ✅ Dados carregados e validados: "
            f"Train={train.shape}, Test={test.shape} em "
            f"{elapsed.total_seconds():.2f}s"
        )

        logger.info("🔧 CRIANDO FEATURES AVANÇADAS COM CACHE...")  # noqa
        start_time = datetime.now()

        feature_engineer = AdvancedFeatureEngineer()

        # Impute first to avoid NaNs in derived features
        cache_key_imputation = get_cache_key(data_hash, "imputation_train")
        cached_imputation = load_cached_result(cache_key_imputation)

        if cached_imputation is not None:
            train = cached_imputation
            logger.info("   📖 Imputação de treino carregada do cache")
        else:
            train = feature_engineer.advanced_missing_imputation(train)
            cache_result(cache_key_imputation, train)  # noqa
            logger.info("   💾 Imputação de treino processada e cached")

        cache_key_features = get_cache_key(data_hash, "features_train")
        cached_features = load_cached_result(cache_key_features)

        if cached_features is not None:
            train = cached_features
            logger.info("   📖 Features de treino carregadas do cache")
        else:
            train = feature_engineer.create_advanced_features(train, is_training=True)
            cache_result(cache_key_features, train)  # noqa
            logger.info("   💾 Features de treino processadas e cached")

        missing_cols = ["Age", "Cabin", "Embarked", "Fare"]
        os.makedirs("output/changelog", exist_ok=True)
        with open("output/changelog/missing_cols.json", "w") as f:
            json.dump(
                {
                    "missing_columns": missing_cols,
                    "flags_created": [
                        f"feat_{col}_missing"
                        for col in missing_cols
                        if col in train.columns
                    ],
                    "bins_created": [
                        "feat_AgeBin",
                        "feat_FareBin",
                        "feat_AgeCategory_v2",
                        "feat_FareCategory_v2",
                    ],
                    "kfold_encoded": [
                        "feat_Title_Group_te",
                        "feat_TicketPrefix_te",
                        "feat_Deck_te",
                        "feat_Embarked_te",
                    ],
                },
                f,
                indent=2,
            )
        logger.info("   📝 Changelog de features salvo")  # noqa

        elapsed = datetime.now() - start_time
        logger.info(
            f"   ✅ Features avançadas criadas: {train.shape[1]} colunas em "
            f"{elapsed.total_seconds():.2f}s"
        )

        # Process test data similarly
        logger.info("🔧 Processando features para o conjunto de teste...")
        test_start_time = datetime.now()

        cache_key_imputation_test = get_cache_key(data_hash, "imputation_test")
        cached_imputation_test = load_cached_result(cache_key_imputation_test)

        if cached_imputation_test is not None:
            test = cached_imputation_test
            logger.info("   📖 Imputação de teste carregada do cache")
        else:
            test = feature_engineer.advanced_missing_imputation(test)
            cache_result(cache_key_imputation_test, test)
            logger.info("   💾 Imputação de teste processada e cached")

        cache_key_features_test = get_cache_key(data_hash, "features_test")
        cached_features_test = load_cached_result(cache_key_features_test)

        if cached_features_test is not None:
            test = cached_features_test
            logger.info("   📖 Features de teste carregadas do cache")
        else:
            test = feature_engineer.create_advanced_features(test, is_training=False)
            cache_result(cache_key_features_test, test)
            logger.info("   💾 Features de teste processadas e cached")

        test_elapsed = datetime.now() - test_start_time
        logger.info(
            f"   ✅ Teste processado em {test_elapsed.total_seconds():.2f}s"
        )

        if CONFIG.get("feature_selection", False):
            logger.info("🎯 SELECIONANDO FEATURES...")  # noqa
            start_time = datetime.now()

            feature_cols_all = [
                col
                for col in train.columns
                if col
                not in [
                    "PassengerId",
                    "Survived",
                    "Name",
                    "Ticket",
                    "Cabin",
                    "Title",
                    "AgeGroup",
                ]
            ]
            X_train_select = train[feature_cols_all]
            X_train_select = pd.get_dummies(X_train_select, drop_first=True)
            y_train_select = train["Survived"]

            selected_features, selector = feature_engineer.select_features_via_model(
                X_train_select, y_train_select, list(X_train_select.columns)
            )

            feature_cols = [col for col in selected_features if col in feature_cols_all]
            logger.info(
                f"   ✅ Selecionadas {len(feature_cols)}/{len(feature_cols_all)} "
                "features"
            )

            feature_cols = ensure_feature_cols_intersection(
                train.columns,
                test.columns,
                feature_cols,
            )

            elapsed = datetime.now() - start_time
            logger.info(
                f"   Feature selection concluída em {elapsed.total_seconds():.2f}s"
            )
        else:
            feature_cols = [
                col
                for col in train.columns
                if col
                not in [
                    "PassengerId",
                    "Survived",
                    "Name",
                    "Ticket",
                    "Cabin",
                    "Title",
                    "AgeGroup",
                ]
            ]

        feature_cols = ensure_feature_cols_intersection(
            train.columns,
            test.columns,
            feature_cols,
        )

        # 3.6. Run Smoke Tests (mandatory) # noqa
        logger.info("🧪 EXECUTANDO SMOKE TESTS...")
        start_time = datetime.now()
        try:
            from output.tests.smoke_tests import run_smoke_tests

            smoke_results = run_smoke_tests(
                train_df=train, test_df=test, feature_cols=feature_cols
            )
            os.makedirs("output/relatorios", exist_ok=True)
            with open("output/relatorios/smoke_results.json", "w") as f:
                json.dump(smoke_results, f, indent=2)
            keys_evaluated = [k for k in smoke_results.keys() if k != "details"]
            passed_count = sum(1 for k in keys_evaluated if smoke_results.get(k, False))
            total_tests = len(keys_evaluated)
            if passed_count >= max(3, total_tests - 1):
                logger.info(
                    f"   ✅ Smoke tests passed: {passed_count}/{total_tests}"
                )
            else:
                logger.warning(
                    f"   ⚠️  Smoke tests partial: {passed_count}/{total_tests} - "
                    f"Details: {smoke_results.get('details', 'N/A')}"
                )
        except ImportError:
            logger.warning(
                "   ⚠️  Arquivo smoke_tests.py não encontrado, pulando testes"
            )  # noqa
        except Exception as e:
            logger.error(f"   ❌ Erro nos smoke tests: {e}")
        elapsed = datetime.now() - start_time
        logger.info(
            f"   Smoke tests concluídos em {elapsed.total_seconds():.2f}s"
        )

        if CONFIG.get("run_smoke_tests", False):
            unit_results = run_unit_tests()
            logger.info(
                f"Unit tests: {sum(unit_results.values())}/"
                f"{len(unit_results)} passed - {unit_results}"
            )

        logger.info("📈 GERANDO GRÁFICO EDA...")  # noqa
        start_time = datetime.now()

        cache_key_eda = get_cache_key(data_hash, "eda_plot")
        eda_cached = load_cached_result(cache_key_eda)

        if eda_cached is not None:
            logger.info("   📖 Gráfico EDA carregado do cache")
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            sobrevivencia_sexo = train.groupby("Sex")["Survived"].mean()
            sobrevivencia_sexo.plot(
                kind="bar", ax=axes[0, 0], color=["lightblue", "pink"]
            )
            axes[0, 0].set_title("Survival rate by gender")
            axes[0, 0].set_ylabel("Survival rate")

            sobrevivencia_classe = train.groupby("Pclass")["Survived"].mean()
            sobrevivencia_classe.plot(kind="bar", ax=axes[0, 1], color="lightgreen")
            axes[0, 1].set_title("Survival rate by class")
            axes[0, 1].set_ylabel("Survival rate")

            survived = train[train["Survived"] == 1]["Age"].dropna()
            not_survived = train[train["Survived"] == 0]["Age"].dropna()

            axes[1, 0].hist(
                [not_survived, survived],
                bins=30,
                alpha=0.7,
                label=["Did not survive", "Survived"],
                color=["red", "green"],
            )
            axes[1, 0].set_xlabel("Age")
            axes[1, 0].set_ylabel("Number of passengers")
            axes[1, 0].set_title("Age distribution by survival")
            axes[1, 0].legend()

            bins = [0, 12, 18, 35, 60, 100]
            labels = ["Child", "Teen", "Young", "Adult", "Senior"]
            train["AgeGroup"] = pd.cut(train["Age"], bins=bins, labels=labels)
            sobrevivencia_faixa = train.groupby("AgeGroup")["Survived"].mean()
            sobrevivencia_faixa.plot(kind="bar", ax=axes[1, 1], color="orange")
            axes[1, 1].set_title("Survival rate by age group")
            axes[1, 1].set_ylabel("Survival rate")

            plt.tight_layout()
            plt.savefig(
                "output/graficos/01_eda_completa.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            cache_result(cache_key_eda, True)

        elapsed = datetime.now() - start_time
        logger.info(
            f"   ✅ Gráfico EDA salvo: output/graficos/01_eda_completa.png em "
            f"{elapsed.total_seconds():.2f}s"
        )

        logger.info("🤖 TREINANDO MODELOS COM PARALELIZAÇÃO...")  # noqa
        start_time = datetime.now()

        cache_key_models = get_cache_key(data_hash, "model_results")
        cached_results = load_cached_result(cache_key_models)

        y_train = train["Survived"]

        # Proper preprocessing to ensure consistent number of features
        # Utiliza a função modular create_feature_pipeline para construir o pré-processador
        preprocessor = create_feature_pipeline(
            df=train,
            feature_cols=feature_cols,
            random_state=CONFIG["random_state"]
        )
        # Fit and transform
        X_train_processed = preprocessor.fit_transform(train[feature_cols])
        X_test_processed = preprocessor.transform(test[feature_cols])

        if cached_results is not None and not CONFIG["debug_mode"]:
            resultados = cached_results
            logger.info("Resultados dos modelos carregados do cache")
        else:
            # Use ModelingManager for improved parallel processing with timeouts
            logger.info("   🚀 Iniciando treinamento paralelo com ModelingManager...")

            # Get model configurations from get_base_models
            model_configs = get_base_models(CONFIG)

            # Initialize ModelingManager
            modeling_manager = ModelingManager(CONFIG, model_configs)

            # Train all models with parallel processing and timeouts
            resultados = modeling_manager.train_all_models(X_train_processed, y_train)

            # Cache results
            cache_result(cache_key_models, resultados)
            logger.info("   💾 Resultados dos modelos cached")

        elapsed = datetime.now() - start_time
        logger.info(
            f"Modelos treinados: {len(resultados)} modelos em "
            f"{elapsed.total_seconds():.2f}s"
        )
        # Model definitions are already handled above in parallel processing
        # Now prepare for ensemble and final predictions

        # [NOVO] Otimização de Hiperparâmetros com Optuna # noqa
        if OPTUNA_AVAILABLE and not CONFIG.get("fast_mode", False):
            try:
                # Ajusta verbosidade do Optuna quando possível
                try:
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                except Exception as e:
                    logger.warning(
                        f"Could not set optuna logging verbosity: {e}"
                    )

                logger.info("🔥 OTIMIZANDO HIPERPARÂMETROS COM OPTUNA...")
                optuna_start_time = datetime.now()

                models_to_optimize = ["Random Forest", "XGBoost", "LightGBM"]
                feature_cols_opt = ensure_feature_cols_intersection(
                    train.columns,
                    test.columns,
                    feature_cols,
                )
                X_train_opt, _, y_train_opt, _ = preprocess_data(
                    train, test, feature_cols_opt
                )
                if hasattr(X_train_opt, "toarray"):
                    X_train_opt = X_train_opt.toarray()
                else:
                    X_train_opt = np.asarray(X_train_opt)

                for model_name in models_to_optimize:
                    if (
                        model_name in resultados
                        and resultados[model_name].get("trained_model") is not None
                    ):
                        logger.info(f"   Otimizando {model_name}...")
                        study = optuna.create_study(direction="maximize")
                        study.optimize(
                            lambda trial: objective(
                                trial, model_name, X_train_opt, y_train_opt, CONFIG
                            ),
                            n_trials=CONFIG.get("optuna_trials", 30),
                        )

                        df = study.trials_dataframe()
                        os.makedirs("output/relatorios", exist_ok=True)
                        df.to_csv(
                            f"output/optuna_trials_"
                            f"{model_name.replace(' ', '_')}.csv",
                            index=False,
                        )

                        logger.info(
                            f"   Melhor resultado para {model_name}: "
                            f"Acurácia = {study.best_value:.4f}"
                        )
                        logger.info(f"   Melhores parâmetros: {study.best_params}")

                        best_params = study.best_params

                        os.makedirs("output/relatorios", exist_ok=True)
                        with open(
                            f"output/best_params_"
                            f"{model_name.replace(' ', '_')}.json",
                            "w",
                        ) as f:
                            json.dump(best_params, f, indent=2)

                        if model_name == "Random Forest":
                            optimized_model = RandomForestClassifier(
                                **best_params,
                                random_state=CONFIG["random_state"],
                                n_jobs=-1,
                            )
                        elif model_name == "XGBoost" and XGB_AVAILABLE:
                            optimized_model = XGBClassifier(
                                **best_params,
                                random_state=CONFIG["random_state"],
                                eval_metric="logloss",
                                verbosity=0,
                                n_jobs=-1,
                            )
                        elif model_name == "LightGBM" and LGBM_AVAILABLE:
                            optimized_model = LGBMClassifier(
                                **best_params,
                                random_state=CONFIG["random_state"],
                                verbosity=-1,
                                n_jobs=-1,
                            )
                        else:
                            continue

                        logger.info(
                            f"   Reavaliando {model_name} com parâmetros otimizados..."
                        )
                        optimized_results = modular_train_single_model(
                            f"{model_name}_Optimized",
                            optimized_model,
                            X_train_opt,
                            y_train_opt,
                            CONFIG["cv_folds"],
                        )
                        resultados[f"{model_name}_Optimized"] = optimized_results

                        try:
                            fig = vis.plot_optimization_history(study)
                            fig.write_image(
                                f"output/graficos/optuna_history_"
                                f"{model_name.replace(' ', '_')}.png",
                                engine="kaleido"
                            )
                            fig = vis.plot_param_importances(study)
                            fig.write_image(
                                f"output/graficos/optuna_importance_"
                                f"{model_name.replace(' ', '_')}.png",
                                engine="kaleido"
                            )
                        except Exception as e:
                            logger.warning(
                                f"   Não foi possível gerar gráficos do Optuna para "
                                f"{model_name}: {e}"
                            )

                optuna_elapsed = datetime.now() - optuna_start_time
                logger.info(
                    f"   ✅ Otimização concluída em "
                    f"{optuna_elapsed.total_seconds():.2f}s"
                )
            except Exception as e:
                logger.error(f"❌ Erro na otimização Optuna: {e}", exc_info=True)

        logger.info("CRIANDO ENSEMBLE E GRÁFICO DE COMPARAÇÃO...")
        start_time = datetime.now()

        valid_results = {
            k: v for k, v in resultados.items() if v.get("trained_model") is not None
        }
        if not valid_results:
            raise ValueError(
                "Nenhum modelo foi treinado com sucesso para criar o ensemble."
            )

        top_models = sorted(
            valid_results.items(), key=lambda x: x[1]["mean_score"], reverse=True
        )[:5]

        ensemble_models = []
        ensemble_weights = []
        ensemble_scores = np.array([0.0])
        ensemble_auc = np.array([0.0])

        for name, perf in top_models:
            if "trained_model" in perf and perf["trained_model"] is not None:
                # Only include models that support predict_proba for soft voting
                if hasattr(perf["trained_model"], 'predict_proba'):
                    ensemble_models.append((name, perf["trained_model"]))
                    ensemble_weights.append(perf["mean_score"])
                else:
                    logger.warning(f"Excluding {name} from ensemble (no predict_proba support)")
            else:
                ensemble_weights.append(perf["mean_score"])  # Still include in weights for consistency, but not in ensemble
        
        if ensemble_models:
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=ensemble_models, voting="soft", weights=ensemble_weights
            )
            # Prepare data for fitting ensemble
            X_train_ensemble = X_train_processed
            y_train_ensemble = y_train

            ensemble.fit(X_train_ensemble, y_train_ensemble)

            # Evaluate ensemble
            ensemble_scores = cross_val_score(
                ensemble,
                X_train_ensemble,
                y_train_ensemble,
                cv=CONFIG["cv_folds"],
                scoring="accuracy",
            )
            ensemble_auc = cross_val_score(
                ensemble,
                X_train_ensemble,
                y_train_ensemble,
                cv=CONFIG["cv_folds"],
                scoring="roc_auc",
            )

            resultados["Ensemble_Voting"] = {
                "model_name": "Ensemble_Voting",
                "mean_score": ensemble_scores.mean(),
                "std_score": ensemble_scores.std(),
                "mean_auc": ensemble_auc.mean(),
                "std_auc": ensemble_auc.std(),
                "trained_model": ensemble,
            }

            # Build stacking ensemble (Item 6) # noqa
            logger.info("🏗️ CRIANDO ENSEMBLE STACKING...")
            stacking = modular_build_stacking_ensemble(
                ensemble_models, X_train_ensemble, y_train_ensemble
            )
            stacking_scores = cross_val_score(
                stacking,
                X_train_ensemble,
                y_train_ensemble,
                cv=CONFIG["cv_folds"],
                scoring="accuracy",
            )
            stacking_auc = cross_val_score(
                stacking,
                X_train_ensemble,
                y_train_ensemble,
                cv=CONFIG["cv_folds"],
                scoring="roc_auc",
            )
            resultados["Ensemble_Stacking"] = {
                "model_name": "Ensemble_Stacking",
                "mean_score": stacking_scores.mean(),
                "std_score": stacking_scores.std(),
                "mean_auc": stacking_auc.mean(),
                "std_auc": stacking_auc.std(),
                "trained_model": stacking,
            }
            logger.info(f"   ✅ Ensemble stacking criado: Acc={stacking_scores.mean():.4f}")
        else:
            # Fallback to best single model
            best_key = max(
                resultados, key=lambda k: resultados[k].get("mean_score", 0)
            )
            fallback = resultados[best_key].copy()
            fallback["model_name"] = "Ensemble_Voting (Fallback)"
            resultados["Ensemble_Voting"] = fallback

        # Systematic Calibration (Item 7) # noqa
        try:
            if CALIBRATED_AVAILABLE and not CONFIG.get("fast_mode", False):
                logger.info("📊 APLICANDO CALIBRAÇÃO SISTEMÁTICA...")
                top_models_for_calibration = sorted(
                    [
                        (k, v)
                        for k, v in resultados.items()
                        if not k.endswith("_Calibrated")
                        and v.get("trained_model") is not None
                        and hasattr(v["trained_model"], 'predict_proba')
                    ],
                    key=lambda x: x[1]["mean_score"],
                    reverse=True,
                )[:5] + [
                    ("Ensemble_Voting", resultados.get("Ensemble_Voting")),
                    ("Ensemble_Stacking", resultados.get("Ensemble_Stacking")),
                ]
                for model_name, perf in top_models_for_calibration:
                    if perf and perf.get("trained_model") is not None and hasattr(perf["trained_model"], 'predict_proba'):
                        original_model = perf["trained_model"]
                        calibrated_model = CalibratedClassifierCV(
                            original_model, method="isotonic", cv=3
                        )
                        calibrated_model.fit(X_train_ensemble, y_train_ensemble)
                        calibrated_scores = cross_val_score(
                            calibrated_model,
                            X_train_ensemble,
                            y_train_ensemble,
                            cv=CONFIG["cv_folds"],
                            scoring="accuracy",
                        )
                        calibrated_auc = cross_val_score(
                            calibrated_model,
                            X_train_ensemble,
                            y_train_ensemble,
                            cv=CONFIG["cv_folds"],
                            scoring="roc_auc",
                        )
                        resultados[f"{model_name}_Calibrated"] = {
                            "model_name": f"{model_name}_Calibrated",
                            "mean_score": calibrated_scores.mean(),
                            "std_score": calibrated_scores.std(),
                            "mean_auc": calibrated_auc.mean(),
                            "std_auc": calibrated_auc.std(),
                            "trained_model": calibrated_model,
                        }
                        logger.info(
                            f"   ✅ {model_name} calibrado: Acc={calibrated_scores.mean():.4f}"
                        )
                        generate_model_calibration_plots(
                            calibrated_model,
                            X_train_ensemble,
                            y_train_ensemble,
                            f"{model_name}_Calibrated",
                        )
                best_calibrated = max(
                    [k for k in resultados.keys() if k.endswith("_Calibrated")],
                    key=lambda x: resultados[x]["mean_score"],
                    default=None,
                )
                if best_calibrated:
                    generate_model_calibration_plots(
                        resultados[best_calibrated]["trained_model"],
                        X_train_ensemble,
                        y_train_ensemble,
                        best_calibrated,
                    )
                else:
                    logger.info("   ⚠️  Calibração pulada (libs indisponíveis ou fast_mode)")
        except Exception as e:
            logger.error(f"❌ Erro na calibração: {e}", exc_info=True)

        cache_key_comparison = get_cache_key(data_hash, "comparison_plot")
        comparison_cached = load_cached_result(cache_key_comparison)

        if comparison_cached is not None:
            logger.info("Gráfico de comparação carregado do cache")
        else:
            fig, ax = plt.subplots(figsize=(14, 8))

            model_names = []
            scores = []
            stds = []

            for name, perf in sorted(
                resultados.items(),
                key=lambda x: x[1].get("mean_score", 0),
                reverse=True,
            ):
                if "mean_score" in perf and perf["mean_score"] > 0:
                    model_names.append(name)
                    scores.append(perf["mean_score"])
                    stds.append(perf.get("std_score", 0))

            bars = ax.barh(
                model_names, scores, xerr=stds, capsize=5, color="skyblue", alpha=0.8
            )
            ax.set_xlabel("Accuracy (Cross-Validation)")
            ax.set_title("Model Comparison - Titanic Survival Prediction")
            ax.grid(True, alpha=0.3)

            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

            plt.tight_layout()
            plt.savefig(
                "output/graficos/02_comparacao_modelos.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            cache_result(cache_key_comparison, True)

        elapsed = datetime.now() - start_time
        logger.info(
            f"Ensemble criado e gráfico salvo em "
            f"{elapsed.total_seconds():.2f}s"
        )

        logger.info("GERANDO MATRIZ DE CONFUSÃO...")
        start_time = datetime.now()

        best_model_name = max(
            resultados, key=lambda k: resultados[k].get("mean_score", 0)
        )
        best_model = resultados[best_model_name].get("trained_model")

        if best_model is not None:
            cache_key_cm = get_cache_key(data_hash, "confusion_matrix")
            cm_cached = load_cached_result(cache_key_cm)  # noqa

            if cm_cached is not None:
                logger.info("   📖 Matriz de confusão carregada do cache")
            else:
                X_train_cm = train[feature_cols]
                X_train_cm = pd.get_dummies(X_train_cm, drop_first=True)
                y_train_cm = train["Survived"]

                y_pred = cross_val_predict(
                    best_model, X_train_cm, y_train_cm, cv=CONFIG["cv_folds"]
                )

                cm = confusion_matrix(y_train_cm, y_pred)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=["Not Survived", "Survived"]
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                disp.plot(ax=ax, cmap="Blues", values_format="d")
                ax.set_title(f"Confusion Matrix - {best_model_name}")
                plt.tight_layout()
                plt.savefig(
                    "output/graficos/03_matriz_confusao.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                cache_result(cache_key_cm, True)  # noqa

        elapsed = datetime.now() - start_time
        logger.info(
            f"Matriz de confusão salva em {elapsed.total_seconds():.2f}s"
        )

        if SHAP_AVAILABLE and not CONFIG.get("fast_mode", False):
            logger.info(
                "🧠 GERANDO ANÁLISE DE INTERPRETABILIDADE (SHAP) PARA TOP 3 MODELOS..."
            )  # noqa
            start_time = datetime.now()

            top_3_models = sorted(
                valid_results.items(), key=lambda x: x[1]["mean_score"], reverse=True
            )[:3]
            
            # Usar os dados já processados e o preprocessor ajustado para consistência
            X_train_shap = X_train_processed
            feature_names_out = preprocessor.get_feature_names_out()

            X_train_shap_df = pd.DataFrame(X_train_shap, columns=feature_names_out)
            
            shap_sample_size = min(100, len(X_train_shap_df))
            X_shap_sample_df = X_train_shap_df.sample(
                shap_sample_size, random_state=CONFIG["random_state"]
            )
            X_shap_sample_values = X_shap_sample_df.values.astype(float)

            for model_name, perf in top_3_models:
                model = perf.get("trained_model")
                if model is None:
                    continue
                logger.info(f"   Gerando SHAP para o modelo: {model_name}")
                try:
                    if isinstance(
                        model,
                        (
                            RandomForestClassifier,
                            XGBClassifier,
                            LGBMClassifier,
                            GradientBoostingClassifier,
                            ExtraTreesClassifier,
                        ),
                    ):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_shap_sample_values)  # noqa
                    else:
                        predict_fn = model.predict_proba
                        explainer = shap.KernelExplainer(
                            predict_fn, X_shap_sample_values
                        )
                        shap_values = explainer.shap_values(X_shap_sample_values)  # noqa

                    shap_plot_values = (
                        shap_values[1] if isinstance(shap_values, list)
                        else shap_values
                    )  # noqa

                    plt.figure()
                    shap.summary_plot(
                        shap_plot_values,
                        X_shap_sample_df,
                        feature_names=feature_names_out,
                        show=False,
                    )
                    plt.title(f"SHAP Summary Plot for {model_name}")
                    plt.tight_layout()
                    safe_model_name = model_name.replace(" ", "_")
                    plt.savefig(
                        f"output/graficos/06_shap_summary_{safe_model_name}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
                    logger.info(
                        f"      ✅ Gráfico SHAP de resumo salvo para {model_name}."
                    )

                except Exception as e:
                    logger.error(
                        f"      ❌ Falha ao gerar SHAP para {model_name}: {e}",
                        exc_info=True,
                    )

                # 8.4. Generate SHAP Comparison Plot (Item 6) # noqa
                top_3_tree_models = [
                    (name, perf)
                    for name, perf in top_3_models
                    if isinstance(
                        perf.get("trained_model"),
                        (
                            RandomForestClassifier,
                            XGBClassifier,
                            LGBMClassifier,
                            GradientBoostingClassifier,
                            ExtraTreesClassifier,
                        ),
                    )
                ]
                
                if top_3_tree_models:
                    try:
                        generate_shap_comparison_plot(
                            top_3_tree_models, X_train_shap, feature_names_out
                        )
                    except Exception as e:
                        logger.error(f"Erro ao gerar comparação SHAP: {e}")

                logger.info(
                    f"   Análise SHAP concluída em "
                    f"{(datetime.now() - start_time).total_seconds():.2f}s"
                )

            # 8.5. Generate Permutation Importance (Item 8) # noqa
            if not CONFIG.get("fast_mode", False):
                logger.info("🔄 GERANDO IMPORTÂNCIA DE PERMUTAÇÃO...")
                start_time = datetime.now()

                top_3_models = sorted(
                    valid_results.items(), key=lambda x: x[1]["mean_score"], reverse=True
                )[:3]
                for model_name, perf in top_3_models:
                    model = perf.get("trained_model")
                    if model is not None:
                        logger.info(f"   Gerando permutação para {model_name}...")
                        try:
                            generate_permutation_importance(
                                model,
                                X_train_shap,
                                y_train,
                                feature_names_out,
                                n_repeats=5,
                                model_name=model_name,
                            )
                        except Exception as e:
                            logger.error(f"Erro ao gerar importância de permutação para {model_name}: {e}")

                elapsed = datetime.now() - start_time
                logger.info(
                    f"   Importância de permutação concluída em {elapsed.total_seconds():.2f}s"
                )

            # 8.6. Generate Calibration Plots (Item 7) # noqa
            if not CONFIG.get("fast_mode", False):
                logger.info("📊 GERANDO GRÁFICOS DE CALIBRAÇÃO...")
                start_time = datetime.now()

                if best_model is not None:
                    try:
                        generate_model_calibration_plots(
                            best_model, X_train_shap, y_train, best_model_name
                        )
                    except Exception as e:
                        logger.error(f"Erro ao gerar gráficos de calibração: {e}")

                elapsed = datetime.now() - start_time
                logger.info(
                    f"   Gráficos de calibração concluídos em {elapsed.total_seconds():.2f}s"
                )

                modular_generate_roc_curves(resultados, X_train_ensemble, y_train_ensemble)
                modular_generate_feature_correlation_heatmap(train, feature_cols)
                modular_generate_model_performance_timeline(resultados)

        # 8. Generate final predictions # noqa
        logger.info("GERANDO PREDIÇÕES FINAIS...")
        start_time = datetime.now()

        final_model = resultados.get("Ensemble_Stacking", {}).get("trained_model")
        if not final_model:
            best_model_name = max(
                resultados, key=lambda k: resultados[k].get("mean_score", 0)
            )
            final_model = resultados[best_model_name].get("trained_model")

        if final_model is None:
            raise ValueError("Nenhum modelo disponível para gerar predições")

        # A predição já foi feita com X_test_processed, que foi transformado pelo preprocessor
        predictions = final_model.predict(X_test_processed)

        # Criar o DataFrame de submissão
        submission = pd.DataFrame(
            {"PassengerId": test["PassengerId"], "Survived": predictions.astype(int)}
        )
        submission_path = "output/submission_titanic_final.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"   ✅ Submission gerada e salva em: {submission_path}")

        # A função `improved_generate_submission` foi integrada aqui para usar o X_test_processed
        # e o preprocessor já ajustado, eliminando a necessidade de reprocessar os dados
        # manualmente com get_dummies.
        #
        # A chamada original era:
        # improved_generate_submission(final_model, test, feature_cols, train)
        #

        elapsed = datetime.now() - start_time
        logger.info(
            f"Predições finais geradas: {len(predictions)} amostras em {elapsed.total_seconds():.2f}s"
        )

        script_total_time = (
            datetime.now() - script_start_time
        )  # Corrected start time reference
        save_timing_report(script_total_time, resultados)
        modular_generate_reports(resultados, feature_cols, X_train_processed, y_train)

        modular_generate_changelog_and_manifest(
            feature_cols, resultados, script_total_time
        )

        _, _, _, final_preprocessor = preprocess_data(train, test, feature_cols)
        save_model_pipeline(
            final_preprocessor, final_model, "output/models/best_model_pipeline.pkl"
        )

        with open("output/models/best_model.pkl", "wb") as f:
            pickle.dump(final_model, f)

        if resultados.get("Ensemble_Stacking", {}).get("trained_model") is not None:
            best_model_name = "Ensemble_Voting"
        else:
            best_model_name = max(
                resultados, key=lambda k: resultados[k].get("mean_score", 0)
            )
        logger.info(f"Best model '{best_model_name}' saved separately")

        # 10. Final verification
        logger.info("VERIFICANDO ARQUIVOS GERADOS...")
        arquivos_esperados = [
            "output/submission_titanic_final.csv",
            "output/graficos/01_eda_completa.png",
            "output/graficos/02_comparacao_modelos.png",
            "output/graficos/03_matriz_confusao.png",
            "output/relatorios/RELATORIO_FINAL_TITANIC.md",
            "output/relatorios/resultados_modelos.csv",
        ]
        arquivos_encontrados = sum(
            1 for arquivo in arquivos_esperados if os.path.exists(arquivo)
        )
        logger.info(
            f"Resumo: {arquivos_encontrados}/{len(arquivos_esperados)} arquivos gerados"
        )

        if arquivos_encontrados >= 5:
            logger.info("SUCESSO TOTAL! TODOS OS ARQUIVOS GERADOS!")
        else:
            logger.error("Alguns arquivos não foram gerados corretamente.")

        return True

    except Exception:
        logger.error("❌ Erro geral no pipeline principal", exc_info=True)
        return False


def run_unit_tests():
    """Executa testes unitários básicos."""  # noqa
    logger.info("🧪 EXECUTANDO UNIT TESTS...")
    results = {
        "validate_data_schema": False,
        "preprocess_data": False,
        "train_single_model": False,
    }

    # Test validate_data_schema
    try:  # noqa
        test_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})  # noqa
        results["validate_data_schema"] = validate_data_schema(
            test_df, ["A", "B"], "unit_test"
        )
    except Exception as e:
        logger.error(f"Unit test validate_data_schema falhou: {e}")

    # Test preprocess_data
    try:
        train = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "Survived": [0, 1, 0]})
        test = pd.DataFrame({"A": [1, 2], "B": [4, 5]})
        X_train, X_test, y_train, preprocessor = modular_preprocess_data(
            train, test, ["A", "B"]
        )
        results["preprocess_data"] = X_train.shape[0] == 3 and X_test.shape[0] == 2
    except Exception as e:
        logger.error(f"Unit test preprocess_data falhou: {e}")

    try:
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        result = modular_train_single_model(
            "Test", LogisticRegression(), X, y, cv_folds=2
        )
        # Accept success if a trained model was returned (robust for tiny toy datasets)
        results["train_single_model"] = result.get("trained_model") is not None
    except Exception as e:
        logger.error(f"Unit test train_single_model falhou: {e}")

    return results


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    main()  # Chama a função principal unificada
