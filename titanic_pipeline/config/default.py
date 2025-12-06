"""
Default configuration values for Titanic ML Pipeline.
"""

import logging

# Default system configuration
DEFAULT_CONFIG = {
    "debug_mode": False,
    "random_state": 42,
    "parallel_jobs": 1,
    "cv_folds": 15,
    "cache_enabled": True,
    "fast_mode": False,
    "feature_selection": True,
    "run_smoke_tests": True,
    "optuna_trials": 150,
    "use_knn_imputation": True,
    "enhanced_balance": True,
    "generate_all_plots": True,
    "log_level": logging.INFO,
    "max_features_for_shap": 100,
    "kfold_te_splits": 5,
    "smote_k": 5,
    "smote_strategy": "auto",
    "calibration_method": "isotonic",
    "calibration_cv": 3,
    "permutation_repeats": 5,
    "randomized_n_iter": 100,
    "report_include_images": True,
    "selection_threshold": 0.01,
    "te_prior": 10,
}

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": DEFAULT_CONFIG["random_state"],
        "n_jobs": 1,
    },
    "XGBoost": {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 0.2,
        "random_state": DEFAULT_CONFIG["random_state"],
        "eval_metric": "logloss",
        "early_stopping_rounds": 10,
        "verbosity": 0,
        "n_jobs": -1,
    },
    "LightGBM": {
        "n_estimators": 1500,
        "learning_rate": 0.015,
        "num_leaves": 40,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": DEFAULT_CONFIG["random_state"],
        "early_stopping_round": 10,
        "n_jobs": -1,
        "verbosity": -1,
    },
    "CatBoost": {
        "iterations": 800,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_strength": 0.8,
        "border_count": 128,
        "verbose": 0,
        "random_state": DEFAULT_CONFIG["random_state"],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": (100, 75, 50, 25, 5),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": DEFAULT_CONFIG["random_state"],
        "early_stopping": True,
    },
    "GaussianProcessClassifier": {
        "random_state": DEFAULT_CONFIG["random_state"],
        "max_iter_predict": 100,
    },
    "BaggingClassifier": {
        "n_estimators": 50,
        "max_samples": 0.8,
        "max_features": 0.8,
        "bootstrap": True,
        "bootstrap_features": False,
        "random_state": DEFAULT_CONFIG["random_state"],
        "n_jobs": -1,
    },
}

# Default feature engineering configuration
DEFAULT_FEATURE_CONFIG = {
    "create_feat_prefix": True,
    "advanced_interactions": True,
    "polynomial_features": True,
    "target_encoding": True,
    "demographic_features": True,
    "kfold_te": True,
    "missing_indicators": True,
    "bins_categories": True,
}

# Default report configuration
DEFAULT_REPORT_CONFIG = {
    "generate_md": True,
    "generate_docx": True,
    "generate_pdf": True,
    "include_table_images": True,
    "include_calibration_plots": True,
    "include_feature_importance": True,
    "include_shap_comparison": True,
}

# Default schema validation
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

# Default logging configuration
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

# Default cache configuration
DEFAULT_CACHE_CONFIG = {
    "cache_dir": "output/cache",
    "cache_hyperparams": True,
    "cache_models": True,
    "cache_features": True,
    "cache_plots": True,
}

# Default test configuration
DEFAULT_TEST_CONFIG = {
    "run_unit_tests": True,
    "run_integration_tests": True,
    "test_accuracy_threshold": 0.78,
    "test_feature_count": 25,
    "smoke_coverage": ["smote", "calibration", "feature_select"],
}
