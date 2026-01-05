CONFIG = {
    "run_smoke_tests": True,
    "debug_mode": True,
    "parallel_jobs": 1,
    "cv_folds": 3,
    "random_state": 25,
    "cache_enabled": False,
    "fast_mode": True,
    "feature_selection": False,
    "use_optuna": False,
    "optuna_trials": 10,
    "enhanced_balance": False,
    "log_level": 20,  # INFO
}

EXPECTED_TRAIN_SCHEMA = {
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

EXPECTED_TEST_SCHEMA = {
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

LOGGING_CONFIG = {
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
        "level": 20,
        "handlers": ["file", "console"],
    },
}
