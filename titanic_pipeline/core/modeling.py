"""
Modeling module for Titanic ML Pipeline.
Contains functions for training, tuning, and ensemble modeling.
"""

import logging
import os
import pickle
from typing import Dict, Any, Tuple, List
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

def safe_mean(arr: np.ndarray) -> float:
    """Compute safe mean, handling NaNs and infinities."""
    mean_val = np.nanmean(arr)
    if not np.isfinite(mean_val):
        return 0.0
    return mean_val

def train_single_model(
    model_name: str,
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train a single model with cross-validation and multiple metrics.

    Args:
        model_name: Name of the model.
        model_class: Unfitted model class.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of CV folds.

    Returns:
        Dict with trained model, scores, and metrics.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    min_class_count = counts.min()

    if min_class_count >= 2 and cv_folds <= min_class_count:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=max(2, min(cv_folds, len(y_train))), shuffle=True, random_state=42)

    scoring = ['accuracy', 'roc_auc', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(
        model_class, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score='raise'
    )

    metrics = {}
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        metrics[f'mean_{metric}'] = safe_mean(scores)
        metrics[f'std_{metric}'] = safe_mean(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    # Train final model on full data
    final_model = clone(model_class)
    final_model.fit(X_train, y_train)

    return {
        'trained_model': final_model,
        'mean_score': metrics['mean_accuracy'],
        'std_score': metrics['std_accuracy'],
        'mean_auc': metrics['mean_roc_auc'],
        'std_auc': metrics['std_roc_auc'],
        'mean_precision_macro': metrics['mean_precision_macro'],
        'std_precision_macro': metrics['std_precision_macro'],
        'mean_recall_macro': metrics['mean_recall_macro'],
        'std_recall_macro': metrics['std_recall_macro'],
        'mean_f1_macro': metrics['mean_f1_macro'],
        'std_f1_macro': metrics['std_f1_macro'],
        'cv_folds_used': cv.n_splits,
        'model_name': model_name
    }

def tune_with_randomized_search(
    model_class,
    param_distributions: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10
) -> Tuple[Any, Dict]:
    """
    Perform randomized search for hyperparameter tuning.

    Args:
        model_class: Model class.
        param_distributions: Parameter distributions.
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of iterations.

    Returns:
        Best estimator and best params.
    """
    search = RandomizedSearchCV(
        model_class, param_distributions, n_iter=n_iter, cv=5,
        scoring='accuracy', n_jobs=-1, random_state=42, error_score='raise'
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def build_stacking_ensemble(
    base_models: List[Tuple[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_model=None
) -> Any:
    """
    Build and train a stacking ensemble.

    Args:
        base_models: List of (name, model) tuples.
        X_train: Training features.
        y_train: Training labels.
        meta_model: Meta model (default LogisticRegression).

    Returns:
        Trained stacking model.
    """
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    stacking.fit(X_train, y_train)

    # Save models (optional, for inspection)
    import pickle
    import os
    os.makedirs("output/models", exist_ok=True)
    with open("output/models/trained_stacking.pkl", "wb") as f:
        pickle.dump(stacking, f)
    for name, model in base_models:
        with open(f"output/models/base_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return stacking


def get_base_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna um dicionário de modelos base para treinamento."""
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "Logistic Regression": LogisticRegression(
            random_state=config["random_state"], max_iter=1000
        ),
        "SVC": SVC(probability=True, random_state=config["random_state"]),
        "KNN": KNeighborsClassifier(),
    }
    if XGB_AVAILABLE:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=config["random_state"], verbosity=0
        )
    if LGBM_AVAILABLE:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=100, random_state=config["random_state"], verbosity=-1
        )
    return modelos


def objective(trial, model_name: str, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> float:
    """Função 'objective' para o Optuna."""
    if model_name in ["Random Forest", "RandomForest"]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": config["random_state"],
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
    elif model_name == "XGBoost" and XGB_AVAILABLE:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": config["random_state"],
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
    elif model_name == "LightGBM" and LGBM_AVAILABLE:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "random_state": config["random_state"],
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
    else:
        return 0.0

    skf = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]
    )
    score = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    return score


def save_model_pipeline(preprocessor, model, filepath: str) -> None:
    """Cria e salva um pipeline com pré-processador e modelo."""
    from sklearn.pipeline import Pipeline

    logger.info(f"💾 Salvando pipeline do modelo em {filepath}...")
    try:
        # Verifica se o pré-processador está treinado antes de salvar
        try:
            check_is_fitted(preprocessor)
        except Exception as fit_error:
            logger.error(
                f"❌ Preprocessor não está treinado. Abortando salvamento do pipeline. Erro: {fit_error}",
                exc_info=True,
            )
            return

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        with open(filepath, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("   ✅ Pipeline do modelo salvo com sucesso.")
    except Exception as e:
        logger.error(f"   ❌ Falha ao salvar o pipeline do modelo: {e}", exc_info=True)