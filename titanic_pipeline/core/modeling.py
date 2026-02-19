"""Modeling manager for Titanic ML Pipeline.

Enhanced with parallel processing, memory optimization, and
performance monitoring.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Defer heavy/optional imports (xgboost, lightgbm, catboost, optuna)
# to runtime to avoid import-time binary issues in CI or when
# optional packages are not installed.
XGBClassifier = None
XGB_AVAILABLE = False
LGBMClassifier = None
LGBM_AVAILABLE = False
CatBoostClassifier = None
CATBOOST_AVAILABLE = False
OPTUNA_AVAILABLE = False

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.pipeline import Pipeline

try:
    import joblib
    SERIALIZER = joblib
except ImportError:
    import pickle
    SERIALIZER = pickle

import os

from ..utils import ParallelProcessor

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train):
    """Train a model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a model."""
    from sklearn.metrics import accuracy_score
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def save_model(model, filepath):
    """Save a model."""
    SERIALIZER.dump(model, filepath)


def load_model(filepath):
    """Load a model."""
    return SERIALIZER.load(filepath)


# Models that don't accept random_state parameter
NO_RANDOM_STATE_MODELS = {
    "BernoulliNB",
    "GaussianNB",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "KNeighbors",
}


class ModelingManager:
    """Manages model training, optimization, and ensemble creation."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_configs: Dict[str, Dict[str, Any]] = None,
        pre_configured_models: Dict[str, Any] = None,
    ):
        self.config = config
        self.model_configs = model_configs or {}
        self.pre_configured_models = pre_configured_models or {}
        self.parallel_processor = ParallelProcessor(
            max_workers=config["parallel_jobs"]
        )

    def train_all_models(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train all models sequentially to avoid pickling issues on Windows.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with model results
        """
        msg = "   🔄 Training models sequentially to avoid pickling issues..."
        logger.info(msg)

        # Get available models
        available_models = self._get_available_models()

        # Train models sequentially
        results = {}
        for model_name, model_class in available_models.items():
            try:
                # OPTUNA INTEGRATION
                if (
                    self.config.get("use_optuna")
                    and not self.config.get("fast_mode")
                    and OPTUNA_AVAILABLE
                    and model_name in ["RandomForest", "XGBoost", "LightGBM"]
                ):
                    logger.info(f"   🔥 Optimizing {model_name} with Optuna...")
                    study = optuna.create_study(
                        direction="maximize", verbosity=optuna.logging.WARNING
                    )
                    study.optimize(
                        lambda trial: objective(
                            trial, model_name, X_train, y_train, self.config
                        ),
                        n_trials=self.config.get("optuna_trials", 10),
                    )
                    best_msg = (
                        f"   ✅ Best params for {model_name}: "
                        f"{study.best_params}"
                    )
                    logger.info(best_msg)

                    # Update model config with best params
                    if model_name not in self.model_configs:
                        self.model_configs[model_name] = {}
                    self.model_configs[model_name].update(study.best_params)

                    # Generate Optuna plots
                    try:
                        optuna_dir = "output/graficos/optuna"
                        os.makedirs(optuna_dir, exist_ok=True)

                        # Optimization history
                        try:
                            fig = (
                                optuna.visualization
                                .plot_optimization_history(study)
                            )
                            fname = f"{optuna_dir}/history_{model_name}.png"
                            fig.write_image(fname)
                        except OSError as e:
                            msg = (
                                f"   ⚠️  Could not save Optuna history plot "
                                f"(check kaleido): {e}"
                            )
                            logger.warning(msg)

                        # Parameter importance
                        try:
                            fig = (
                                optuna.visualization
                                .plot_param_importances(study)
                            )
                            fname = f"{optuna_dir}/importance_{model_name}.png"
                            fig.write_image(fname)
                        except OSError as e:
                            msg = (
                                f"   ⚠️  Could not save Optuna importance "
                                f"plot: {e}"
                            )
                            logger.warning(msg)

                    except (ImportError, OSError, AttributeError) as e:
                        msg = f"   ⚠️  Optuna plotting setup failed: {e}"
                        logger.warning(msg)

                result = self._train_single_model(
                    model_name, model_class, X_train, y_train
                )
                results[model_name] = result
                score = result['mean_score']
                logger.info(f"   ✅ {model_name} trained: {score:.4f}")
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.error(f"   ❌ {model_name} training failed: {e}")

        return results

    def _train_single_model(
        self,
        model_name: str,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Train a single model with cross-validation."""
        try:
            # Use pre-configured model if available
            if model_name in self.pre_configured_models:
                model = self.pre_configured_models[model_name]
                msg = f"   📋 Using pre-configured model for {model_name}"
                logger.info(msg)
            else:
                # Create model instance with optimized parameters
                if model_name in self.model_configs:
                    model_params = self.model_configs[model_name].copy()

                    # SAFETY CHECK: Remove random_state for models that
                    # don't support it
                    if (
                        model_name in NO_RANDOM_STATE_MODELS
                        and "random_state" in model_params
                    ):
                        del model_params["random_state"]

                    # Add performance optimizations for slow models
                    if model_name in ["SVC", "KNeighbors"]:
                        max_iter_val = (
                            10000 if model_name == "SVC" else None
                        )
                        n_jobs_val = (
                            1 if model_name == "KNeighbors" else None
                        )
                        model_params.update({
                            "max_iter": max_iter_val,
                            "n_jobs": n_jobs_val,  # Set to 1 for safety
                        })
                    model = model_class(**model_params)
                else:
                    # Get default parameters for the model
                    default_params = self._get_model_default_params(model_name)
                    model = model_class(**default_params)

            # Convert sparse matrix to dense for models that don't
            # support sparse input
            if model_name in [
                "GaussianNB",
                "LinearDiscriminantAnalysis",
                "QuadraticDiscriminantAnalysis",
            ]:
                if hasattr(X_train, "toarray"):
                    X_train = X_train.toarray()

            # Perform cross-validation
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.config["cv_folds"],
                scoring="accuracy",
                n_jobs=1,  # Avoid nested parallelism
            )

            # Train final model
            model.fit(X_train, y_train)

            result = {
                "trained_model": model,
                "mean_score": cv_scores.mean(),
                "std_score": cv_scores.std(),
                "cv_scores": cv_scores.tolist(),
                "model_name": model_name,
                "error": None,
            }

            return result

        except (ValueError, RuntimeError, TypeError) as e:
            logger.error(f"   ❌ Error training {model_name}: {e}")
            # Return failed result instead of raising exception
            result = {
                "trained_model": None,
                "mean_score": 0.0,
                "std_score": 0.0,
                "cv_scores": [],
                "model_name": model_name,
                "error": str(e),
            }
            return result

    def create_ensembles(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create ensemble models from trained base models.

        Args:
            X_train: Training features
            y_train: Training labels
            model_results: Results from individual model training

        Returns:
            Dictionary with ensemble results
        """
        logger.info("   🎯 Creating ensemble models...")

        ensemble_results = {}

        # Prepare base models for ensemble
        base_models = []
        for model_name, result in model_results.items():
            if result.get("trained_model") is not None:
                model = result["trained_model"]
                # Only include models with predict_proba for soft voting
                # ensemble
                if hasattr(model, "predict_proba"):
                    base_models.append((model_name, model))
                else:
                    msg = (
                        f"   ⚠️  Excluding {model_name} from VotingEnsemble "
                        f"(no predict_proba)"
                    )
                    logger.warning(msg)

        if len(base_models) < 3:
            logger.warning(
                f"   ⚠️  Apenas {len(base_models)} modelos válidos. Mínimo: 3"
            )
            return ensemble_results

        # Convert to dense if necessary for ensemble training
        # VotingClassifier and StackingClassifier refit base models.
        # Models like GaussianNB fail with sparse input.
        X_train_ensemble = X_train
        if hasattr(X_train, "toarray"):
            msg = (
                "   🔄 Converting sparse matrix to dense for Ensemble "
                "training..."
            )
            logger.info(msg)
            X_train_ensemble = X_train.toarray()

        try:
            # Voting Classifier
            voting_clf = VotingClassifier(
                estimators=base_models, voting="soft", n_jobs=1
            )

            voting_scores = cross_val_score(
                voting_clf,
                X_train_ensemble,
                y_train,
                cv=self.config["cv_folds"],
                scoring="accuracy",
            )

            voting_clf.fit(X_train_ensemble, y_train)

            ensemble_results["VotingEnsemble"] = {
                "trained_model": voting_clf,
                "mean_score": voting_scores.mean(),
                "std_score": voting_scores.std(),
                "cv_scores": voting_scores.tolist(),
                "model_name": "VotingEnsemble",
            }

            logger.info(f"   ✅ Voting Ensemble: {voting_scores.mean():.4f}")

            # Stacking Classifier
            meta_model = LogisticRegression(
                random_state=self.config["random_state"], max_iter=1000
            )

            stacking_clf = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=self.config["cv_folds"],
                n_jobs=1,
                passthrough=True,
            )

            stacking_scores = cross_val_score(
                stacking_clf,
                X_train_ensemble,
                y_train,
                cv=self.config["cv_folds"],
                scoring="accuracy",
            )

            stacking_clf.fit(X_train_ensemble, y_train)

            ensemble_results["StackingEnsemble"] = {
                "trained_model": stacking_clf,
                "mean_score": stacking_scores.mean(),
                "std_score": stacking_scores.std(),
                "cv_scores": stacking_scores.tolist(),
                "model_name": "StackingEnsemble",
            }

            score = stacking_scores.mean()
            logger.info(f"   ✅ Stacking Ensemble: {score:.4f}")
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"   ❌ Ensemble creation failed: {e}")

        return ensemble_results

    def _get_available_models(self) -> Dict[str, Any]:
        """Get dictionary of available models based on installed libraries."""
        models = {
            "RandomForest": "sklearn.ensemble.RandomForestClassifier",
            "LogisticRegression": "sklearn.linear_model.LogisticRegression",
            "SVC": "sklearn.svm.SVC",
            "KNeighbors": "sklearn.neighbors.KNeighborsClassifier",
            "GaussianNB": "sklearn.naive_bayes.GaussianNB",
            "MLPClassifier": "sklearn.neural_network.MLPClassifier",
            "GradientBoosting": "sklearn.ensemble.GradientBoostingClassifier",
            "ExtraTrees": "sklearn.ensemble.ExtraTreesClassifier",
            "AdaBoost": "sklearn.ensemble.AdaBoostClassifier",
            "Bagging": "sklearn.ensemble.BaggingClassifier",
            "SGDClassifier": "sklearn.linear_model.SGDClassifier",
            "RidgeClassifier": "sklearn.linear_model.RidgeClassifier",
            "LinearSVC": "sklearn.svm.LinearSVC",
            "DecisionTree": "sklearn.tree.DecisionTreeClassifier",
            "BernoulliNB": "sklearn.naive_bayes.BernoulliNB",
            "LinearDiscriminantAnalysis": (
                "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
            ),
            "QuadraticDiscriminantAnalysis": (
                "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
            ),
        }

        # Check for optional libraries at runtime to avoid hard failures
        try:
            from xgboost import XGBClassifier as _XGB

            models["XGBoost"] = _XGB
        except Exception:
            logger.warning("   ⚠️  XGBoost not available or failed to import")

        try:
            from lightgbm import LGBMClassifier as _LGBM

            models["LightGBM"] = _LGBM
        except Exception:
            logger.warning("   ⚠️  LightGBM not available or failed to import")

        try:
            from catboost import CatBoostClassifier as _CAT

            models["CatBoost"] = _CAT
        except Exception:
            logger.warning("   ⚠️  CatBoost not available or failed to import")

        # Convert string paths to actual classes
        available_models = {}
        for name, model_path in models.items():
            if isinstance(model_path, str):
                try:
                    module_name, class_name = model_path.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[class_name])
                    available_models[name] = getattr(module, class_name)
                except (ImportError, AttributeError):
                    logger.warning(f"   ⚠️  Could not import {model_path}")
            else:
                available_models[name] = model_path

        logger.info(f"   📊 Available models: {list(available_models.keys())}")
        return available_models

    def _get_model_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model to ensure robust training."""
        base_params = {}
        if model_name not in NO_RANDOM_STATE_MODELS:
            base_params["random_state"] = self.config["random_state"]

        # Models that need special parameters for ensemble compatibility
        special_params = {
            "SVC": {
                "probability": True,  # Required for soft voting in ensembles
                "max_iter": 10000,  # Prevent convergence issues
            },
            "LinearSVC": {
                "max_iter": 10000,  # Prevent convergence issues
            },
            "MLPClassifier": {
                "max_iter": 1000,  # Prevent convergence issues
                "early_stopping": True,
            },
            "SGDClassifier": {
                "max_iter": 1000,  # Prevent convergence issues
                "loss": "log_loss",  # Enable probabilities
            },
            "LogisticRegression": {
                "max_iter": 1000,  # Prevent convergence issues
            },
            "KNeighbors": {
                "n_jobs": 1,  # Avoid nested parallelism
            },
            "XGBoost": {
                "verbosity": 0,  # Reduce output
            },
            "LightGBM": {
                "verbosity": -1,  # Reduce output
            },
        }

        # Add special parameters if available
        if model_name in special_params:
            base_params.update(special_params[model_name])

        return base_params


def train_single_model(
    model_name: str,
    model,
    X_train,
    y_train,
    cv_folds: int = 5,
):
    """Train a single model with cross-validation."""
    try:
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="accuracy"
        )
        # Train final model
        model.fit(X_train, y_train)

        result = {
            "trained_model": model,
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "model_name": model_name,
        }
        return result
    except Exception as e:
        logger.error(f"Error training {model_name}: {e}")
        raise


def build_stacking_ensemble(base_models, X_train, y_train):
    """Build a stacking ensemble."""
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=1,
        passthrough=True,
    )
    stacking.fit(X_train, y_train)
    return stacking


def get_base_models(config):
    """Get base models dictionary."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,  # Aumenta o número de árvores (padrão era 100)
            max_depth=10,  # Limita a profundidade para evitar overfitting
            min_samples_split=5,  # Exige mais amostras para dividir um nó
            random_state=config["random_state"],
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "Bagging": BaggingClassifier(
            n_estimators=100, random_state=config["random_state"]
        ),
        "Logistic Regression": LogisticRegression(
            random_state=config["random_state"], max_iter=1000
        ),
        "SGD Classifier": SGDClassifier(
            random_state=config["random_state"], max_iter=1000, loss="log_loss"
        ),
        "Ridge Classifier": RidgeClassifier(
            random_state=config["random_state"]
        ),
        "SVC": SVC(probability=True, random_state=config["random_state"]),
        "Linear SVC": LinearSVC(
            random_state=config["random_state"], max_iter=10000
        ),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=config["random_state"]
        ),
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=config["random_state"], verbosity=0
        )

    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, random_state=config["random_state"], verbosity=-1
        )

    return models


def objective(trial, model_name, X, y, config):
    """Objective function for Optuna optimization."""
    if model_name in ["Random Forest", "RandomForest"]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }
        model = RandomForestClassifier(
            **params, random_state=config["random_state"]
        )
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
        }
        model = XGBClassifier(**params, random_state=config["random_state"])
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
        }
        model = LGBMClassifier(
            **params, random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scores = cross_val_score(
        model, X, y, cv=config["cv_folds"], scoring="accuracy"
    )
    return scores.mean()


def save_model_pipeline(preprocessor, model, filepath):
    """Save preprocessor and model as a pipeline."""
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", model)]
    )
    # Use joblib for better sklearn compatibility if available,
    # otherwise pickle
    SERIALIZER.dump(pipeline, filepath)
    logger.info(f"Pipeline saved to {filepath}")


def load_and_predict(
    pipeline_path: str, test_data: pd.DataFrame
) -> np.ndarray:
    """Load a saved pipeline and make predictions on test data.

    Args:
        pipeline_path: Path to the saved pipeline pickle file
        test_data: Test DataFrame with features

    Returns:
        Array of predictions
    """
    try:
        # Try loading with the primary serializer first (joblib if available)
        try:
            pipeline = SERIALIZER.load(pipeline_path)
            serializer_name = SERIALIZER.__name__
            logger.info(
                f"Pipeline loaded from {pipeline_path} using {serializer_name}"
            )
        except Exception:
            # Fallback to pickle for backward compatibility with existing files
            msg = (
                f"Failed to load with {SERIALIZER.__name__}, "
                f"trying pickle fallback..."
            )
            logger.warning(msg)
            with open(pipeline_path, "rb") as f:
                pipeline = pickle.load(f)
            logger.info(
                f"Pipeline loaded from {pipeline_path} using pickle fallback"
            )

        # Verify it has a predict method
        if not hasattr(pipeline, "predict"):
            msg = (
                f"Loaded object is not a valid model/pipeline "
                f"(type: {type(pipeline)}). "
                f"It lacks a 'predict' method."
            )
            raise TypeError(msg)

        # Make predictions
        predictions = pipeline.predict(test_data)

        logger.info(f"Predictions generated for {len(test_data)} samples")

        return predictions

    except Exception as e:
        logger.error(f"Error loading pipeline or making predictions: {e}")
        raise
