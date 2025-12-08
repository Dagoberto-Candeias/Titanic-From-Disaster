"""
Modeling manager for Titanic ML Pipeline.
Enhanced with parallel processing, memory optimization, and performance monitoring.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Optional imports for optimization
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LGBM_AVAILABLE = False

from ..utils import ParallelProcessor

logger = logging.getLogger(__name__)


class ModelingManager:
    """Manages model training, optimization, and ensemble creation."""

    def __init__(self, config: Dict[str, Any], model_configs: Dict[str, Dict[str, Any]]):
        self.config = config
        self.model_configs = model_configs
        self.parallel_processor = ParallelProcessor(max_workers=config["parallel_jobs"])

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all configured models in parallel.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with model results
        """
        logger.info("   🔄 Training models in parallel...")

        # Get available models
        available_models = self._get_available_models()

        # Train models in parallel
        results = {}
        futures = {}

        with self.parallel_processor.executor as executor:
            for model_name, model_class in available_models.items():
                future = executor.submit(
                    self._train_single_model,
                    model_name,
                    model_class,
                    X_train,
                    y_train
                )
                futures[future] = model_name

            for future in self.parallel_processor.as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    logger.info(f"   ✅ {model_name} trained: {result['mean_score']:.4f}")
                except Exception as e:
                    logger.error(f"   ❌ {model_name} training failed: {e}")

        return results

    def _train_single_model(self, model_name: str, model_class, X_train: np.ndarray,
                          y_train: np.ndarray) -> Dict[str, Any]:
        """Train a single model with cross-validation and timeout protection."""
        try:
            # Create model instance with optimized parameters
            if model_name in self.model_configs:
                model_params = self.model_configs[model_name].copy()
                # Add performance optimizations for slow models
                if model_name in ["SVC", "KNeighbors"]:
                    model_params.update({
                        "max_iter": 10000 if model_name == "SVC" else None,
                        "n_jobs": -1 if model_name == "KNeighbors" else None
                    })
                model = model_class(**model_params)
            else:
                model = model_class(random_state=self.config["random_state"])

            # Timeout protection for slow models
            timeout_seconds = 300  # 5 minutes timeout
            if model_name in ["SVC", "KNeighbors", "MLPClassifier"]:
                timeout_seconds = 600  # 10 minutes for very slow models

            # Perform cross-validation with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future_cv = executor.submit(
                    cross_val_score,
                    model, X_train, y_train,
                    cv=self.config["cv_folds"],
                    scoring="accuracy",
                    n_jobs=1  # Avoid nested parallelism
                )
                try:
                    cv_scores = future_cv.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"   ⚠️  CV for {model_name} timed out after {timeout_seconds}s, skipping")
                    raise TimeoutError(f"CV timeout for {model_name}")

            # Train final model with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future_fit = executor.submit(model.fit, X_train, y_train)
                try:
                    future_fit.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"   ⚠️  Training for {model_name} timed out after {timeout_seconds}s, skipping")
                    raise TimeoutError(f"Training timeout for {model_name}")

            result = {
                "trained_model": model,
                "mean_score": cv_scores.mean(),
                "std_score": cv_scores.std(),
                "cv_scores": cv_scores.tolist(),
                "model_name": model_name
            }

            return result

        except Exception as e:
            logger.error(f"   ❌ Error training {model_name}: {e}")
            raise

    def create_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                        model_results: Dict[str, Any]) -> Dict[str, Any]:
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
            if "trained_model" in result:
                base_models.append((model_name, result["trained_model"]))

        if len(base_models) < 2:
            logger.warning("   ⚠️  Not enough models for ensemble")
            return ensemble_results

        try:
            # Voting Classifier
            voting_clf = VotingClassifier(
                estimators=base_models,
                voting="soft",
                n_jobs=1
            )

            voting_scores = cross_val_score(
                voting_clf, X_train, y_train,
                cv=self.config["cv_folds"],
                scoring="accuracy"
            )

            voting_clf.fit(X_train, y_train)

            ensemble_results["VotingEnsemble"] = {
                "trained_model": voting_clf,
                "mean_score": voting_scores.mean(),
                "std_score": voting_scores.std(),
                "cv_scores": voting_scores.tolist(),
                "model_name": "VotingEnsemble"
            }

            logger.info(f"   ✅ Voting Ensemble: {voting_scores.mean():.4f}")

            # Stacking Classifier
            meta_model = LogisticRegression(
                random_state=self.config["random_state"],
                max_iter=1000
            )

            stacking_clf = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=self.config["cv_folds"],
                n_jobs=1,
                passthrough=True
            )

            stacking_scores = cross_val_score(
                stacking_clf, X_train, y_train,
                cv=self.config["cv_folds"],
                scoring="accuracy"
            )

            stacking_clf.fit(X_train, y_train)

            ensemble_results["StackingEnsemble"] = {
                "trained_model": stacking_clf,
                "mean_score": stacking_scores.mean(),
                "std_score": stacking_scores.std(),
                "cv_scores": stacking_scores.tolist(),
                "model_name": "StackingEnsemble"
            }

            logger.info(f"   ✅ Stacking Ensemble: {stacking_scores.mean():.4f}")
        except Exception as e:
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
            "LinearDiscriminantAnalysis": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "QuadraticDiscriminantAnalysis": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        }

        # Check for optional libraries
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier
        except ImportError:
            logger.warning("   ⚠️  XGBoost not available")

        try:
            from lightgbm import LGBMClassifier
            models["LightGBM"] = LGBMClassifier
        except ImportError:
            logger.warning("   ⚠️  LightGBM not available")

        try:
            from catboost import CatBoostClassifier
            models["CatBoost"] = CatBoostClassifier
        except ImportError:
            logger.warning("   ⚠️  CatBoost not available")

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
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

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

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config["random_state"]
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
            random_state=config["random_state"], max_iter=1000
        ),
        "Ridge Classifier": RidgeClassifier(random_state=config["random_state"]),
        "SVC": SVC(probability=True, random_state=config["random_state"]),
        "Linear SVC": LinearSVC(
            random_state=config["random_state"], max_iter=10000
        ),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=config["random_state"]),
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
    if model_name == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }
        model = RandomForestClassifier(**params, random_state=config["random_state"])
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = XGBClassifier(**params, random_state=config["random_state"])
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = LGBMClassifier(**params, random_state=config["random_state"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scores = cross_val_score(model, X, y, cv=config["cv_folds"], scoring="accuracy")
    return scores.mean()


def save_model_pipeline(preprocessor, model, filepath):
    """Save preprocessor and model as a pipeline."""
    from sklearn.pipeline import Pipeline
    import pickle

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Pipeline saved to {filepath}")


def load_and_predict(pipeline_path: str, test_data: pd.DataFrame) -> np.ndarray:
    """
    Load a saved pipeline and make predictions on test data.
    
    Handles feature extraction and selection for compatibility with various models.

    Args:
        pipeline_path: Path to the saved pipeline pickle file
        test_data: Test DataFrame with all available features

    Returns:
        Array of predictions
    """
    import pickle

    try:
        # Load the pipeline
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        logger.info(f"Pipeline loaded from {pipeline_path}")

        # Attempt 1: Direct prediction (works if pipeline handles full data)
        try:
            predictions = pipeline.predict(test_data)
            logger.info(f"Predictions generated for {len(test_data)} samples")
            return predictions
        except Exception as e_direct:
            logger.debug(f"Direct prediction failed: {e_direct}")
            
            # Attempt 2: Try with common numeric features only
            feature_cols = ['Age', 'Fare', 'SibSp', 'Parch']
            available_features = [col for col in feature_cols if col in test_data.columns]
            
            if not available_features:
                # Attempt 3: Use all numeric columns
                available_features = test_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if not available_features:
                raise ValueError("No numeric features found in test data")
            
            logger.info(f"Using features: {available_features}")
            X_test = test_data[available_features].fillna(test_data[available_features].mean())
            
            predictions = pipeline.predict(X_test)
            logger.info(f"Predictions generated for {len(test_data)} samples using selected features")
            return predictions

    except Exception as e:
        logger.error(f"Error loading pipeline or making predictions: {e}")
        raise
