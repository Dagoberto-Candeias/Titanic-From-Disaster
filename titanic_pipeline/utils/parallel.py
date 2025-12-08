"""
Parallel processing utilities for Titanic ML Pipeline.
Enhanced with robust error handling and memory management.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Union, Iterator, Iterable, TypeVar
import pickle
import os
import gc
import psutil
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


@contextmanager
def memory_monitor_context(memory_limit_mb: int = 1000) -> Iterator[None]:
    """Context manager to monitor memory usage during parallel processing."""
    process = psutil.Process(os.getpid())

    try:
        yield
    finally:
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        if memory_usage > memory_limit_mb:
            logger.warning(f"   💾 High memory usage detected: {memory_usage:.1f} MB")


def safe_parallel_map(
    func: Callable[[Any], T],
    items: Iterable[Any],
    max_workers: Optional[int] = None,
    use_threads: bool = False,
    chunk_size: int = 1,
    timeout: Optional[float] = None,
    memory_limit_mb: int = 1000,
    error_handling: str = "raise"
) -> List[T]:
    """Safe parallel mapping with robust error handling and memory management."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(items) if hasattr(items, '__len__') else 10)

    results = []

    try:
        with memory_monitor_context(memory_limit_mb):
            if use_threads:
                executor_class = ThreadPoolExecutor
            else:
                executor_class = ProcessPoolExecutor

            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(func, item): item for item in items
                }

                # Collect results as they complete
                for future in as_completed(future_to_item, timeout=timeout):
                    item = future_to_item[future]
                    try:
                        result = future.result(timeout=timeout if timeout else None)
                        results.append(result)
                    except Exception as exc:
                        if error_handling == "raise":
                            raise exc
                        elif error_handling == "log":
                            logger.error(f"   ❌ Error processing item {item}: {exc}")
                            results.append(None)  # or some default value
                        elif error_handling == "ignore":
                            results.append(None)  # or some default value
                        else:
                            raise ValueError(f"Invalid error_handling mode: {error_handling}")

    except Exception as e:
        logger.error(f"   ❌ Parallel processing failed: {e}")

        # Fallback to sequential processing if parallel fails
        logger.info("   🔄 Falling back to sequential processing...")
        results = [func(item) for item in items if error_handling != "ignore" or True]  # Adjust as needed

    return results


def is_tree_model(model: Any) -> bool:
    """Check if a model is a tree-based model."""
    tree_models = (
        "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost", "DecisionTree", "AdaBoost"
    )

    model_name = type(model).__name__

    return any(tree_name in model_name for tree_name in tree_models) or hasattr(model, 'feature_importances_')


def robust_pickle_dump(obj: Any, filepath: str, protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
    """Robust pickle dump with error handling."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol)
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to pickle dump to {filepath}: {e}")

        # Try with lower protocol if high protocol fails
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f, protocol=2)  # Lower protocol for compatibility
            logger.info(f"   ✅ Pickle dump succeeded with protocol 2 to {filepath}")

            return True
        except Exception as e2:
            logger.error(f"   ❌ Failed to pickle dump even with protocol 2: {e2}")

            return False


def robust_pickle_load(filepath: str) -> Optional[Any]:
    """Robust pickle load with error handling."""
    if not os.path.exists(filepath):
        logger.warning(f"   ⚠️  File not found: {filepath}")

        return None

    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        logger.error(f"   ❌ Failed to pickle load from {filepath}: {e}")

        return None


def parallel_predict_proba(models: List[Any], X: Any, max_workers: Optional[int] = None) -> List[Any]:
    """Parallel prediction of probabilities for multiple models."""
    def predict_single(model, X_data):
        try:
            return model.predict_proba(X_data) if hasattr(model, 'predict_proba') else model.predict(X_data)
        except Exception as e:
            logger.error(f"   ❌ Prediction failed for {type(model).__name__}: {e}")

            return None

    # Use partial to fix X parameter
    predict_func = partial(predict_single, X_data=X)

    return safe_parallel_map(predict_func, models, max_workers=max_workers, use_threads=True, error_handling="log")


def parallel_feature_importance(models: List[Any], feature_names: List[str], max_workers: Optional[int] = None) -> Dict[str, List[float]]:
    """Parallel computation of feature importances for multiple models."""
    def get_importance(model, feature_names_list):
        try:
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                return list(model.feature_importances_)
            elif hasattr(model, 'coef_') and model.coef_ is not None:
                # For linear models, use absolute coefficients as importance
                return list(abs(model.coef_.flatten()) if hasattr(model.coef_, 'flatten') else abs(model.coef_))
            else:
                logger.warning(f"   ⚠️  No feature importance available for {type(model).__name__}")

                return [0.0] * len(feature_names_list)  # Default to zero importance
        except Exception as e:
            logger.error(f"   ❌ Feature importance failed for {type(model).__name__}: {e}")

            return [0.0] * len(feature_names_list)  # Default to zero importance

    # Use partial to fix feature_names parameter
    importance_func = partial(get_importance, feature_names_list=feature_names)

    importances = safe_parallel_map(importance_func, models, max_workers=max_workers, use_threads=True, error_handling="log")

    return {"feature_names": feature_names, "importances": importances}


def parallel_cross_validate(model: Any, X: Any, y: Any, cv: int = 5, scoring: str = "accuracy", max_workers: Optional[int] = None) -> Dict[str, List[float]]:
    """Parallel cross-validation for a single model."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

    scorer_map = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average='weighted'),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True if hasattr(model, 'predict_proba') else False, average='weighted' if hasattr(model, 'predict_proba') else None)
    }

    scorer = scorer_map.get(scoring, make_scorer(accuracy_score))

    try:
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if hasattr(y, 'value_counts') and len(y.value_counts()) > 1 else cv

        scores = cross_val_score(model, X, y, cv=cv_obj, scoring=scorer, n_jobs=max_workers or -1, error_score='raise')

        return {"scores": list(scores), "mean": float(scores.mean()), "std": float(scores.std())}
    except Exception as e:
        logger.error(f"   ❌ Cross-validation failed: {e}")

        return {"scores": [], "mean": 0.0, "std": 0.0}


def parallel_model_training(models: List[Dict[str, Any]], X_train: Any, y_train: Any, X_val: Optional[Any] = None, y_val: Optional[Any] = None, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parallel training of multiple models with optional validation."""
    def train_single_model(model_config, X_train_data, y_train_data, X_val_data=None, y_val_data=None):
        try:
            model = model_config['model']

            # Train the model
            model.fit(X_train_data, y_train_data)

            # Optional validation
            val_score = None
            if X_val_data is not None and y_val_data is not None and hasattr(model, 'score'):
                val_score = model.score(X_val_data, y_val_data)

            return {
                "model_name": model_config['name'],
                "model": model,
                "training_success": True,
                "validation_score": val_score
            }
        except Exception as e:
            logger.error(f"   ❌ Training failed for {model_config['name']}: {e}")

            return {
                "model_name": model_config['name'],
                "model": None,
                "training_success": False,
                "error": str(e)
            }

    # Use partial to fix training data parameters
    train_func = partial(train_single_model, X_train_data=X_train, y_train_data=y_train, X_val_data=X_val, y_val_data=y_val)

    return safe_parallel_map(train_func, models, max_workers=max_workers, use_threads=False, error_handling="log")  # Use processes for training


def parallel_grid_search(model: Any, param_grid: Dict[str, List[Any]], X: Any, y: Any, cv: int = 5, scoring: str = "accuracy", max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Parallel grid search for hyperparameter tuning."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

    scorer_map = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average='weighted'),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True if hasattr(model, 'predict_proba') else False, average='weighted' if hasattr(model, 'predict_proba') else None)
    }

    scorer = scorer_map.get(scoring, make_scorer(accuracy_score))

    try:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scorer, n_jobs=max_workers or -1, error_score='raise', verbose=0)

        grid_search.fit(X, y)

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "best_estimator": grid_search.best_estimator_,
            "success": True
        }
    except Exception as e:
        logger.error(f"   ❌ Grid search failed: {e}")

        return {"success": False, "error": str(e)}


def parallel_random_search(model: Any, param_distributions: Dict[str, Any], X: Any, y: Any, n_iter: int = 10, cv: int = 5, scoring: str = "accuracy", random_state: int = 42, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Parallel randomized search for hyperparameter tuning."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score

    scorer_map = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average='weighted'),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True if hasattr(model, 'predict_proba') else False, average='weighted' if hasattr(model, 'predict_proba') else None)
    }

    scorer = scorer_map.get(scoring, make_scorer(accuracy_score))

    try:
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv, scoring=scorer, n_jobs=max_workers or -1, random_state=random_state, error_score='raise', verbose=0)

        random_search.fit(X, y)

        return {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "cv_results": random_search.cv_results_,
            "best_estimator": random_search.best_estimator_,
            "success": True
        }
    except Exception as e:
        logger.error(f"   ❌ Random search failed: {e}")

        return {"success": False, "error": str(e)}


def parallel_optuna_optimization(objective_func: Callable, n_trials: int = 100, timeout: Optional[float] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """Parallel Optuna optimization."""
    try:
        import optuna
    except ImportError:
        logger.error("   ❌ Optuna not installed. Install with: pip install optuna")
        return {"success": False, "error": "Optuna not installed"}

    try:
        # Create study with parallel optimization if max_workers > 1
        if max_workers and max_workers > 1:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())

            # Use Optuna's built-in parallel optimization (if available) or manual parallelization
            study.optimize(objective_func, n_trials=n_trials, timeout=timeout, n_jobs=max_workers or 1)
        else:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner())

            study.optimize(objective_func, n_trials=n_trials, timeout=timeout, n_jobs=1)  # Sequential for simplicity

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": study.trials_dataframe().to_dict() if hasattr(study, 'trials_dataframe') else None,
            "success": True
        }
    except Exception as e:
        logger.error(f"   ❌ Optuna optimization failed: {e}")

        return {"success": False, "error": str(e)}


def parallel_data_processing(func: Callable[[Any], T], data_chunks: List[Any], max_workers: Optional[int] = None, use_threads: bool = False) -> List[T]:
    """Parallel data processing with chunking support."""
    return safe_parallel_map(func, data_chunks, max_workers=max_workers, use_threads=use_threads, error_handling="log")


def get_optimal_workers(n_items: int, memory_per_item_mb: float = 10.0, max_memory_gb: float = 4.0) -> int:
    """Calculate optimal number of workers based on memory constraints."""
    available_memory_gb = psutil.virtual_memory().available / (1024**3)  # GB
    max_workers_by_memory = int(available_memory_gb / (memory_per_item_mb / 1024))  # Convert MB to GB

    max_workers_by_cpu = mp.cpu_count()

    optimal_workers = min(max_workers_by_memory, max_workers_by_cpu, n_items)  # Don't exceed number of items

    return max(1, optimal_workers)  # At least 1 worker


def parallel_pipeline_execution(pipeline_steps: List[Callable], data: Any, max_workers: Optional[int] = None) -> Any:
    """Execute pipeline steps in parallel where possible."""
    # For now, execute sequentially as pipeline steps often depend on each other
    # Future enhancement: identify independent steps and parallelize them

    result = data
    for step in pipeline_steps:
        try:
            result = step(result)
        except Exception as e:
            logger.error(f"   ❌ Pipeline step {step.__name__ if hasattr(step, '__name__') else 'unknown'} failed: {e}")

            raise e

    return result


    class ParallelProcessor:
        """Simple parallel processing wrapper using ProcessPoolExecutor."""

        def __init__(self, max_workers: Optional[int] = None):
            """Initialize the parallel processor."""
            if max_workers is None:
                max_workers = min(4, mp.cpu_count())
            self.max_workers = max_workers
            logger.debug(f"ParallelProcessor initialized with {max_workers} workers")

        def process(
            self,
            func: Callable[[Any], T],
            items: List[Any],
            error_handling: str = "raise",
            timeout: Optional[float] = None
        ) -> List[T]:
            """Process items in parallel."""
            return safe_parallel_map(
                func=func,
                items=items,
                max_workers=self.max_workers,
                use_threads=False,
                error_handling=error_handling,
                timeout=timeout
            )

        def process_with_threads(
            self,
            func: Callable[[Any], T],
            items: List[Any],
            error_handling: str = "raise",
            timeout: Optional[float] = None
        ) -> List[T]:
            """Process items in parallel using threads."""
            return safe_parallel_map(
                func=func,
                items=items,
                max_workers=self.max_workers,
                use_threads=True,
                error_handling=error_handling,
                timeout=timeout
            )
class ParallelProcessor:
    """Simple parallel processing wrapper using ProcessPoolExecutor."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the parallel processor."""
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        self.max_workers = max_workers
        logger.debug(f"ParallelProcessor initialized with {max_workers} workers")

    def process(
        self,
        func: Callable[[Any], T],
        items: List[Any],
        error_handling: str = "raise",
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process items in parallel."""
        return safe_parallel_map(
            func=func,
            items=items,
            max_workers=self.max_workers,
            use_threads=False,
            error_handling=error_handling,
            timeout=timeout
        )

    def process_with_threads(
        self,
        func: Callable[[Any], T],
        items: List[Any],
        error_handling: str = "raise",
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process items in parallel using threads."""
        return safe_parallel_map(
            func=func,
            items=items,
            max_workers=self.max_workers,
            use_threads=True,
            error_handling=error_handling,
            timeout=timeout
        )
