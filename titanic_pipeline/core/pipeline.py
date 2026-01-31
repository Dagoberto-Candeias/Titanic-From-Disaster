import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Handle imports for both script and module execution
try:
    # Try relative imports first (when run as module)
    from ..preprocessing import AdvancedFeatureEngineer
    from .preprocessing import preprocess_data as modular_preprocess_data
    from .modeling import ModelingManager
    from ..config import DEFAULT_CONFIG as CONFIG
    from ..utils import set_global_seeds
except ImportError:
    # Fallback to absolute imports (when run as script)
    import sys
    import os
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from titanic_pipeline.preprocessing import AdvancedFeatureEngineer
    from titanic_pipeline.core.preprocessing import preprocess_data as modular_preprocess_data
    from titanic_pipeline.core.modeling import ModelingManager
    from titanic_pipeline.config import DEFAULT_CONFIG as CONFIG
    from titanic_pipeline.utils import set_global_seeds

logger = logging.getLogger(__name__)


class TitanicPipeline:
    """
    Main pipeline class for Titanic ML workflow.

    Orchestrates the complete machine learning pipeline including
    data loading, preprocessing, feature engineering, model training,
    and evaluation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Titanic pipeline.

        Args:
            config: Configuration dictionary. If None, uses DEFAULT_CONFIG.
            config_override: Configuration overrides to merge with config.
        """
        self.config = config or CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        self.modeling_manager = ModelingManager(self.config)
        self.is_fitted = False
        logger.info("TitanicPipeline initialized")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'TitanicPipeline':
        """
        Fit the pipeline on training data.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Self for method chaining
        """
        logger.info("Fitting TitanicPipeline...")

        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Train models
        self.model_results = self.modeling_manager.train_all_models(X_train, y_train)

        # Create ensembles
        self.ensemble_results = self.modeling_manager.create_ensembles(
            X_train, y_train, self.model_results
        )

        self.is_fitted = True
        logger.info("TitanicPipeline fitted successfully")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            X_test: Test features

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")

        # Convert to numpy if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Use the best performing model (VotingEnsemble if available, else best individual)
        if 'VotingEnsemble' in self.ensemble_results:
            best_model = self.ensemble_results['VotingEnsemble']['trained_model']
        else:
            # Find best individual model
            best_score = 0
            best_model = None
            for model_name, result in self.model_results.items():
                if result.get('mean_score', 0) > best_score:
                    best_score = result['mean_score']
                    best_model = result['trained_model']

        if best_model is None:
            raise ValueError("No trained models available")

        return best_model.predict(X_test)

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all trained models.

        Returns:
            Dictionary with model performance data
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting performance")

        performance = {}

        # Individual models
        for model_name, result in self.model_results.items():
            performance[model_name] = {
                'mean_score': result.get('mean_score', 0),
                'std_score': result.get('std_score', 0),
                'cv_scores': result.get('cv_scores', [])
            }

        # Ensembles
        for model_name, result in self.ensemble_results.items():
            performance[model_name] = {
                'mean_score': result.get('mean_score', 0),
                'std_score': result.get('std_score', 0),
                'cv_scores': result.get('cv_scores', [])
            }

        return performance

    def save_model(self, filepath: str) -> None:
        """
        Save the best trained model.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        # Find best model
        if 'VotingEnsemble' in self.ensemble_results:
            best_model = self.ensemble_results['VotingEnsemble']['trained_model']
        else:
            best_score = 0
            best_model = None
            for model_name, result in self.model_results.items():
                if result.get('mean_score', 0) > best_score:
                    best_score = result['mean_score']
                    best_model = result['trained_model']

        if best_model is None:
            raise ValueError("No trained models available to save")

        # Save using joblib or pickle
        try:
            import joblib
            joblib.dump(best_model, filepath)
        except ImportError:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(best_model, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> Any:
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            import joblib
            model = joblib.load(filepath)
        except ImportError:
            import pickle
            with open(filepath, 'rb') as f:
                model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model
