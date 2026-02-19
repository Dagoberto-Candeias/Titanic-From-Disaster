import logging
import os
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
        # Ensure output directories exist (do not create them at import time)
        os.makedirs("output/reports", exist_ok=True)
        os.makedirs("output/cache", exist_ok=True)
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

        # Convert to numpy arrays if needed (use .to_numpy() instead of .values)
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy(dtype=np.float64, na_value=np.nan)
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy(dtype=np.int64)

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

        # Convert to numpy if needed (use .to_numpy() instead of .values)
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy(dtype=np.float64, na_value=np.nan)

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

    def run_pipeline(
        self,
        train_path: str,
        test_path: str,
        target_col: str = "Survived",
    ) -> Dict[str, Any]:
        """Run complete pipeline from data loading to model training.

        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
            target_col: Name of target column

        Returns:
            Dictionary with pipeline results including models and metrics
        """
        logger.info("Starting TitanicPipeline.run_pipeline()...")

        try:
            # 1. Load data
            logger.info(f"Loading training data from {train_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(
                f"Loaded: {train_df.shape[0]} training samples, "
                f"{test_df.shape[0]} test samples"
            )

            # 2. Separate target from features
            if target_col not in train_df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found in training data"
                )

            y_train = train_df[target_col]
            X_train = train_df.drop(columns=[target_col])

            # Align test features with training features
            X_test = test_df[[col for col in X_train.columns
                             if col in test_df.columns]]

            logger.info(
                f"Features shape: X_train={X_train.shape}, "
                f"X_test={X_test.shape}"
            )

            # 3. Preprocess data
            logger.info("Preprocessing data...")
            X_train_processed, X_test_processed, _, _ = (
                modular_preprocess_data(
                    train_df,
                    test_df,
                    X_train.columns.tolist(),
                    apply_smote=self.config.get("enhanced_balance", False),
                )
            )

            logger.info(
                f"Preprocessed shapes: "
                f"X_train={X_train_processed.shape}, "
                f"X_test={X_test_processed.shape}"
            )

            # 4. Train models
            logger.info("Training models...")
            self.fit(X_train_processed, y_train)

            # 5. Make predictions
            logger.info("Making predictions on test set...")
            y_pred = self.predict(X_test_processed)

            # 6. Get performance metrics
            performance = self.get_model_performance()

            # 7. Compile results
            results = {
                "status": "success",
                "train_shape": X_train.shape,
                "test_shape": X_test.shape,
                "predictions": y_pred,
                "performance": performance,
                "best_model": (
                    "VotingEnsemble"
                    if "VotingEnsemble" in self.ensemble_results
                    else max(
                        self.model_results.items(),
                        key=lambda x: x[1].get("mean_score", 0),
                    )[0]
                ),
                "model_results": self.model_results,
                "ensemble_results": self.ensemble_results,
            }

            logger.info(
                f"Pipeline completed successfully. "
                f"Best model: {results['best_model']}"
            )
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "predictions": None,
                "performance": {},
            }
            with open(filepath, 'rb') as f:
                model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model
