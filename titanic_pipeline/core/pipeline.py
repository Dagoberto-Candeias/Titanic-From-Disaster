import logging
import time
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..config import ConfigManager
from ..utils import CacheManager, DataValidator
from ..features import FeatureEngineer, FeatureSelector
from .modeling import ModelingManager
from .preprocessing import preprocess_data as core_preprocess_data
from .reporting import ReportingManager

logger = logging.getLogger(__name__)


class TitanicPipeline:
    """Main pipeline class that orchestrates the entire ML workflow."""

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the Titanic ML Pipeline.

        Args:
            config_override: Optional configuration overrides
        """
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()

        # Apply overrides
        if config_override:
            self.config["config"].update(config_override)

        self.config = self.config["config"]  # Flatten for easier access

        # Initialize components
        self.cache_manager = CacheManager(
            cache_dir=self.config.get("cache_dir", "output/cache"),
            enabled=self.config.get("cache_enabled", True),
        )

        # Construct expected schemas dictionary from config manager
        expected_schemas = {
            "train": self.config_manager.train_schema,
            "test": self.config_manager.test_schema,
        }
        self.data_validator = DataValidator(expected_schemas)
        self.feature_engineer = FeatureEngineer(self.config, self.cache_manager)
        self.feature_selector = FeatureSelector(self.config)
        self.modeling_manager = ModelingManager(
            self.config, getattr(self.config_manager, "model_configs", {})
        )
        self.reporting_manager = ReportingManager(self.config)

        logger.info("   🚀 Titanic ML Pipeline initialized")
        logger.info(self.config_manager.get_config_summary())

    def run_pipeline(self, train_path: str, test_path: str) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.

        Args:
            train_path: Path to training data
            test_path: Path to test data

        Returns:
            Pipeline results dictionary
        """
        start_time = time.time()

        try:
            logger.info("   🎯 Starting Titanic ML Pipeline...")

            # Ensure output directory is writable
            output_dir = "output"
            if os.path.exists(output_dir):
                if not os.access(output_dir, os.W_OK):
                    raise PermissionError(
                        f"Output directory '{output_dir}' is not writable."
                    )
            else:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    raise PermissionError(
                        f"Cannot create output directory '{output_dir}': {e}"
                    )

            # Step 1: Load and validate data
            train_df, test_df = self._load_and_validate_data(train_path, test_path)

            # Step 2: Feature engineering
            X_train, y_train, feature_cols = self.feature_engineer.engineer_features(
                train_df, is_training=True
            )
            X_test, _, feature_cols_test = self.feature_engineer.engineer_features(
                test_df, is_training=False
            )

            # Ensure X_train and X_test are DataFrames
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=feature_cols)

            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=feature_cols_test)

            # Combine X_train and y_train for preprocessing that might need the target
            train_df_engineered = X_train.copy()
            train_df_engineered["Survived"] = (
                y_train.values if hasattr(y_train, "values") else y_train
            )

            # Step 3: Feature selection
            selected_features = feature_cols  # Default to all features
            if hasattr(self.feature_selector, "select_features"):
                try:
                    # Attempt to select features if implemented
                    _, selected_features = self.feature_selector.select_features(
                        X_train, y_train, feature_cols
                    )
                except Exception as e:
                    logger.warning(
                        f"   ⚠️  Feature selection failed: {e}. Using all features."
                    )
            else:
                logger.warning(
                    "   ⚠️  FeatureSelector.select_features not found. Skipping feature selection."
                )

            # Step 3.5: Preprocess data for modeling
            logger.info("   ⚙️  Preprocessing data for modeling...")
            X_train_processed, X_test_processed, y_train_processed, preprocessor = (
                core_preprocess_data(
                    train_df_engineered,
                    X_test,
                    selected_features,
                    apply_smote=self.config.get("enhanced_balance", False),
                    config=self.config,
                )
            )
            # Step 4: Train models
            model_results = self.modeling_manager.train_all_models(
                X_train_processed, y_train_processed
            )

            # Step 5: Create ensembles
            ensemble_results = self.modeling_manager.create_ensembles(
                X_train_processed, y_train_processed, model_results
            )

            # Combine all results
            all_results = {**model_results, **ensemble_results}

            # Step 6: Generate predictions
            predictions = self._generate_predictions(all_results, X_test_processed)

            # Step 7: Generate reports
            self.reporting_manager.generate_reports(
                all_results, selected_features, X_train_processed, y_train_processed
            )

            # Step 8: Feature importance analysis
            if hasattr(self.feature_selector, "get_feature_importance_report"):
                importance_report = self.feature_selector.get_feature_importance_report(
                    all_results, selected_features
                )
            else:
                importance_report = {}

            # Step 9: Validate feature stability
            if hasattr(self.feature_selector, "validate_feature_stability"):
                stability_report = self.feature_selector.validate_feature_stability(
                    X_train_processed, X_test_processed, selected_features
                )
            else:
                stability_report = {}

            # Compile final results
            pipeline_results = {
                "success": True,
                "execution_time": time.time() - start_time,
                "model_results": all_results,
                "predictions": predictions,
                "selected_features": selected_features,
                "feature_importance": importance_report,
                "feature_stability": stability_report,
                "config": self.config,
                "data_info": {
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "original_features": len(feature_cols),
                    "selected_features": len(selected_features),
                },
            }

            logger.info(
                f"   ⏱️  Pipeline completed in {pipeline_results['execution_time']:.2f}s"
            )
            logger.info(f"   📊 Best model: {self._get_best_model_name(all_results)}")

            return pipeline_results

        except Exception as e:
            logger.error(f"   ❌ Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _load_and_validate_data(
        self, train_path: str, test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate training and test data."""
        logger.info("   📂 Loading and validating data...")

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(
            f"   ✅ Data loaded: {len(train_df)} train, {len(test_df)} test samples"
        )

        # Validate schemas
        self.data_validator.validate_data_schema(
            train_df,
            list(self.config_manager.train_schema.keys()),
            "train",
            "output/relatorios",
        )
        self.data_validator.validate_data_schema(
            test_df,
            list(self.config_manager.test_schema.keys()),
            "test",
            "output/relatorios",
        )

        # Basic data integrity checks
        self.data_validator.validate_data_integrity(train_df, "train")
        self.data_validator.validate_data_integrity(test_df, "test")

        return train_df, test_df

    def _generate_predictions(
        self, model_results: Dict[str, Any], X_test: np.ndarray
    ) -> Dict[str, Any]:
        """Generate predictions for all models."""
        logger.info("   🔮 Generating predictions...")

        predictions = {}
        test_ids = None  # We'll need to extract this from test data

        for model_name, result in model_results.items():
            try:
                model = result.get("trained_model")
                if model is not None:
                    # Handle dense requirement for specific models
                    X_pred = X_test
                    dense_required_models = [
                        "GaussianNB",
                        "LinearDiscriminantAnalysis",
                        "QuadraticDiscriminantAnalysis",
                        "VotingEnsemble",
                        "StackingEnsemble",
                    ]
                    if model_name in dense_required_models:
                        if hasattr(X_test, "toarray"):
                            X_pred = X_test.toarray()

                    if hasattr(model, "predict_proba"):
                        pred_proba = model.predict_proba(X_pred)[:, 1]
                    else:
                        pred_proba = model.predict(X_pred).astype(float)

                    predictions[model_name] = {
                        "probabilities": pred_proba,
                        "predictions": (pred_proba > 0.5).astype(int),
                    }

            except Exception as e:
                logger.error(f"   ❌ Prediction failed for {model_name}: {e}")

        return predictions

    def _get_best_model_name(self, model_results: Dict[str, Any]) -> str:
        """Get the name of the best performing model."""
        if not model_results:
            return "None"

        best_model = max(model_results.items(), key=lambda x: x[1].get("mean_score", 0))
        return best_model[0]

    def get_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable pipeline summary."""
        if not results.get("success", False):
            return f"Pipeline failed: {results.get('error', 'Unknown error')}"

        summary = []
        summary.append("🎯 Titanic ML Pipeline Summary")
        summary.append("=" * 40)
        summary.append(
            f"📊 Best Model: {self._get_best_model_name(results.get('model_results', {}))}"
        )
        summary.append(
            f"🔍 Features Selected: {len(results.get('selected_features', []))}"
        )
        summary.append(f"📈 Models Trained: {len(results.get('model_results', {}))}")
        summary.append(
            f"📋 Train Samples: {results.get('data_info', {}).get('train_samples', 0)}"
        )
        summary.append(
            f"🧪 Test Samples: {results.get('data_info', {}).get('test_samples', 0)}"
        )

        return "\n".join(summary)
