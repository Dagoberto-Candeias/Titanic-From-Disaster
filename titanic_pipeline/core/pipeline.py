"""
Main pipeline orchestrator for Titanic ML Pipeline.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..config import ConfigManager
from ..utils import CacheManager, DataValidator
from ..features import FeatureEngineer, FeatureSelector
from .modeling import ModelingManager
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
            enabled=self.config.get("cache_enabled", True)
        )

        self.data_validator = DataValidator(self.config_manager.expected_schemas)
        self.feature_engineer = FeatureEngineer(self.config, self.cache_manager)
        self.feature_selector = FeatureSelector(self.config)
        self.modeling_manager = ModelingManager(self.config, self.config_manager.model_configs)
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

            # Step 1: Load and validate data
            train_df, test_df = self._load_and_validate_data(train_path, test_path)

            # Step 2: Feature engineering
            X_train, y_train, feature_cols = self.feature_engineer.engineer_features(train_df, is_training=True)
            X_test, _, _ = self.feature_engineer.engineer_features(test_df, is_training=False)

            # Step 3: Feature selection
            X_train_selected, selected_features = self.feature_selector.select_features(
                X_train, y_train, feature_cols
            )
            X_test_selected, _ = self.feature_selector.select_features(
                X_test, None, feature_cols
            )

            # Step 4: Train models
            model_results = self.modeling_manager.train_all_models(X_train_selected, y_train)

            # Step 4.5: Apply calibration if enabled
            if self.config.get("calibration_enabled", True):
                model_results = self.modeling_manager._calibrate_models(model_results, X_train_selected, y_train)

            # Step 5: Create ensembles
            ensemble_results = self.modeling_manager.create_ensembles(X_train_selected, y_train, model_results)

            # Combine all results
            all_results = {**model_results, **ensemble_results}

            # Step 6: Generate predictions
            predictions = self._generate_predictions(all_results, X_test_selected)

            # Step 7: Generate reports
            self.reporting_manager.generate_reports(all_results, selected_features, X_train_selected, y_train)

            # Step 8: Feature importance analysis
            importance_report = self.feature_selector.get_feature_importance_report(all_results, selected_features)

            # Step 9: Validate feature stability
            stability_report = self.feature_selector.validate_feature_stability(
                X_train_selected, X_test_selected, selected_features
            )

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
                    "selected_features": len(selected_features)
                }
            }

            logger.info(f"   ⏱️  Pipeline completed in {pipeline_results['execution_time']:.2f}s")
            logger.info(f"   📊 Best model: {self._get_best_model_name(all_results)}")

            return pipeline_results

        except Exception as e:
            logger.error(f"   ❌ Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def _load_and_validate_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate training and test data."""
        logger.info("   📂 Loading and validating data...")

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"   ✅ Data loaded: {len(train_df)} train, {len(test_df)} test samples")

        # Validate schemas
        self.data_validator.validate_data_schema(
            train_df, list(self.config_manager.train_schema.keys()),
            "train", "output/relatorios"
        )
        self.data_validator.validate_data_schema(
            test_df, list(self.config_manager.test_schema.keys()),
            "test", "output/relatorios"
        )

        # Basic data integrity checks
        self.data_validator.validate_data_integrity(train_df, "train")
        self.data_validator.validate_data_integrity(test_df, "test")

        return train_df, test_df

    def _generate_predictions(self, model_results: Dict[str, Any], X_test: np.ndarray) -> Dict[str, Any]:
        """Generate predictions for all models."""
        logger.info("   🔮 Generating predictions...")

        predictions = {}
        test_ids = None  # We'll need to extract this from test data

        for model_name, result in model_results.items():
            try:
                model = result.get("trained_model")
                if model is not None:
                    if hasattr(model, "predict_proba"):
                        pred_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        pred_proba = model.predict(X_test).astype(float)

                    predictions[model_name] = {
                        "probabilities": pred_proba,
                        "predictions": (pred_proba > 0.5).astype(int)
                    }

            except Exception as e:
                logger.error(f"   ❌ Prediction failed for {model_name}: {e}")

        return predictions

    def _get_best_model_name(self, model_results: Dict[str, Any]) -> str:
        """Get the name of the best performing model."""
        if not model_results:
            return "None"

        best_model = max(model_results.items(),
                        key=lambda x: x[1].get("mean_score", 0))
        return best_model[0]

    def get_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable pipeline summary."""
        if not results.get("success", False):
            return f"Pipeline failed: {results.get('error', 'Unknown error')}"

        summary = []
        summary.append("🎯 Titanic ML Pipeline Summary")
        summary.append("=" * 40)
        summary.append(f"📊 Best Model: {self._get_best_model_name(results.get('model_results', {}))}")
        summary.append(f"🔍 Features Selected: {len(results.get('selected_features', []))}")
        summary.append(f"📈 Models Trained: {len(results.get('model_results', {}))}")
        summary.append(f"📋 Train Samples: {results.get('data_info', {}).get('train_samples', 0)}")
        summary.append(f"🧪 Test Samples: {results.get('data_info', {}).get('test_samples', 0)}")

        return "\n".join(summary)
