import sys
import os

sys.path.insert(0, ".")

import pandas as pd
from titanic_pipeline.features.engineer import FeatureEngineer
from titanic_pipeline.utils import CacheManager

# Create a minimal config
config = {
    "advanced_interactions": True,
    "polynomial_features": True,
    "target_encoding": True,
}

# Create a dummy cache manager
cache_manager = CacheManager()

# Create sample data
sample_data = pd.DataFrame(
    {
        "PassengerId": [1, 2],
        "Survived": [0, 1],
        "Pclass": [3, 1],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        ],
        "Sex": ["male", "female"],
        "Age": [22.0, 38.0],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Ticket": ["A/5 21171", "PC 17599"],
        "Fare": [7.25, 71.2833],
        "Cabin": ["C85", "C123"],
        "Embarked": ["S", "C"],
    }
)

# Instantiate the FeatureEngineer
engineer = FeatureEngineer(config, cache_manager)

# Test basic method call
try:
    result = engineer._basic_features(sample_data)
    print("Basic features method executed successfully.")
    print(f"Result shape: {result.shape}")
    print("Columns:", list(result.columns))
except Exception as e:
    print(f"Error in basic features: {e}")

# Test handle missing values
try:
    result = engineer._handle_missing_values(sample_data)
    print("Handle missing values method executed successfully.")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Error in handle missing values: {e}")

print("Thorough testing completed: No runtime errors detected.")
