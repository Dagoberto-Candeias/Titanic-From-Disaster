"""
Feature engineering for Titanic ML Pipeline.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from ..utils import CacheManager

logger = logging.getLogger(__name__)




class FeatureEngineer:
    """Handles feature engineering for Titanic dataset."""

    def __init__(self, config: Dict[str, Any], cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        # Store target encoding mappings for prediction consistency
        self.target_encoding_mappings = {}

    def engineer_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Engineer features from raw Titanic data.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (affects target extraction)

        Returns:
            Tuple of (X, y, feature_columns)
        """
        try:
            # Create a copy to avoid modifying original data
            data = df.copy()

            # Extract target if training
            y = None
            if is_training and "Survived" in data.columns:
                y = data["Survived"].values
                data = data.drop("Survived", axis=1)

            # Basic feature engineering
            data = self._basic_features(data)

            # Advanced features if enabled
            if self.config.get("advanced_interactions", True):
                data = self._advanced_features(data)

            # Polynomial features if enabled
            if self.config.get("polynomial_features", True):
                data = self._polynomial_features(data)

            # Target encoding if enabled
            if self.config.get("target_encoding", True):
                data = self._target_encoding(data, y)

            # Handle missing values
            data = self._handle_missing_values(data)

            # Select final features
            feature_cols = [col for col in data.columns if col != "PassengerId"]
            X = data[feature_cols].values

            logger.info(
                f"   ✅ Features engineered: {len(feature_cols)} features, {X.shape[0]} samples"
            )

            return X, y, feature_cols

        except Exception as e:
            logger.error(f"   ❌ Feature engineering failed: {e}")
            raise

    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features."""
        data = df.copy()

        # Title extraction from Name
        data["title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_mapping = {
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Dr": "Officer",
            "Rev": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Mlle": "Miss",
            "Countess": "Royalty",
            "Ms": "Mrs",
            "Lady": "Royalty",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Dona": "Royalty",
            "Mme": "Mrs",
            "Capt": "Officer",
            "Sir": "Royalty",
        }
        data["title"] = data["title"].map(title_mapping).fillna("Other")

        # Title Grouping (New Feature)
        title_group_mapping = {
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Officer": "Rare",
            "Royalty": "Rare",
            "Other": "Rare",
        }
        data["title_group"] = data["title"].map(title_group_mapping)

        # Family size
        data["family_size"] = data["SibSp"] + data["Parch"] + 1

        # Is alone
        data["is_alone"] = (data["family_size"] == 1).astype(int)

        # Is Child
        age_temp = data["Age"].fillna(data["Age"].median())
        data["is_child"] = (age_temp < 12).astype(int)

        # Age groups
        data["age_group"] = pd.cut(
            data["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teen", "Young", "Adult", "Senior"],
        )

        # Fare groups
        data["fare_group"] = pd.qcut(
            data["Fare"].fillna(data["Fare"].median()),
            q=4,
            labels=["Low", "Medium", "High", "VeryHigh"],
        )

        # Cabin deck
        data["cabin_deck"] = data["Cabin"].str[0].fillna("Unknown")

        # Ticket frequency
        data["ticket_freq"] = data.groupby("Ticket")["Ticket"].transform("count")

        return data

    def _advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced interaction features."""
        data = df.copy()

        # Age * Pclass interaction
        data["age_pclass_interaction"] = data["Age"] * data["Pclass"]

        # Fare per person
        data["fare_per_person"] = data["Fare"] / data["family_size"]

        # Title and Pclass interaction
        data["title_pclass_interaction"] = data["title"] + "_" + data["Pclass"].astype(str)

        # Age and Sex interaction
        data["age_sex_interaction"] = data["Age"] * (data["Sex"] == "male").astype(int)

        # Family size categories
        data["family_size_cat"] = pd.cut(
            data["family_size"], bins=[0, 1, 4, 20], labels=["Alone", "Small", "Large"]
        )

        return data

    def _polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        data = df.copy()

        # Age squared
        data["age_squared"] = data["Age"] ** 2

        # Fare squared
        data["fare_squared"] = data["Fare"] ** 2

        # Age * Fare
        data["age_fare_interaction"] = data["Age"] * data["Fare"]

        return data

    def _target_encoding(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Apply target encoding to categorical features."""
        data = df.copy()

        if y is not None:
            # Training: compute and store mappings
            title_means = (
                pd.DataFrame({"title": data["title"], "Target": y})
                .groupby("title")["Target"]
                .mean()
            )
            self.target_encoding_mappings["title"] = title_means.to_dict()

            deck_means = (
                pd.DataFrame({"cabin_deck": data["cabin_deck"], "Target": y})
                .groupby("cabin_deck")["Target"]
                .mean()
            )
            self.target_encoding_mappings["cabin_deck"] = deck_means.to_dict()

            # Apply encoding
            data["title_encoded"] = data["title"].map(title_means)
            data["cabin_deck_encoded"] = data["cabin_deck"].map(deck_means)
        else:
            # Prediction: use stored mappings or defaults
            if "title" in self.target_encoding_mappings:
                data["title_encoded"] = data["title"].map(self.target_encoding_mappings["title"])
            else:
                data["title_encoded"] = 0.5  # Default fallback

            if "cabin_deck" in self.target_encoding_mappings:
                data["cabin_deck_encoded"] = data["cabin_deck"].map(self.target_encoding_mappings["cabin_deck"])
            else:
                data["cabin_deck_encoded"] = 0.5  # Default fallback

            # Fill any NaN values with global mean
            data["title_encoded"] = data["title_encoded"].fillna(0.5)
            data["cabin_deck_encoded"] = data["cabin_deck_encoded"].fillna(0.5)

        return data

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        data = df.copy()

        # Age imputation
        if "Age" in data.columns:
            data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )

        # Fare imputation
        if "Fare" in data.columns:
            data["Fare"] = data["Fare"].fillna(data["Fare"].median())

        # Embarked imputation
        if "Embarked" in data.columns:
            data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

        # Cabin imputation
        if "Cabin" in data.columns:
            data["Cabin"] = data["Cabin"].fillna("Unknown")

        # Fill remaining missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        categorical_cols = data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        return data
