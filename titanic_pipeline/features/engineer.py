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

    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
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

            # Target encoding if enabled and training
            if self.config.get("target_encoding", True) and is_training and y is not None:
                data = self._target_encoding(data, y)

            # Handle missing values
            data = self._handle_missing_values(data)

            # Select final features
            feature_cols = [col for col in data.columns if col != "PassengerId"]
            X = data[feature_cols].values

            logger.info(f"   ✅ Features engineered: {len(feature_cols)} features, {X.shape[0]} samples")

            return X, y, feature_cols

        except Exception as e:
            logger.error(f"   ❌ Feature engineering failed: {e}")
            raise

    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features."""
        data = df.copy()

        # Title extraction from Name
        data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_mapping = {
            "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
            "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
            "Mlle": "Miss", "Countess": "Royalty", "Ms": "Mrs", "Lady": "Royalty",
            "Jonkheer": "Royalty", "Don": "Royalty", "Dona": "Royalty", "Mme": "Mrs",
            "Capt": "Officer", "Sir": "Royalty"
        }
        data["Title"] = data["Title"].map(title_mapping).fillna("Other")

        # Family size
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

        # Is alone
        data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

        # Age groups
        data["AgeGroup"] = pd.cut(data["Age"], bins=[0, 12, 18, 35, 60, 100],
                                labels=["Child", "Teen", "Young", "Adult", "Senior"])

        # Fare groups
        data["FareGroup"] = pd.qcut(data["Fare"].fillna(data["Fare"].median()),
                                  q=4, labels=["Low", "Medium", "High", "VeryHigh"])

        # Cabin deck
        data["CabinDeck"] = data["Cabin"].str[0].fillna("Unknown")

        # Ticket frequency
        data["TicketFreq"] = data.groupby("Ticket")["Ticket"].transform("count")

        return data

    def _advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced interaction features."""
        data = df.copy()

        # Age * Pclass interaction
        data["Age*Pclass"] = data["Age"] * data["Pclass"]

        # Fare per person
        data["FarePerPerson"] = data["Fare"] / data["FamilySize"]

        # Title and Pclass interaction
        data["Title*Pclass"] = data["Title"] + "_" + data["Pclass"].astype(str)

        # Age and Sex interaction
        data["Age*Sex"] = data["Age"] * (data["Sex"] == "male").astype(int)

        # Additional advanced interactions
        if self.config.get("advanced_interactions", True):
            # Title and FamilySize interaction
            data["Title*FamilySize"] = data["Title"] + "_" + data["FamilySize"].astype(str)

            # Pclass and Embarked interaction
            data["Pclass*Embarked"] = data["Pclass"].astype(str) + "_" + data["Embarked"]

            # Fare and Pclass interaction
            data["Fare*Pclass"] = data["Fare"] * data["Pclass"]

            # Age and Fare interaction
            data["Age*Fare"] = data["Age"] * data["Fare"]

            # Sex and Embarked interaction
            data["Sex*Embarked"] = data["Sex"] + "_" + data["Embarked"]

            # Nova feature: FamilySize * Pclass (interação classe-família)
            data["FamilySize*Pclass"] = data["FamilySize"] * data["Pclass"]

        # Family size categories
        data["FamilySizeCat"] = pd.cut(data["FamilySize"], bins=[0, 1, 4, 20],
                                     labels=["Alone", "Small", "Large"])

        return data

    def _polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        data = df.copy()

        # Age squared
        data["Age^2"] = data["Age"] ** 2

        # Fare squared
        data["Fare^2"] = data["Fare"] ** 2

        # Age * Fare
        data["Age*Fare"] = data["Age"] * data["Fare"]

        return data

    def _target_encoding(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Apply target encoding to categorical features."""
        data = df.copy()

        # Create target encoding for Title
        title_means = pd.DataFrame({"Title": data["Title"], "Target": y}).groupby("Title")["Target"].mean()
        data["Title_encoded"] = data["Title"].map(title_means)

        # Create target encoding for CabinDeck
        deck_means = pd.DataFrame({"CabinDeck": data["CabinDeck"], "Target": y}).groupby("CabinDeck")["Target"].mean()
        data["CabinDeck_encoded"] = data["CabinDeck"].map(deck_means)

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
