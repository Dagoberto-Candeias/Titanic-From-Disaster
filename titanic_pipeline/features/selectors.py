"""
Feature selection utilities for Titanic ML Pipeline.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Manages feature selection using various methods."""

    def __init__(self, method: str = "kbest", k: int = 10, score_func=None):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('kbest', 'rf_importance', 'mutual_info')
            k: Number of features to select
            score_func: Scoring function for SelectKBest
        """
        self.method = method
        self.k = k
        self.score_func = score_func or f_classif
        self.selector = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the feature selector to the data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info(f"   🔍 Selecting features using {self.method} method...")

        if self.method == "kbest":
            self.selector = SelectKBest(score_func=self.score_func, k=self.k)
            self.selector.fit(X, y)
            scores = self.selector.scores_
            feature_names = X.columns
            selected_indices = self.selector.get_support(indices=True)
            self.selected_features = feature_names[selected_indices].tolist()
            logger.info(f"   ✅ Selected {len(self.selected_features)} features: {self.selected_features}")

        elif self.method == "rf_importance":
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            feature_names = X.columns
            indices = importances.argsort()[-self.k:][::-1]
            self.selected_features = feature_names[indices].tolist()
            logger.info(f"   ✅ Selected {len(self.selected_features)} features: {self.selected_features}")

        elif self.method == "mutual_info":
            mi_scores = mutual_info_classif(X, y, random_state=42)
            feature_names = X.columns
            indices = mi_scores.argsort()[-self.k:][::-1]
            self.selected_features = feature_names[indices].tolist()
            logger.info(f"   ✅ Selected {len(self.selected_features)} features: {self.selected_features}")

        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted selector.

        Args:
            X: Feature matrix

        Returns:
            Selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        return X[self.selected_features]

    def get_support(self, indices: bool = False) -> List[int] or List[str]:
        """
        Get selected feature indices or names.

        Args:
            indices: If True, return indices; else return names

        Returns:
            List of indices or feature names
        """
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if indices:
            return [list(X.columns).index(f) for f in self.selected_features]
        return self.selected_features

    def get_scores(self) -> Dict[str, Any]:
        """
        Get feature scores from the selector.

        Returns:
            Dictionary with scores
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if hasattr(self.selector, 'scores_'):
            return dict(zip(X.columns, self.selector.scores_))
        return {}
