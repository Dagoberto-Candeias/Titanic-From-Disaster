"""
Preprocessing module for Titanic ML Pipeline.
Contains functions for data preprocessing, feature engineering, and imputation.
"""

import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from typing import List
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = None

def preprocess_data(train, test, feature_cols, apply_smote=False, config=None):
    """Centralized data preprocessing function. Returns processed data and the fitted preprocessor."""
    if config is None:
        config = {}
    
    # Ensure we only use columns that exist in both train and test
    feature_cols = [
        col for col in feature_cols if col in train.columns and col in test.columns
    ]
    
    # Identify categorical columns (object type or with few unique values)
    categorical_features = []
    numerical_features = []
    
    for col in feature_cols:
        # Check if column is object type or has few unique values
        if (train[col].dtype == 'object' or 
            train[col].nunique() < 10 or 
            col in ["Sex", "Embarked", "Title_Group", "Deck", "TicketPrefix"] or
            col.startswith('feat_') and ('Bin' in col or 'Category' in col or 'missing' in col)):
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {numerical_features}")

    if config.get("polynomial_features", False):
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True)),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train = train[feature_cols].copy()
    y_train = train["Survived"].copy()
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test = test[feature_cols].copy()
    X_test_processed = preprocessor.transform(X_test)

    if apply_smote and IMBLEARN_AVAILABLE:
        logger.info("🔄 APLICANDO SMOTE PARA BALANCEAMENTO...")
        smote = SMOTE(
            random_state=config["random_state"], k_neighbors=config.get("smote_k", 5)
        )
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        logger.info(
            f"   ✅ SMOTE aplicado: {len(y_train)} amostras após balanceamento (de {len(train)} para {len(y_train)})"
        )
    elif apply_smote and not IMBLEARN_AVAILABLE:
        logger.warning(
            "   ⚠️  SMOTE solicitado mas imblearn não disponível - pulando balanceamento"
        )

    return X_train_processed, X_test_processed, y_train, preprocessor

def kfold_target_encode(df: pd.DataFrame, feature: str, target: str, suffix: str = "_te", n_splits: int = 5) -> pd.Series:
    """Perform K-fold target encoding on a categorical feature."""
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_feature = pd.Series(index=df.index, dtype=float)

    for train_idx, val_idx in skf.split(df, df[target]):
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # Calculate mean target for each category in train set
        means = train_data.groupby(feature)[target].mean()
        prior = train_data[target].mean()  # Global mean as prior

        # Map to validation set, using prior for unseen categories
        encoded_vals = val_data[feature].map(means).fillna(prior)
        encoded_feature.iloc[val_idx] = encoded_vals

    # For consistency, also encode the entire dataset using global means (for test set)
    global_means = df.groupby(feature)[target].mean()
    global_prior = df[target].mean()
    encoded_feature = encoded_feature.fillna(df[feature].map(global_means).fillna(global_prior))

    encoded_feature.name = f"{feature}{suffix}"
    return encoded_feature

def parallel_feature_engineering(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Apply parallel feature engineering operations."""
    df = df.copy()

    # Define operations that can be parallelized
    operations = [
        lambda d: d.assign(AgeClass=d["Age"] * d["Pclass"]),
        lambda d: d.assign(FarePerPerson=d["Fare"] / (d["SibSp"] + d["Parch"] + 1).replace(0, 1)),
        lambda d: d.assign(Title_Interactions=d["Title_Group"] + "_" + d["Sex"]),
    ]

    # Execute in parallel if multiple operations
    if len(operations) > 1:
        with ProcessPoolExecutor(max_workers=min(len(operations), multiprocessing.cpu_count())) as executor:
            futures = [executor.submit(op, df) for op in operations]
            results = [f.result() for f in futures]
            for res in results:
                df = df.assign(**{k: v for k, v in res.items() if k not in df.columns})
    else:
        for op in operations:
            df = op(df)

    return df

def advanced_missing_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Perform advanced missing imputation."""
    df = df.copy()

    # Age imputation based on Title_Group and Pclass
    if "Age" in df.columns and df["Age"].isnull().any():
        age_medians = df.groupby(["Title_Group", "Pclass"])["Age"].median()
        df["Age"] = df.apply(
            lambda row: age_medians.get((row["Title_Group"], row["Pclass"]), df["Age"].median())
            if pd.isnull(row["Age"]) else row["Age"],
            axis=1
        )

    # Fare imputation based on Pclass and Embarked
    if "Fare" in df.columns and df["Fare"].isnull().any():
        fare_medians = df.groupby(["Pclass", "Embarked"])["Fare"].median()
        df["Fare"] = df.apply(
            lambda row: fare_medians.get((row["Pclass"], row["Embarked"]), df["Fare"].median())
            if pd.isnull(row["Fare"]) else row["Fare"],
            axis=1
        )

    # Embarked mode
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    return df

def create_feature_pipeline(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Create a comprehensive feature pipeline."""
    df = df.copy()

    # Basic feature creation
    df = parallel_feature_engineering(df, is_training)

    # Bins and categories
    df["feat_AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young", "Adult", "Senior"],
    ).astype(str)
    df["feat_FareBin"] = pd.cut(
        df["Fare"],
        bins=[-1, 7.91, 14.45, 31, 1000],
        labels=["Low", "Medium", "High", "Luxury"],
    ).astype(str)
    df["feat_AgeCategory_v2"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 60, 100],
        labels=["Minor", "YoungAdult", "Adult", "Senior"],
    ).astype(str)
    df["feat_FareCategory_v2"] = pd.cut(
        df["Fare"],
        bins=[-1, 10, 50, 1000],
        labels=["Cheap", "Moderate", "Expensive"],
    ).astype(str)

    # Missing indicators
    df["feat_Age_missing"] = df["Age"].isnull().astype(int)
    df["feat_Cabin_missing"] = df["Cabin"].isnull().astype(int)
    df["feat_Embarked_missing"] = df["Embarked"].isnull().astype(int)
    df["feat_Fare_missing"] = df["Fare"].isnull().astype(int)

    # Target encoding if training
    if is_training and "Survived" in df.columns:
        df["feat_Title_Group_te"] = kfold_target_encode(df, "Title_Group", "Survived", suffix="_te")
        ticket_prefix_series = df["Ticket"].str[:3]
        ticket_prefix_series.name = "TicketPrefix"
        df["feat_TicketPrefix_te"] = kfold_target_encode(df, ticket_prefix_series, "Survived", suffix="_te")
        deck_series = df["Cabin"].str[0].fillna("U")
        deck_series.name = "Deck"
        df["feat_Deck_te"] = kfold_target_encode(df, deck_series, "Survived", suffix="_te")
        df["feat_Embarked_te"] = kfold_target_encode(df, "Embarked", "Survived", suffix="_te")
    else:
        # For test, use placeholders
        df["feat_Title_Group_te"] = 0.5
        df["feat_TicketPrefix_te"] = 0.5
        df["feat_Deck_te"] = 0.5
        df["feat_Embarked_te"] = 0.5

    return df

def build_feature_set(df: pd.DataFrame) -> List[str]:
    """Build the list of feature columns for modeling."""
    base_features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked",
        "Title_Group", "Deck", "TicketPrefix", "FamilySize", "IsAlone"
    ]
    engineered_features = [
        col for col in df.columns if col.startswith("feat_")
    ]
    feature_cols = [col for col in base_features + engineered_features if col in df.columns]
    return feature_cols
