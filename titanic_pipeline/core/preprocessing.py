"""
Preprocessing module for Titanic ML Pipeline.
Contains functions for data preprocessing, feature engineering, and imputation.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from typing import List, Union, Tuple, Optional, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Tenta importar configuração padrão do pacote
try:
    from ..config import DEFAULT_CONFIG as CONFIG
except ImportError:
    CONFIG = {"feature_selection": False, "random_state": 42, "polynomial_features": False}

from ..utils import set_global_seeds

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
        config = CONFIG
    
    # Ensure we only use columns that exist in both train and test
    feature_cols = [
        col for col in feature_cols if col in train.columns and col in test.columns
    ]
    
    # Identify categorical and numerical columns
    categorical_features = []
    numerical_features = []

    # Known categorical features (these should be treated as categorical even if they look numeric)
    known_categorical = [
        "Sex", "Embarked", "title", "title_group", "age_group", "fare_group",
        "cabin_deck", "family_size_cat", "Pclass", "SibSp", "Parch", "is_alone",
        "is_child", "ticket_freq", "Title", "Deck", "TicketPrefix", "Title_Group", "Surname"
    ]

    # Known numerical features (these should be treated as numerical)
    known_numerical = [
        "Age", "Fare", "family_size", "age_pclass_interaction", "fare_per_person",
        "age_sex_interaction", "age_squared", "fare_squared", "age_fare_interaction",
        "title_encoded", "cabin_deck_encoded", "FamilySize", "AgeClass", "FarePerPerson"
    ]

    for col in feature_cols:
        if col in known_numerical:
            numerical_features.append(col)
        elif (train[col].dtype == 'object' or
              col in known_categorical or
              col.startswith('feat_') and ('Bin' in col or 'Category' in col or 'missing' in col)):
            categorical_features.append(col)
        else:
            # Default to numerical for any remaining features
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
            random_state=config.get("random_state", 42), k_neighbors=config.get("smote_k", 5)
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

def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family size and is_alone features."""
    df = df.copy()
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from Name."""
    df = df.copy()
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\\.', expand=False)
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df["Title"] = df["Title"].replace(rare_titles, 'Rare')
        df["Title"] = df["Title"].replace(['Mlle', 'Ms'], 'Miss')
        df["Title"] = df["Title"].replace('Mme', 'Mrs')
        df["Title"] = df["Title"].fillna('Mr')
        # Create Title_Group for compatibility
        df["Title_Group"] = df["Title"]
    return df

def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck from Cabin."""
    df = df.copy()
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].str[0].fillna("U")
    else:
        df["Deck"] = "U"
    return df

def extract_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract prefix from Ticket."""
    df = df.copy()
    if "Ticket" in df.columns:
        df["TicketPrefix"] = df["Ticket"].apply(lambda x: x.split()[0] if not x.isdigit() else "UNKNOWN")
    else:
        df["TicketPrefix"] = "UNKNOWN"
    return df

def extract_surname(df: pd.DataFrame) -> pd.DataFrame:
    """Extract surname from Name."""
    df = df.copy()
    if "Name" in df.columns:
        df["Surname"] = df["Name"].apply(lambda x: x.split(',')[0].strip())
    return df

def kfold_target_encode(df: pd.DataFrame, feature: str, target: str, suffix: str = "_te", n_splits: int = 5) -> pd.Series:
    """Perform K-fold target encoding on a categorical feature."""
    logger = logging.getLogger(__name__)

    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_feature = pd.Series(index=df.index, dtype=float)

    # Handle feature argument: if it's a Series, use its name or values
    if isinstance(feature, pd.Series):
        col_name = feature.name
    else:
        col_name = feature

    if col_name not in df.columns:
        logger.warning(f"Feature {col_name} not found in DataFrame for target encoding. Returning a series of 0.0")
        return pd.Series(0.0, index=df.index, name=f"feat_{col_name}{suffix}")

    for train_idx, val_idx in skf.split(df, df[target]):
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # Calculate mean target for each category in train set
        try:
            means = train_data.groupby(col_name)[target].mean()
        except KeyError as e:
            logger.error(f"Error during groupby: {e}")
            raise

        #  Handle NaNs if present
        prior = train_data[target].mean()  # Global mean as prior

        # Map to validation set
        encoded_vals = val_data[col_name].map(means).fillna(prior)
        encoded_feature.iloc[val_idx] = encoded_vals

    # For consistency, also encode the entire dataset using global means (for test set)
    global_means = df.groupby(col_name)[target].mean()
    global_prior = df[target].mean()
    encoded_feature = encoded_feature.fillna(df[col_name].map(global_means).fillna(global_prior))

    # set encoded series name
    encoded_feature.name = f"{col_name}{suffix}"
    return encoded_feature

# Helper functions for parallel processing to avoid pickling issues with lambdas
def _create_age_class(d: pd.DataFrame) -> pd.DataFrame:
    if "Age" in d.columns and "Pclass" in d.columns:
        return d.assign(AgeClass=d["Age"] * d["Pclass"])
    return d

def _create_fare_per_person(d: pd.DataFrame) -> pd.DataFrame:
    if "Fare" in d.columns:
        # Use SibSp and Parch if available, else default to 1
        denom = 1
        if "SibSp" in d.columns and "Parch" in d.columns:
            denom = d["SibSp"] + d["Parch"] + 1
        return d.assign(FarePerPerson=d["Fare"] / denom)
    return d

def _create_title_interactions(d: pd.DataFrame) -> pd.DataFrame:
    if "Title_Group" in d.columns and "Sex" in d.columns:
        return d.assign(Title_Interactions=d["Title_Group"] + "_" + d["Sex"])
    return d

def parallel_feature_engineering(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Apply parallel feature engineering operations."""
    df = df.copy()

    # Define operations that can be parallelized using top-level functions
    operations = [
        _create_age_class,
        _create_fare_per_person,
        _create_title_interactions,
    ]

    # Execute in parallel if multiple operations AND enough data to justify overhead
    if len(operations) > 1 and len(df) > 100:
        try:
            with ProcessPoolExecutor(max_workers=min(len(operations), multiprocessing.cpu_count())) as executor:
                futures = [executor.submit(op, df) for op in operations]
                results = [f.result() for f in futures]
                for res in results:
                    # Merge results back into df
                    # Identify new columns in res that are not in df
                    new_cols = [c for c in res.columns if c not in df.columns]
                    if new_cols:
                        df[new_cols] = res[new_cols]
        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}. Falling back to sequential.")
            for op in operations:
                df = op(df)
    else:
        for op in operations:
            df = op(df)

    return df

def advanced_missing_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Perform advanced missing imputation."""
    df = df.copy()

    # Age imputation based on Title_Group and Pclass
    if "Age" in df.columns and df["Age"].isnull().any():
        # Ensure Title_Group exists if we are going to use it, otherwise fallback
        if "Title_Group" not in df.columns:
             df = extract_title(df)
             
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

    # Apply basic extractions first
    df = create_family_features(df)
    df = extract_title(df)
    df = extract_deck(df)
    df = extract_ticket_prefix(df)
    df = extract_surname(df)

    # Basic feature creation
    df = parallel_feature_engineering(df, is_training)

    # Bins and categories
    df["feat_AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 60, 100],
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
        bins=[-1, 7.91, 14.45, 31, 1000],
        labels=["Low", "Medium", "High", "Luxury"],
    ).astype(str)
    df["feat_FamilySizeCategory"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 7, 11],
        labels=["Single", "Small", "Average", "Large"],
    ).astype(str)

    # Interactions
    if "Age" in df.columns and "Pclass" in df.columns:
        df["AgeClass"] = df["Age"] * df["Pclass"]

    # Advanced imputations
    df = advanced_missing_imputation(df)

    return df
