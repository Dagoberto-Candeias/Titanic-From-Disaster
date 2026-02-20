import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline
from typing import Callable, List, Optional, Tuple, Union

# IterativeImputer is experimental in sklearn < 1.5, stable in 1.5+
try:
    from sklearn.impute import IterativeImputer
except ImportError:
    # sklearn < 1.5: need to enable experimental API
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

from titanic_pipeline.utils import set_global_seeds

from .config import CONFIG


logger = logging.getLogger(__name__)

# Note: do NOT create output directories at import time to avoid side-effects.
# Directory creation is handled by the pipeline initializer when needed.

def create_feature_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    numerical_imputer: str = 'simple',
    categorical_encoder: str = 'onehot',
    scaler: bool = True,
    knn_neighbors: int = 5,
    iterative_max_iter: int = 10,
    random_state: int = 42
) -> ColumnTransformer:
    """
    Cria um pipeline de pré-processamento completo para variáveis numéricas e
    categóricas. Detecta colunas numéricas e categóricas automaticamente e
    aplica imputação e codificação apropriadas.

    Creates a complete preprocessing pipeline for numerical and categorical
    variables. Automatically detects numerical and categorical columns and
    applies appropriate imputation and encoding.

    Parameters:
    -----------
    numerical_imputer : str, default='simple'
        Tipo de imputador para variáveis numéricas: 'simple'
        (SimpleImputer com média), 'knn' (KNNImputer), 'iterative'
        (IterativeImputer).
    categorical_encoder : str, default='onehot'
        Tipo de codificador para variáveis categóricas: 'onehot'
        (OneHotEncoder), 'ordinal' (OrdinalEncoder).
    scaler : bool, default=True
        Se deve aplicar StandardScaler às variáveis numéricas após
        imputação.
    knn_neighbors : int, default=5
        Número de vizinhos para KNNImputer.
    iterative_max_iter : int, default=10
        Número máximo de iterações para IterativeImputer.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Returns:
    --------
    ColumnTransformer
        Pipeline de pré-processamento configurado.

    Raises:
    -------
    ValueError
        Se o tipo de imputador ou codificador for inválido.
    """
    set_global_seeds(random_state)

    # Identificar colunas numéricas e categóricas
    numerical_features = []
    categorical_features = []

    for col in feature_cols:
        # Colunas sempre numéricas (mesmo com poucos valores únicos)
        always_numerical = ["Pclass", "SibSp", "Parch", "FamilySize", "IsAlone"]
        if col in always_numerical:
            numerical_features.append(col)
        # Colunas sempre categóricas
        elif (
            df[col].dtype == 'object' or
            col in ["Sex", "Embarked", "Title", "Title_Group", "Deck", "TicketPrefix"] or
            (
                col.startswith('feat_') and
                ('Bin' in col or 'Category' in col or 'missing' in col)
            )
        ):
            categorical_features.append(col)
        # Outras colunas: verificar se são numéricas ou têm poucos valores únicos
        elif df[col].dtype in ['int64', 'float64'] or df[col].nunique() >= 10:
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    logger.info(
        f"Features numéricas: {numerical_features}"
    )
    logger.info(
        f"Features categóricas: {categorical_features}"
    )

    # Imputadores numéricos
    if numerical_imputer == 'simple':
        num_imputer = SimpleImputer(strategy='mean')
    elif numerical_imputer == 'knn':
        try:
            num_imputer = KNNImputer(n_neighbors=knn_neighbors)
        except ImportError:
            logger.warning("KNNImputer não disponível. Usando SimpleImputer.")
            num_imputer = SimpleImputer(strategy='mean')
    elif numerical_imputer == 'iterative':
        try:
            num_imputer = IterativeImputer(
                max_iter=iterative_max_iter, random_state=random_state
            )
        except ImportError:
            logger.warning(
                "IterativeImputer não disponível. Usando SimpleImputer."
            )
            num_imputer = SimpleImputer(strategy='mean')
    else:
        raise ValueError(
            "numerical_imputer deve ser 'simple', 'knn' ou 'iterative'. "
            f"Recebido: {numerical_imputer}"
        )

    # Pipeline numérico
    num_pipeline = Pipeline([
        ('imputer', num_imputer)
    ])
    if scaler:
        num_pipeline.steps.append(('scaler', StandardScaler()))

    # Codificadores categóricos
    if categorical_encoder == 'onehot':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif categorical_encoder == 'ordinal':
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        raise ValueError(
            "categorical_encoder deve ser 'onehot' ou 'ordinal'. "
            f"Recebido: {categorical_encoder}"
        )

    # Pipeline categórico (apenas codificação, sem imputação aqui - será tratada separadamente)
    cat_pipeline = Pipeline([
        ('encoder', cat_encoder)
    ])

    # Transformer final
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    
    logger.info("Pipeline de pré-processamento criado com sucesso.")
    return preprocessor



def advanced_missing_imputation(
    df: pd.DataFrame,
    strategy: str = 'auto',
    report: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Realiza imputação automática de valores ausentes com relatório opcional.
    Aplica imputadores adequados conforme o tipo de coluna e gera relatório
    de missing values.
    
    Performs automatic imputation of missing values with optional reporting.
    Applies appropriate imputers based on column type and generates a missing
    values report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com valores ausentes.
    strategy : str, default='auto'
        Estratégia de imputação: 'simple' (média/modo), 'knn', 'iterative',
        ou 'auto' (detecta melhor).
    report : bool, default=True
        Se deve gerar e salvar relatório de missing values.
    
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        DataFrame imputado e relatório de missing values (se report=True).
    
    Notes:
    ------
    - Salva relatório em output/reports/missing_report.csv se report=True.
    - Usa SimpleImputer por padrão para categóricas (strategy='constant',
      fill_value='missing').
    """
    set_global_seeds(42)
    df_imputed = df.copy()
    missing_report = None
    
    if report:
        missing_before = df.isnull().sum()
        missing_pct = (missing_before / len(df)) * 100
        missing_report = pd.DataFrame({
            'column': df.columns,
            'missing_count': missing_before,
            'missing_percentage': missing_pct
        })
        missing_count = len(
            missing_report[missing_report['missing_percentage'] > 0]
        )
        logger.info(
            f"Relatório de missing values gerado. Colunas com >0% "
            f"ausentes: {missing_count}"
        )
    
    # Imputação numérica
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0 and df[num_cols].isnull().any().any():
        if strategy == 'auto':
            # Escolhe baseado no % de missing
            if (
                df[num_cols].isnull().sum().sum()
                / (len(df) * len(num_cols))
                < 0.5
            ):
                imputer = SimpleImputer(strategy='mean')
            else:
                try:
                    imputer = KNNImputer(n_neighbors=5)
                except ImportError:
                    imputer = SimpleImputer(strategy='mean')
        else:
            # Implementar baseado em strategy
            imputer = None
            if strategy == 'simple':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'knn':
                try:
                    imputer = KNNImputer(n_neighbors=5)
                except ImportError:
                    imputer = SimpleImputer(strategy='mean')
            elif strategy == 'iterative':
                try:
                    imputer = IterativeImputer(random_state=42)
                except ImportError:
                    imputer = SimpleImputer(strategy='mean')

        df_imputed[num_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols,
            index=df.index
        )
        logger.info(
            f"Imputação numérica aplicada em {len(num_cols)} colunas."
        )
    
    # Imputação categórica
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0 and df[cat_cols].isnull().any().any():
        cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        df_imputed[cat_cols] = pd.DataFrame(
            cat_imputer.fit_transform(df[cat_cols]),
            columns=cat_cols,
            index=df.index
        )
        logger.info(
            f"Imputação categórica aplicada em {len(cat_cols)} colunas."
        )
    
    if report:
        missing_after = df_imputed.isnull().sum()
        missing_report['imputed_count'] = missing_before - missing_after
        missing_report.to_csv('output/reports/missing_report.csv', index=False)
        logger.info("Relatório de imputação salvo em output/reports/missing_report.csv")
    
    return df_imputed, missing_report

def get_title(name: str) -> str:
    """
    Extract title from passenger name with robust error handling.

    Parameters:
    -----------
    name : str
        Passenger name.

    Returns:
    --------
    str
        Extracted title or 'Other' if not found.
    """
    titles = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Don': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr', 'Mme': 'Mrs', 'Ms': 'Miss',
        'Major': 'Mr', 'Lady': 'Mrs', 'Sir': 'Mr', 'Mlle': 'Miss',
        'Col': 'Mr', 'Capt': 'Mr', 'the Countess': 'Mrs', 'Jonkheer': 'Mr',
        'Dona': 'Mrs'
    }
    try:
        if ',' not in name or '.' not in name:
            return 'Other'
        title = name.split(',')[1].split('.')[0].strip()
        return titles.get(title, 'Other')
    except (IndexError, AttributeError):
        return 'Other'


def get_ticket_prefix(ticket: str) -> str:
    """
    Extract ticket prefix with robust error handling.

    Parameters:
    -----------
    ticket : str
        Ticket string.

    Returns:
    --------
    str
        Extracted prefix or 'UNKNOWN'.
    """
    try:
        if isinstance(ticket, str) and ' ' in ticket:
            return ticket.split(' ')[0]
        return 'UNKNOWN'
    except (AttributeError, TypeError):
        return 'UNKNOWN'


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family-related features."""
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    logger.info("Features de família criadas: FamilySize, IsAlone.")
    return df


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from passenger names."""
    df = df.copy()
    df['Title'] = df['Name'].apply(get_title)
    logger.info("Feature Title extraída de Name.")
    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck from cabin information."""
    df = df.copy()
    df['Deck'] = df['Cabin'].fillna('U').str[0].str.upper()
    df['Deck'] = df['Deck'].replace({'T': 'U'})  # T é raro, tratar como desconhecido
    logger.info("Feature Deck extraída de Cabin.")
    return df


def extract_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ticket prefix."""
    df = df.copy()
    df['TicketPrefix'] = df['Ticket'].apply(get_ticket_prefix)
    logger.info("Feature TicketPrefix extraída de Ticket.")
    return df


def create_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    df = df.copy()
    df['AgeClass'] = df['Age'] * df['Pclass']
    df['FarePerPerson'] = df['Fare'] / (df['SibSp'] + df['Parch'] + 1).replace(0, 1)
    if 'Title' in df.columns:
        df['Title_Interactions'] = df['Title'] + "_" + df['Sex']
    logger.info("Interações criadas: AgeClass, FarePerPerson, Title_Interactions.")
    return df


def create_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create binned features."""
    df = df.copy()
    # Handle NaNs in pd.cut by using np.inf for upper bound
    df['feat_AgeBin'] = pd.cut(
        df['Age'], bins=[0, 12, 18, 35, 60, np.inf],
        labels=['Child', 'Teen', 'Young', 'Adult', 'Senior']
    ).astype('category')
    df['feat_FareBin'] = pd.cut(
        df['Fare'], bins=[-1, 7.91, 14.45, 31, np.inf],
        labels=['Low', 'Medium', 'High', 'Luxury']
    ).astype('category')
    df['feat_AgeCategory_v2'] = pd.cut(
        df['Age'], bins=[0, 18, 35, 60, np.inf],
        labels=['Minor', 'YoungAdult', 'Adult', 'Senior']
    ).astype('category')
    df['feat_FareCategory_v2'] = pd.cut(
        df['Fare'], bins=[-1, 10, 50, np.inf],
        labels=['Cheap', 'Moderate', 'Expensive']
    ).astype('category')
    logger.info("Bins criados: feat_AgeBin, feat_FareBin, feat_AgeCategory_v2, feat_FareCategory_v2.")
    return df


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing value indicators."""
    df = df.copy()
    df['feat_Age_missing'] = df['Age'].isnull().astype(int)
    df['feat_Cabin_missing'] = df['Cabin'].isnull().astype(int)
    df['feat_Embarked_missing'] = df['Embarked'].isnull().astype(int)
    df['feat_Fare_missing'] = df['Fare'].isnull().astype(int)
    logger.info("Indicadores de missing criados: feat_*_missing.")
    return df


def apply_target_encoding(train: pd.DataFrame, test: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply target encoding to categorical features."""
    train = train.copy()
    test = test.copy()

    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
    for col in cat_cols:
        if col in train.columns:
            encoded_train = kfold_target_encode(train, col, target)
            train[encoded_train.name] = encoded_train

            # Apply to test using saved map
            map_path = f"output/cache/te_maps_{col}.json"
            if os.path.exists(map_path):
                try:
                    with open(map_path, 'r') as f:
                        te_map = json.load(f)
                    global_mean = te_map.pop('global_mean')
                    test[encoded_train.name] = test[col].map(te_map).fillna(global_mean)
                except Exception as e:
                    logger.warning(
                        f"Erro ao carregar mapa de target encoding para {col}: {e}. "
                        "Usando global mean."
                    )
                    global_mean = train[target].mean()
                    test[encoded_train.name] = global_mean
            else:
                global_mean = train[target].mean()
                test[encoded_train.name] = global_mean
                logger.warning(
                    f"Mapa de target encoding não encontrado para {col}, "
                    "usando global mean."
                )

    logger.info("Target encoding aplicado.")
    return train, test


def parallel_feature_engineering(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_parallel: bool = True,
    include_heavy_ops: bool = True,
    is_training: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica engenharia de atributos de forma paralela ou sequencial.
    Cria features básicas e opcionais pesadas (interações, bins,
    indicadores de missing).

    Applies feature engineering in parallel or sequential mode.
    Creates basic features and optional heavy ops (interactions, bins,
    missing indicators).

    Parameters:
    -----------
    train : pd.DataFrame
        Conjunto de treino.
    test : pd.DataFrame
        Conjunto de teste.
    use_parallel : bool, default=True
        Se deve usar processamento paralelo.
    include_heavy_ops : bool, default=True
        Se deve incluir operações pesadas (interações, bins, indicadores).
    is_training : bool, default=True
        Se é conjunto de treino (para target encoding).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Conjuntos de treino e teste com novas features.
    """
    set_global_seeds(42)

    # Lista de funções de feature engineering
    feature_functions = [
        create_family_features,
        extract_title,
        extract_deck,
        extract_ticket_prefix
    ]

    if include_heavy_ops:
        feature_functions.extend([
            create_interactions,
            create_bins,
            create_missing_indicators
        ])

    try:
        if use_parallel:
            # Import inside function to avoid multiprocessing issues
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                max_workers=max(1, multiprocessing.cpu_count() - 1)
            ) as executor:
                for func in feature_functions:
                    future = executor.submit(func, train)
                    try:
                        result = future.result()
                        if isinstance(result, pd.DataFrame):
                            train = result
                        else:
                            logger.warning(
                                f"Function {func.__name__} returned "
                                "non-DataFrame, skipping."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Parallel failed for {func.__name__} "
                            f"on train: {e}. Fallback."
                        )
                        result = func(train)
                        if isinstance(result, pd.DataFrame):
                            train = result
                for func in feature_functions:
                    future = executor.submit(func, test)
                    try:
                        result = future.result()
                        if isinstance(result, pd.DataFrame):
                            test = result
                        else:
                            logger.warning(
                                f"Function {func.__name__} returned "
                                "non-DataFrame, skipping."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Parallel failed for {func.__name__} "
                            f"on test: {e}. Fallback."
                        )
                        result = func(test)
                        if isinstance(result, pd.DataFrame):
                            test = result
            logger.info(
                "Feature engineering aplicada em paralelo com "
                "ProcessPoolExecutor."
            )
        else:
            for func in feature_functions:
                result_train = func(train)
                if isinstance(result_train, pd.DataFrame):
                    train = result_train
                result_test = func(test)
                if isinstance(result_test, pd.DataFrame):
                    test = result_test
                logger.info(
                    f"Feature engineering sequencial: {func.__name__}"
                )
    except Exception as e:
        logger.warning(
            f"Paralelismo falhou: {e}. Usando sequencial."
        )
        for func in feature_functions:
            result_train = func(train)
            if isinstance(result_train, pd.DataFrame):
                train = result_train
            result_test = func(test)
            if isinstance(result_test, pd.DataFrame):
                test = result_test

    # Aplicar target encoding se for treino
    if is_training and 'Survived' in train.columns:
        train, test = apply_target_encoding(train, test, 'Survived')
        logger.info("Target encoding aplicado durante feature engineering.")

    # Garantir que colunas novas existam em ambos
    new_cols = [
        'FamilySize', 'IsAlone', 'Title', 'Deck', 'TicketPrefix'
    ]
    if include_heavy_ops:
        new_cols.extend([
            'AgeClass', 'FarePerPerson', 'Title_Interactions',
            'feat_AgeBin', 'feat_FareBin', 'feat_AgeCategory_v2',
            'feat_FareCategory_v2', 'feat_Age_missing',
            'feat_Cabin_missing', 'feat_Embarked_missing',
            'feat_Fare_missing',
        ])
    if is_training and 'Survived' in train.columns:
        new_cols.extend([
            'feat_Pclass_te', 'feat_Sex_te', 'feat_Embarked_te',
            'feat_Title_te', 'feat_Deck_te', 'feat_TicketPrefix_te',
        ])
    for col in new_cols:
        if col not in test.columns and col in train.columns:
            test[col] = (
                train[col].mode()[0]
                if train[col].dtype == 'object'
                else train[col].median()
            )

    # Apply feature selection if enabled
    if (
        CONFIG["feature_selection"] and is_training and
        'Survived' in train.columns
    ):
        # Identify potential feature columns (exclude IDs, names, tickets,
        # cabin, target)
        exclude_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
        potential_features = [
            col for col in train.columns
            if col not in exclude_cols
        ]
        selected_features = select_features_via_model(
            train[potential_features + ['Survived']],
            'Survived',
            potential_features,
            model_type=CONFIG["selection_model"],
            threshold=CONFIG["selection_threshold"],
            random_state=CONFIG["random_state"]
        )
        # Filter train and test to selected features + essential columns
        essential_train = ['PassengerId', 'Survived']
        essential_test = ['PassengerId']
        train_cols = (
            [col for col in selected_features if col in train.columns] +
            [col for col in essential_train if col in train.columns]
        )
        test_cols = (
            [col for col in selected_features if col in test.columns] +
            [col for col in essential_test if col in test.columns]
        )
        train = train[train_cols]
        test = test[test_cols]
        logger.info(
            "Feature selection applied: {} features selected from {}."
            .format(
                len(selected_features), len(potential_features)
            )
        )

    logger.info(
        f"Feature engineering concluída. Novas features: {new_cols}"
    )
    return train, test

def kfold_target_encode(
    df: pd.DataFrame,
    col: Union[str, pd.Series],
    target: str,
    n_splits: int = 5,
    prior: float = 10,
    suffix: str = "_te",
    seed: int = 42
) -> pd.Series:
    """
    Codifica uma variável categórica utilizando Target Encoding com KFold e
    smoothing. Encodes a categorical feature using KFold Target Encoding with
    smoothing to prevent leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Conjunto de treino.
    col : str or pd.Series
        Coluna a ser codificada.
    target : str
        Coluna target (Survived).
    n_splits : int, default=5
        Número de folds.
    prior : float, default=10
        Parâmetro de smoothing (Bayesian prior).
    suffix : str, default="_te"
        Sufixo para a nova coluna.
    seed : int, default=42
        Semente para reprodutibilidade.

    Returns:
    --------
    pd.Series
        Série com os valores encoded.
    """
    set_global_seeds(seed)

    if isinstance(col, str):
        col_name = col
    else:
        col_name = col.name or 'unknown'

    logger.info(
        f"Iniciando target encoding para coluna {col_name}"
    )

    # Cálculo global mean para unseen
    global_mean = df[target].mean()

    # Usar StratifiedKFold se possível, senão KFold
    y_arr = np.asarray(df[target])
    unique, counts = np.unique(y_arr, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    if min_count >= 2 and n_splits <= min_count:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_stratified = True
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        use_stratified = False

    encoded = np.zeros(len(df))

    for train_idx, val_idx in (
        kf.split(df, df[target]) if use_stratified else kf.split(df)
    ):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]

        # Mean por categoria no fold de treino
        fold_means = train_fold.groupby(col_name, observed=False)[target].agg(
            ["mean", "count"]
        )

        # Aplicar smoothing
        smoothed_means = (
            fold_means["count"] * fold_means["mean"] + prior * global_mean
        ) / (fold_means["count"] + prior)

        # Aplicar no fold de validação
        # Mapear os valores e garantir que o resultado seja numérico antes de
        # preencher NaNs. Isso evita erros quando a coluna original é do tipo
        # 'category' e o fillna tenta inserir um valor que não está nas
        # categorias (levando a TypeError).
        mapped = val_fold[col_name].map(smoothed_means)
        mapped = pd.to_numeric(mapped, errors='coerce')
        mapped = mapped.fillna(global_mean)
        encoded[val_idx] = mapped.to_numpy(dtype=np.float64)

    # Salvar mapa (usando o último smoothed_means para compatibilidade)
    te_map = {k: float(v) for k, v in smoothed_means.to_dict().items()}
    te_map['global_mean'] = float(global_mean)
    map_path = f"output/cache/te_maps_{col_name}.json"
    try:
        with open(map_path, 'w') as f:
            json.dump(te_map, f)
        logger.info(
            f"Target encoding concluído e salvo em {map_path}"
        )
    except Exception as e:
        logger.warning(
            f"Não foi possível salvar o mapa de target encoding: {e}"
        )

    return pd.Series(encoded, index=df.index, name=f"feat_{col_name}{suffix}")


def select_features_via_model(
    train: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    model_type: str = 'RandomForest',
    threshold: float = 0.01,
    random_state: int = 42
) -> List[str]:
    """
    Select features using model-based importance scores.

    Fits a model (RandomForest or XGBoost) on the training data and selects
    features with importance above the specified threshold.

    Parameters:
    -----------
    train : pd.DataFrame
        Training dataset.
    target : str
        Target column name.
    feature_cols : List[str]
        List of feature columns to consider.
    model_type : str, default='RandomForest'
        Model to use for feature selection ('RandomForest' or 'XGBoost').
    threshold : float, default=0.01
        Minimum importance threshold for feature selection.
    random_state : int, default=42
        Random state for reproducibility.

    Returns:
    --------
    List[str]
        Selected feature columns.

    Raises:
    -------
    ValueError
        If model_type is not supported or no features are selected.
    """
    from sklearn.ensemble import RandomForestClassifier

    # Prepare data
    X = train[feature_cols]
    y = train[target]

    # Select model
    if model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'XGBoost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
        except ImportError:
            logger.warning(
                "XGBoost not available, using RandomForest."
            )
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Use "
            "'RandomForest' or 'XGBoost'."
        )

    # Fit model
    model.fit(X, y)

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError(
            f"Model {model_type} does not have "
            "feature_importances_ attribute."
        )

    # Create importance dict
    importance_dict = dict(zip(feature_cols, importances))

    # Select features above threshold
    selected_features = [feat for feat, imp in importance_dict.items() if imp > threshold]

    if len(selected_features) == 0:
        logger.warning(
            f"No features selected with threshold {threshold}. "
            "Using all features."
        )
        selected_features = feature_cols

    logger.info(
        f"Feature selection with {model_type}: "
        f"{len(selected_features)}/{len(feature_cols)} features "
        f"selected (threshold={threshold})"
    )
    return selected_features


def build_feature_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    extra_features: Optional[List[str]] = None
) -> List[str]:
    """
    Constrói lista final de features utilizáveis em ambos os conjuntos.
    Garante interseção de colunas entre train e test.
    
    Builds the final list of usable features for both datasets.
    Ensures intersection of columns between train and test.
    
    Parameters:
    -----------
    train : pd.DataFrame
        Conjunto de treino.
    test : pd.DataFrame
        Conjunto de teste.
    extra_features : list of str, optional
        Features adicionais a incluir.
    
    Returns:
    --------
    list of str
        Lista ordenada de features comuns.
    """
    # Colunas comuns
    common_cols = train.columns.intersection(test.columns)
    
    # Remover target se presente
    if 'Survived' in common_cols:
        common_cols = common_cols.drop('Survived')
    
    # Adicionar extra se fornecidas e presentes
    if extra_features:
        common_cols = common_cols.union(extra_features)
    
    # Ordenar
    feature_cols = sorted(common_cols.tolist())
    
    logger.info(
        f"Feature set construído: {len(feature_cols)} features."
    )
    return feature_cols


class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering class for Titanic dataset.

    Encapsulates all feature engineering steps including family features,
    title extraction, deck extraction, interactions, bins, missing indicators,
    and target encoding. Provides options for parallel processing and
    configurable heavy operations.

    Parameters:
    -----------
    use_parallel : bool, default=True
        Whether to use parallel processing for feature engineering.
    include_heavy_ops : bool, default=True
        Whether to include heavy operations (interactions, bins, missing indicators).
    is_training : bool, default=True
        Whether this is training data (affects target encoding).
    target_col : str, default='Survived'
        Name of the target column for target encoding.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        use_parallel: bool = True,
        include_heavy_ops: bool = True,
        is_training: bool = True,
        target_col: str = 'Survived',
        random_state: int = 42
    ):
        self.use_parallel = use_parallel
        self.include_heavy_ops = include_heavy_ops
        self.is_training = is_training
        self.target_col = target_col
        self.random_state = random_state

        # Set global seeds for reproducibility
        set_global_seeds(random_state)

        logger.info(
            f"AdvancedFeatureEngineer initialized: parallel={use_parallel}, "
            f"heavy_ops={include_heavy_ops}, training={is_training}"
        )

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply all feature engineering steps to train and test datasets.

        Parameters:
        -----------
        train : pd.DataFrame
            Training dataset.
        test : pd.DataFrame
            Test dataset.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Processed train and test datasets.
        """
        logger.info("Starting advanced feature engineering...")

        # Make copies to avoid modifying originals
        train_processed = train.copy()
        test_processed = test.copy()

        # List of basic feature engineering functions
        basic_functions = [
            self._create_family_features,
            self._extract_title,
            self._extract_deck,
            self._extract_ticket_prefix
        ]

        # Add heavy operations if enabled
        if self.include_heavy_ops:
            basic_functions.extend([
                self._create_interactions,
                self._create_bins,
                self._create_missing_indicators
            ])

        # Apply feature engineering
        if self.use_parallel:
            train_processed, test_processed = self._apply_parallel(
                train_processed, test_processed, basic_functions
            )
        else:
            train_processed, test_processed = self._apply_sequential(
                train_processed, test_processed, basic_functions
            )

        # Apply target encoding if training and target column exists
        if self.is_training and self.target_col in train_processed.columns:
            train_processed, test_processed = self._apply_target_encoding(
                train_processed, test_processed
            )

        # Ensure consistency between train and test columns
        train_processed, test_processed = self._ensure_column_consistency(
            train_processed, test_processed
        )

        logger.info("Advanced feature engineering completed.")
        return train_processed, test_processed

    def _create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family-related features."""
        return create_family_features(df)

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract title from passenger names."""
        return extract_title(df)

    def _extract_deck(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract deck from cabin information."""
        return extract_deck(df)

    def _extract_ticket_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ticket prefix."""
        return extract_ticket_prefix(df)

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        return create_interactions(df)

    def _create_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned features."""
        return create_bins(df)

    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create missing value indicators."""
        return create_missing_indicators(df)

    def _apply_target_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply target encoding to categorical features."""
        return apply_target_encoding(train, test, self.target_col)

    def _apply_parallel(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        functions: List[Callable]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature engineering functions in parallel."""
        try:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(
                max_workers=max(1, multiprocessing.cpu_count() - 1)
            ) as executor:
                # Apply to train
                for func in functions:
                    future = executor.submit(func, train)
                    try:
                        result = future.result()
                        if isinstance(result, pd.DataFrame):
                            train = result
                        else:
                            logger.warning(
                                f"Function {func.__name__} returned "
                                "non-DataFrame, skipping."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Parallel failed for {func.__name__} "
                            f"on train: {e}. Fallback to sequential."
                        )
                        result = func(train)
                        if isinstance(result, pd.DataFrame):
                            train = result

                # Apply to test
                for func in functions:
                    future = executor.submit(func, test)
                    try:
                        result = future.result()
                        if isinstance(result, pd.DataFrame):
                            test = result
                        else:
                            logger.warning(
                                f"Function {func.__name__} returned "
                                "non-DataFrame, skipping."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Parallel failed for {func.__name__} "
                            f"on test: {e}. Fallback to sequential."
                        )
                        result = func(test)
                        if isinstance(result, pd.DataFrame):
                            test = result

            logger.info("Feature engineering applied in parallel.")
        except Exception as e:
            logger.warning(
                f"Parallel processing failed: {e}. Using sequential."
            )
            train, test = self._apply_sequential(train, test, functions)

        return train, test

    def _apply_sequential(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        functions: List[Callable]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature engineering functions sequentially."""
        for func in functions:
            try:
                result_train = func(train)
                if isinstance(result_train, pd.DataFrame):
                    train = result_train

                result_test = func(test)
                if isinstance(result_test, pd.DataFrame):
                    test = result_test

                logger.info(f"Applied {func.__name__} sequentially.")
            except Exception as e:
                logger.warning(
                    f"Failed to apply {func.__name__}: {e}. Skipping."
                )

        return train, test

    def _ensure_column_consistency(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ensure train and test have consistent columns."""
        # Get all possible new columns
        new_cols = [
            'FamilySize', 'IsAlone', 'Title', 'Deck', 'TicketPrefix'
        ]

        if self.include_heavy_ops:
            new_cols.extend([
                'AgeClass', 'FarePerPerson', 'Title_Interactions',
                'feat_AgeBin', 'feat_FareBin', 'feat_AgeCategory_v2',
                'feat_FareCategory_v2', 'feat_Age_missing',
                'feat_Cabin_missing', 'feat_Embarked_missing',
                'feat_Fare_missing'
            ])

        if self.is_training and self.target_col in train.columns:
            new_cols.extend([
                'feat_Pclass_te', 'feat_Sex_te', 'feat_Embarked_te',
                'feat_Title_te', 'feat_Deck_te', 'feat_TicketPrefix_te'
            ])

        # Add missing columns to test with appropriate defaults
        for col in new_cols:
            if col not in test.columns and col in train.columns:
                if train[col].dtype == 'object' or train[col].dtype.name == 'category':
                    test[col] = train[col].mode()[0] if not train[col].mode().empty else 'Unknown'
                else:
                    test[col] = train[col].median() if not train[col].isna().all() else 0

        logger.info("Column consistency ensured between train and test.")
        return train, test
