"""
Utils module for Titanic ML Pipeline.
Contains utility functions for caching, validation, etc.
"""

import os
import pickle
import hashlib
import json
import logging
from typing import Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

def get_cache_key(data_hash: str, operation: str) -> str:
    return hashlib.md5(f"{data_hash}_{operation}".encode()).hexdigest()

def validate_data_schema(df: pd.DataFrame, expected_columns: List[str], dataset_name: str) -> bool:
    """Valida schema dos dados de entrada e salva relatório JSON"""
    logger.info(f"🔍 VALIDANDO SCHEMA: {dataset_name}")

    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)

    schema_report = {
        "dataset_name": dataset_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "expected_columns": expected_columns,
        "actual_columns": list(df.columns),
        "missing_columns": list(missing_cols),
        "extra_columns": list(extra_cols),
        "is_valid": len(missing_cols) == 0,
        "data_types": df.dtypes.to_dict(),
    }

    os.makedirs("output/relatorios", exist_ok=True)
    with open(f"output/relatorios/schema_validation_{dataset_name}.json", "w") as f:
        json.dump(schema_report, f, indent=2, default=str)

    if missing_cols:
        logger.error(f"Colunas faltantes em {dataset_name}: {missing_cols}")
        return False

    if extra_cols:
        logger.warning(f"Colunas extras em {dataset_name}: {extra_cols}")

    logger.info(
        f"✅ Schema válido para {dataset_name}: {len(df)} linhas, "
        f"{len(df.columns)} colunas"
    )
    logger.info(
        f"   📄 Relatório salvo em "
        f"output/relatorios/schema_validation_{dataset_name}.json"
    )
    return True

def cache_result(key: str, result: Any, cache_dir: str = "output/cache") -> None:
    # Assume CONFIG is global or imported
    try:
        from __main__ import CONFIG
        if not CONFIG.get('cache_enabled', True):
            return
    except ImportError:
        pass  # Assume enabled

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{key}.pkl")

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        logger.debug(f"💾 Resultado cached: {key}")
    except Exception as e:
        logger.warning(f"⚠️  Cache falhou para {key}: {e}")

def load_cached_result(key: str, cache_dir: str = "output/cache") -> Optional[Any]:
    try:
        from __main__ import CONFIG
        if not CONFIG.get('cache_enabled', True):
            return None
    except ImportError:
        pass

    cache_file = os.path.join(cache_dir, f"{key}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
            logger.debug(f"📖 Resultado loaded from cache: {key}")
            return result
        except Exception as e:
            logger.warning(f"⚠️  Cache load falhou para {key}: {e}")

    return None

def data_hash_calc(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
