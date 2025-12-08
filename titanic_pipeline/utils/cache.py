"""
Cache management utilities for Titanic ML Pipeline.
"""

import hashlib
import logging
import os
import pickle
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of expensive operations."""

    def __init__(self, cache_dir: str = "output/cache", enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, data_hash: str, operation: str) -> str:
        """Generate a cache key from data hash and operation."""
        return hashlib.md5(f"{data_hash}_{operation}".encode()).hexdigest()

    def load_cached_result(self, key: str) -> Optional[Any]:
        """Load a cached result if it exists."""
        if not self.enabled:
            return None

        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                logger.info(f"   📖 Loaded cached result for key: {key}")
                return result
            except Exception as e:
                logger.warning(
                    f"   ⚠️  Failed to load cache for key {key}: {e}"
                )
                return None
        return None

    def save_cached_result(self, key: str, result: Any) -> None:
        """Save a result to cache."""
        if not self.enabled:
            return

        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"   💾 Cached result for key: {key}")
        except Exception as e:
            logger.warning(
                f"   ⚠️  Failed to save cache for key {key}: {e}"
            )

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache files, optionally matching a pattern."""
        if not os.path.exists(self.cache_dir):
            return 0

        cleared = 0
        for filename in os.listdir(self.cache_dir):
            if pattern is None or pattern in filename:
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                    cleared += 1
                except Exception as e:
                    logger.warning(
                        f"   ⚠️  Failed to remove {filename}: {e}"
                    )

        logger.info(f"   🗑️  Cleared {cleared} cache files")
        return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not os.path.exists(self.cache_dir):
            return {"total_files": 0, "total_size_mb": 0}

        total_size = 0
        files = []
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                files.append({"name": filename, "size_bytes": size})

        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "files": files,
        }


def get_cache_key(data_hash: str, operation: str) -> str:
    """Generate a cache key from data hash and operation."""
    return hashlib.md5(f"{data_hash}_{operation}".encode()).hexdigest()


def cache_result(key: str, result: Any, cache_dir: str = "output/cache") -> None:
    """Save a result to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(
            f"   💾 Cached result for key: {key}"
        )
    except Exception as e:
        logger.warning(
            f"   ⚠️  Failed to save cache for key {key}: {e}"
        )


def load_cached_result(key: str, cache_dir: str = "output/cache") -> Optional[Any]:
    """Load a cached result if it exists."""
    cache_path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            logger.info(f"   📖 Loaded cached result for key: {key}")
            return result
        except Exception as e:
            logger.warning(
                f"   ⚠️  Failed to load cache for key {key}: {e}"
            )
            return None
    return None
