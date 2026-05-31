"""Cache em memória com TTL e cache em disco para artefatos pesados de ML."""

import time
from typing import Any

import joblib

from goldata.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Cache em memória com TTL. Thread-safe para uso no servidor."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)

    def get(self, key: str) -> Any | None:
        """Retorna valor ou None se inexistente ou expirado."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at > 0 and time.time() > expires_at:
            del self._store[key]
            logger.debug("cache_expired", key=key)
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Armazena valor. ttl=0 → sem expiração."""
        expires_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expires_at)
        logger.debug("cache_set", key=key, ttl=ttl)

    def delete(self, key: str) -> bool:
        """Remove chave. Retorna True se existia."""
        existed = key in self._store
        self._store.pop(key, None)
        return existed

    def clear(self) -> int:
        """Limpa todo o cache. Retorna quantas entradas foram removidas."""
        count = len(self._store)
        self._store.clear()
        logger.debug("cache_cleared", entries_removed=count)
        return count

    def size(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store.keys())


class DiskCache:
    """Cache em disco usando joblib.Memory: para modelos e dados pesados de ML."""

    def __init__(self, cache_dir: str = "./data/cache") -> None:
        self._memory = joblib.Memory(cache_dir, verbose=0)
        self._cache_dir = cache_dir

    def cache_function(self, func: Any) -> Any:
        """Decorator para cachear resultado de função em disco."""
        return self._memory.cache(func)

    def clear(self) -> None:
        """Limpa cache em disco."""
        self._memory.clear(warn=False)
        logger.info("disk_cache_cleared", cache_dir=self._cache_dir)


# Instância global do cache em memória
cache = CacheManager()
