"""Conexão assíncrona com banco de dados: PostgreSQL ou SQLite (fallback)."""

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from goldata.config import get_settings
from goldata.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _build_engine() -> AsyncEngine:
    url = settings.database_url
    try:
        engine = create_async_engine(
            url,
            echo=settings.is_development,
            connect_args={"check_same_thread": False} if "sqlite" in url else {},
        )
        logger.info("db_engine_created", url=url.split("://")[0])
        return engine
    except Exception:
        sqlite_url = "sqlite+aiosqlite:///goldata_dev.db"
        logger.warning("db_fallback_to_sqlite")
        return create_async_engine(sqlite_url, echo=settings.is_development)


engine: AsyncEngine = _build_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency FastAPI: sessão de banco."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_connection() -> bool:
    """Verifica se o banco está acessível."""
    try:
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("db_health_check_failed", error=str(e))
        return False


async def create_all_tables() -> None:
    """Cria todas as tabelas."""
    from goldata.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db_tables_created")


async def drop_all_tables() -> None:
    """Remove todas as tabelas."""
    from goldata.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("db_tables_dropped")
