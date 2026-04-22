from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.settings import get_settings


def create_session_factory(database_url: str | None = None) -> async_sessionmaker[AsyncSession]:
    url = database_url or get_settings().database_url
    engine = create_async_engine(url, pool_pre_ping=True)
    return async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return create_session_factory()


def SessionLocal() -> AsyncSession:
    return get_session_factory()()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
