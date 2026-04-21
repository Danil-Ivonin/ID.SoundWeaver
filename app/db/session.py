from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.settings import get_settings


def create_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    url = database_url or get_settings().database_url
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


SessionLocal = create_session_factory()


def get_db_session() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
