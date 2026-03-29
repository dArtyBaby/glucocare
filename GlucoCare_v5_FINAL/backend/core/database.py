# core/database.py
"""
WHY asyncpg over psycopg2:
  asyncpg is a pure-async PostgreSQL driver. With FastAPI (which is async-first),
  using a synchronous driver like psycopg2 would BLOCK the event loop on every
  database call — killing your concurrency advantage entirely.
  asyncpg can handle thousands of concurrent DB operations without blocking.

WHY a connection pool (create_async_engine with pool settings):
  Creating a new DB connection for every request is expensive (~30ms).
  A connection pool keeps N connections open and reuses them.
  pool_size=20 means 20 concurrent DB operations; pool_pre_ping=True checks
  if a connection is still alive before using it (prevents "stale connection" errors
  after Supabase network timeouts).
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event
import logging

from core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """
    All SQLAlchemy models inherit from this.
    WHY one shared Base: SQLAlchemy needs to know about ALL models
    to generate migrations and enforce relationships.
    """
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    # WHY echo=False in production: SQL logging is only for debugging.
    # In production it floods your logs and leaks query structure.
    echo=settings.DEBUG,
    pool_size=20,           # Max 20 persistent connections
    max_overflow=10,        # Allow 10 extra connections under peak load
    pool_pre_ping=True,     # Test connections before use (handles Supabase timeouts)
    pool_recycle=3600,      # Recycle connections every hour (prevents stale connections)
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # WHY expire_on_commit=False: By default SQLAlchemy expires all attributes
    # after commit, forcing a re-fetch. In async code this causes implicit lazy loads
    # (which are forbidden in async SQLAlchemy). Disabling it prevents subtle bugs.
)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a database session.

    WHY yield (not return): Using 'yield' makes this a context manager.
    FastAPI guarantees the code after 'yield' runs even if the route raises
    an exception — ensuring the session is ALWAYS closed.

    EDGE CASE: What if the DB is down?
    SQLAlchemy will raise OperationalError. Our global error handler
    (core/exceptions.py) catches this and returns a 503 Service Unavailable
    instead of a raw 500 crash.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables. Called once on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised successfully.")
