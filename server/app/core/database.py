from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(DATABASE_URL)

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    connect_args={"ssl": None}
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,
)

Base = declarative_base()


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def get_db():
    async with AsyncSessionFactory() as session:
        yield session
