"""
Database configuration và session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from config import Config

# Tạo engine
engine = create_engine(
    Config.DATABASE_URL,
    poolclass=NullPool,
    echo=False  # Set True để debug SQL queries
)

# Tạo session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Dependency để lấy database session
    Sử dụng với FastAPI Depends
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

