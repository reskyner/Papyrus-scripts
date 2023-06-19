from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
import os

# "postgresql://postgres:postgres@papyrusdb/papyrus"
url = 'postgresql://postgres:postgres@localhost:5432/papyrus'

engine = create_engine(url,)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    engine = create_engine(
        'postgresql://postgres:postgres@localhost:5432/papyrus',
        pool_recycle=3600, pool_size=10)
    db_session = scoped_session(sessionmaker(
        autocommit=False, autoflush=False, bind=engine))
    
    return db_session