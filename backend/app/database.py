from pathlib import Path

from sqlmodel import SQLModel, create_engine

from . import models  # Register database tables


BACKEND_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BACKEND_DIR / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_FILE = DATABASE_DIR / "elan.db"

DATABASE_URL = (
    f"sqlite:///{DATABASE_FILE.as_posix()}"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
