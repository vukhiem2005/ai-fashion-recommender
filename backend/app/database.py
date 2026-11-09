import os
from pathlib import Path

from sqlmodel import SQLModel, create_engine

from . import models  # Register database tables
BACKEND_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BACKEND_DIR / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# Database selection.
# Default: preserve the existing local catalog.
CATALOG_MODE = os.environ.get(
    "ELAN_CATALOG_MODE", "local"
).strip().lower()

DATABASE_FILENAMES = {
    "local": "elan.db",
    "public_demo": "elan_public_demo.db",
}

if CATALOG_MODE not in DATABASE_FILENAMES:
    raise ValueError(
        "Invalid ELAN_CATALOG_MODE. "
        "Use 'local' or 'public_demo'."
    )

DATABASE_FILE = (
    DATABASE_DIR / DATABASE_FILENAMES[CATALOG_MODE]
)

# Never silently create an empty public demo database.
if CATALOG_MODE == "public_demo" and not DATABASE_FILE.is_file():
    raise FileNotFoundError(
        f"Public demo database not found: {DATABASE_FILE}"
    )


DATABASE_URL = (
    f"sqlite:///{DATABASE_FILE.as_posix()}"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
