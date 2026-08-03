"""Central database connection for the hotel pricing project.

Every script and notebook imports get_engine() from here, so the
connection is defined in exactly one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Load .env from the project root, regardless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_connection_url() -> str:
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    database = os.getenv("POSTGRES_DB")

    missing = [k for k, v in {
        "POSTGRES_USER": user,
        "POSTGRES_PASSWORD": password,
        "POSTGRES_DB": database,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing in .env: {', '.join(missing)}. "
            "Check that .env exists in the project root."
        )

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine() -> Engine:
    """SQLAlchemy engine. pool_pre_ping avoids stale-connection errors
    when the Docker container restarts between work sessions."""
    return create_engine(get_connection_url(), pool_pre_ping=True)


if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as conn:
        db = conn.execute(text("SELECT current_database()")).scalar()
        user = conn.execute(text("SELECT current_user")).scalar()
        version = conn.execute(text("SHOW server_version")).scalar()
    print(f"Connected")
    print(f"  database : {db}")
    print(f"  user     : {user}")
    print(f"  version  : PostgreSQL {version}")