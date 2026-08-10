import logging
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Naive UTC datetime, matching the existing DateTime columns.

    Replacement for the deprecated datetime.utcnow(). We store naive UTC
    timestamps throughout the schema; callers that need aware datetimes
    should attach `timezone.utc` themselves.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./volumes/speakers.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Enable foreign key constraints for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        # Give writers 30 s to acquire the write lock before raising
        # "database is locked". Python's sqlite3 default is 5 s, which is
        # easily exceeded by transactions that wait on GPU work.
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Lightweight migrations for columns added after the initial schema shipped.
# Each tuple is (table, column, sql_type_with_default). Nullable by default
# for the app's read paths; supply DEFAULT ... for columns the ORM treats as
# non-null scalars.
_MIGRATIONS = (
    ("conversation_segments", "words_data", "TEXT"),
    ("conversation_segments", "avg_logprob", "REAL"),
    ("conversation_segments", "emotion_corrected", "INTEGER DEFAULT 0"),
    ("conversation_segments", "emotion_corrected_at", "TEXT"),
    ("conversation_segments", "emotion_misidentified", "INTEGER DEFAULT 0"),
    ("conversation_segments", "speaker_embedding", "BLOB"),
    ("conversation_segments", "emotion_embedding", "BLOB"),
    ("speakers", "emotion_threshold", "REAL"),
    ("conversations", "processing_token", "TEXT"),
    ("conversations", "snapshot_uuid", "TEXT"),
    ("conversation_segments", "snapshot_uuid", "TEXT"),
)

_SNAPSHOT_UUID_TABLES = ("conversations", "conversation_segments")


def init_db():
    """Create tables, then apply additive column migrations in a single pass.

    Errors are raised rather than swallowed — a silently-skipped ALTER leaves
    the schema in a state that only fails later at query time, far from the
    real cause.
    """
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table, column, sql_type in _MIGRATIONS:
            if table not in existing_tables:
                continue
            columns = {col["name"] for col in inspector.get_columns(table)}
            if column in columns:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
            logger.info(f"Added {column} column to {table}")

        # SQLite integer primary keys can be reused after deletion. Backups
        # therefore need a persistent row identity before they may replay
        # historical segment state. Backfill existing installations and make
        # the migration restart-safe; new rows receive the ORM default.
        for table in _SNAPSHOT_UUID_TABLES:
            if table not in existing_tables:
                continue
            missing_ids = conn.execute(text(
                f"SELECT id FROM {table} "
                "WHERE snapshot_uuid IS NULL OR snapshot_uuid = ''"
            )).scalars().all()
            for row_id in missing_ids:
                conn.execute(
                    text(f"UPDATE {table} SET snapshot_uuid = :value WHERE id = :id"),
                    {"value": str(uuid.uuid4()), "id": row_id},
                )
            conn.execute(text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS ix_{table}_snapshot_uuid "
                f"ON {table} (snapshot_uuid)"
            ))
