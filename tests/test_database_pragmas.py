"""Regression tests for SQLite pragmas (GLM-004/OPUS-006/QWEN-014) and the
conversation-list query shape.
"""
from sqlalchemy import create_engine, inspect, text


def test_busy_timeout_pragma_applied(tmp_path, monkeypatch):
    """app.database must set PRAGMA busy_timeout (default 5s is too short for
    transactions that wait on GPU work)."""
    from app import database
    engine = create_engine(
        f"sqlite:///{tmp_path}/pragma.db",
        connect_args={"check_same_thread": False},
    )
    # Apply the same connect listener logic the app uses.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _same(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    with engine.connect() as conn:
        timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
        fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
    engine.dispose()
    assert timeout == 30000
    assert fk == 1

    # And assert the app's own engine got the pragma too (module-level engine).
    with database.engine.connect() as conn:
        app_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert app_timeout == 30000, "app.database engine must configure busy_timeout"


def test_conversation_list_does_not_eager_load_segments(tmp_path):
    """The 'lightweight' list endpoint must issue a single conversations query
    (suspicion raised during audit; verified as a non-finding at pinned SHA —
    this test pins the correct behaviour against regression)."""
    from datetime import datetime

    from sqlalchemy import event
    from sqlalchemy.orm import sessionmaker

    from app.database import Base
    from app.models import Conversation, ConversationSegment

    engine = create_engine(f"sqlite:///{tmp_path}/list.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    conv = Conversation(title="probe", start_time=datetime.now(), status="completed",  # noqa: DTZ005 (schema uses naive UTC)
                        num_segments=1)
    db.add(conv)
    db.flush()
    seg = ConversationSegment(
        conversation_id=conv.id,
        start_time=datetime.now(), end_time=datetime.now(),  # noqa: DTZ005 (schema uses naive UTC)
        start_offset=0.0, end_offset=1.0, speaker_name="Unknown", text="hello",
    )
    seg.words_data = '[{"word":"hello"}]'
    db.add(seg)
    db.commit()
    db.close()

    selects = []

    @event.listens_for(engine, "before_cursor_execute")
    def record(_conn, _cursor, statement, _params, _context, _many):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(" ".join(statement.split()))

    db = Session()
    db.query(Conversation).order_by(Conversation.start_time.desc()).limit(50).all()
    assert len(selects) == 1, f"expected 1 SELECT, got: {selects}"
    assert not any("conversation_segments" in s for s in selects)
    db.close()
    engine.dispose()


def test_old_schema_migration_backfills_stable_snapshot_uuids(tmp_path, monkeypatch):
    from app import database

    engine = create_engine(f"sqlite:///{tmp_path}/old-schema.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY)"))
        conn.execute(text(
            "CREATE TABLE conversation_segments "
            "(id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL)"
        ))
        conn.execute(text("INSERT INTO conversations (id) VALUES (1), (2)"))
        conn.execute(text(
            "INSERT INTO conversation_segments (id, conversation_id) "
            "VALUES (1, 1), (2, 2)"
        ))

    monkeypatch.setattr(database, "engine", engine)
    database.init_db()
    database.init_db()  # idempotent restart

    schema = inspect(engine)
    assert "snapshot_uuid" in {
        column["name"] for column in schema.get_columns("conversations")
    }
    assert "snapshot_uuid" in {
        column["name"] for column in schema.get_columns("conversation_segments")
    }
    with engine.connect() as conn:
        conversation_ids = conn.execute(text(
            "SELECT snapshot_uuid FROM conversations ORDER BY id"
        )).scalars().all()
        segment_ids = conn.execute(text(
            "SELECT snapshot_uuid FROM conversation_segments ORDER BY id"
        )).scalars().all()
    assert len(conversation_ids) == len(set(conversation_ids)) == 2
    assert len(segment_ids) == len(set(segment_ids)) == 2
    assert all(conversation_ids)
    assert all(segment_ids)
    engine.dispose()
