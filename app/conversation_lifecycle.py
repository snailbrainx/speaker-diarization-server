"""Authoritative internal lifecycle leases for conversation writers.

Public status text is presentation state, not a lock.  Long-running workers own a
random token stored in SQLite.  Segment persistence and finalisation both take a
conditional write lock against that token, so a timeout cannot become a late
post-finalisation commit.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy.orm import Session

from .models import Conversation

ACTIVE_CONVERSATION_STATUSES = frozenset({
    "recording",
    "processing",
    "finalizing",
    "deleting",
})


def new_processing_token() -> str:
    return uuid.uuid4().hex


def acquire_processing_lease(
    db: Session,
    conversation_id: int,
    *,
    status: str = "processing",
    allowed_statuses: Iterable[str] | None = None,
) -> str | None:
    """Atomically acquire an idle conversation for a long-running operation.

    Callers invoke this before making any request-local mutations.  Rolling back
    first ends a stale SQLite read snapshot so the conditional UPDATE observes
    the current owner and cannot be upgraded from an obsolete snapshot.
    """
    db.rollback()
    token = new_processing_token()
    query = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.processing_token.is_(None),
    )
    if allowed_statuses is None:
        query = query.filter(
            Conversation.status.notin_(ACTIVE_CONVERSATION_STATUSES)
        )
    else:
        query = query.filter(Conversation.status.in_(tuple(allowed_statuses)))

    updated = query.update(
        {"processing_token": token, "status": status},
        synchronize_session=False,
    )
    if updated != 1:
        db.rollback()
        return None
    db.commit()
    return token


def claim_segment_persistence(
    db: Session,
    conversation_id: int,
    token: str,
) -> bool:
    """Acquire SQLite's write lock iff this streaming generation is still live.

    The no-op UPDATE is intentional.  Once it succeeds, finalisation cannot
    revoke the lease until this transaction commits or rolls back.
    """
    db.rollback()
    updated = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.processing_token == token,
        Conversation.status == "recording",
    ).update(
        {"processing_token": token},
        synchronize_session=False,
    )
    if updated != 1:
        db.rollback()
        return False
    return True


def begin_finalization(
    db: Session,
    conversation_id: int,
    token: str,
) -> bool:
    """Revoke a streaming token while retaining the write lock until commit."""
    db.rollback()
    updated = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.processing_token == token,
    ).update(
        {"processing_token": None, "status": "finalizing"},
        synchronize_session=False,
    )
    if updated != 1:
        db.rollback()
        return False
    # Do not commit here.  The caller computes and publishes final metadata in
    # this same write transaction, closing the delete/late-writer gap.
    return True


def finish_processing_lease(
    db: Session,
    conversation_id: int,
    token: str,
    *,
    status: str,
    values: Mapping[str, Any] | None = None,
) -> bool:
    """Commit pending work and release its lease only for the current owner."""
    updates: dict[str, Any] = dict(values or {})
    updates.update({"processing_token": None, "status": status})
    updated = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.processing_token == token,
    ).update(updates, synchronize_session=False)
    if updated != 1:
        db.rollback()
        return False
    db.commit()
    return True


def fail_processing_lease(
    db: Session,
    conversation_id: int,
    token: str,
    *,
    status: str = "failed",
) -> bool:
    """Rollback partial work, then fail/release only the matching owner."""
    db.rollback()
    return finish_processing_lease(
        db,
        conversation_id,
        token,
        status=status,
    )
