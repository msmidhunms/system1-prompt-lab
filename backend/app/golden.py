"""The golden dataset of each example: labelled inputs the evaluation and the Karpathy loop score against.

A row is identified by its task and its input. Two inputs that differ only in case or
spacing are the same row, so adding one of them again updates the label instead of
creating a near-duplicate.
"""

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import GoldenRow
from app.laya_inference import guess_language
from app.tasks.registry import Task

SOURCES = ("imported", "manual", "feedback")
_UNSET = object()


def input_key(text: str) -> str:
    """What identifies an input within a task: its text with case and spacing normalised, hashed."""
    normalised = " ".join(text.split()).casefold()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _check_labels(task: Task, label: str, secondary_labels: Optional[List[str]]) -> Optional[List[str]]:
    if label not in task.labels:
        raise ValueError(f"'{label}' is not a label of this example. Choose one of: {', '.join(task.labels)}")
    secondary = list(dict.fromkeys(secondary_labels or []))
    unknown = [s for s in secondary if s not in task.labels]
    if unknown:
        raise ValueError(f"Unknown secondary labels: {', '.join(unknown)}")
    if label in secondary:
        raise ValueError("The main label cannot also be a secondary label")
    return secondary or None


def find_row(db: Session, task: Task, text: str) -> Optional[GoldenRow]:
    return db.query(GoldenRow).filter(
        GoldenRow.task_id == task.id, GoldenRow.input_key == input_key(text)
    ).first()


def upsert_row(
    db: Session,
    task: Task,
    text: str,
    label: str,
    secondary_labels: Any = _UNSET,
    language: Any = _UNSET,
    source: str = "manual",
    meta: Any = _UNSET,
) -> Tuple[GoldenRow, bool, Optional[str]]:
    """Add a labelled input, or update the row that already holds this input.

    Fields left out keep their value on an existing row. Returns (row, created,
    previous label or None). The caller commits.
    """
    text = text.strip()
    if not text:
        raise ValueError("The input text cannot be empty")
    secondary = _check_labels(task, label, None if secondary_labels is _UNSET else secondary_labels)

    row = find_row(db, task, text)
    if row is not None:
        previous = row.label
        row.label = label
        if secondary_labels is not _UNSET:
            row.secondary_labels = secondary
        elif row.secondary_labels and label in row.secondary_labels:
            row.secondary_labels = [s for s in row.secondary_labels if s != label] or None
        if language is not _UNSET and language:
            row.language = language
        if meta is not _UNSET:
            row.meta = meta
        return row, False, previous

    if task.language and (language is _UNSET or not language):
        # No reviewed language for a new row, so this is a guess.
        language = guess_language(text)
    row = GoldenRow(
        id=str(uuid.uuid4()),
        task_id=task.id,
        input_text=text,
        input_key=input_key(text),
        label=label,
        secondary_labels=secondary,
        language=None if language is _UNSET else (language or None),
        source=source,
        meta=None if meta is _UNSET else meta,
    )
    db.add(row)
    return row, True, None


def update_row(
    db: Session,
    task: Task,
    row: GoldenRow,
    text: Optional[str] = None,
    label: Optional[str] = None,
    secondary_labels: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> GoldenRow:
    """Edit a row in place. Raises LookupError when the new text is already another row's input."""
    if text is not None:
        text = text.strip()
        if not text:
            raise ValueError("The input text cannot be empty")
        other = find_row(db, task, text)
        if other is not None and other.id != row.id:
            raise LookupError("Another row already has this input")
        row.input_text = text
        row.input_key = input_key(text)
    new_label = label if label is not None else row.label
    new_secondary = secondary_labels if secondary_labels is not None else (row.secondary_labels or [])
    if label is not None and secondary_labels is None:
        new_secondary = [s for s in new_secondary if s != label]
    row.secondary_labels = _check_labels(task, new_label, new_secondary)
    row.label = new_label
    if language is not None:
        row.language = language.strip() or None
    return row


def serialise(row: GoldenRow) -> Dict[str, Any]:
    return {
        "id": row.id,
        "task": row.task_id,
        "text": row.input_text,
        "label": row.label,
        "secondary_labels": row.secondary_labels or [],
        "language": row.language,
        "source": row.source,
        "meta": row.meta or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def list_rows(
    db: Session,
    task: Task,
    page: int = 1,
    page_size: int = 20,
    q: Optional[str] = None,
    label: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    query = db.query(GoldenRow).filter(GoldenRow.task_id == task.id)
    if q and q.strip():
        # autoescape: % and _ in the search text match themselves.
        query = query.filter(GoldenRow.input_text.icontains(q.strip(), autoescape=True))
    if label:
        query = query.filter(GoldenRow.label == label)
    if source:
        query = query.filter(GoldenRow.source == source)
    total = query.count()
    rows = (
        query.order_by(GoldenRow.updated_at.desc(), GoldenRow.id)
        .offset((page - 1) * page_size).limit(page_size).all()
    )
    return {
        "items": [serialise(row) for row in rows],
        "pagination": {
            "current_page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }


def _usable(task: Task):
    """Filter for the rows the evaluation and the loop use: the task's language only, when it has one."""
    conditions = [GoldenRow.task_id == task.id, GoldenRow.label.in_(task.labels)]
    if task.language:
        conditions.append(GoldenRow.language == task.language)
    return conditions


def stats(db: Session, task: Task) -> Dict[str, Any]:
    def counts(column) -> Dict[str, int]:
        rows = db.query(column, func.count(GoldenRow.id)).filter(GoldenRow.task_id == task.id).group_by(column).all()
        return {key or "unknown": count for key, count in rows}

    by_label = counts(GoldenRow.label)
    total = sum(by_label.values())
    usable = db.query(func.count(GoldenRow.id)).filter(*_usable(task)).scalar() or 0
    return {
        "task": task.id,
        "total": total,
        "by_label": {label: by_label.get(label, 0) for label in task.labels},
        "by_source": counts(GoldenRow.source),
        "by_language": counts(GoldenRow.language) if task.language else {},
        "eval_language": task.language,
        "usable": usable,
        "excluded_other_language": total - usable,
    }


def examples_for(db: Session, task: Task) -> List[Dict[str, str]]:
    """The labelled inputs the evaluation and the loop score against, as {"text", "label"}."""
    rows = db.query(GoldenRow.input_text, GoldenRow.label).filter(*_usable(task)).all()
    return [{"text": text, "label": label} for text, label in rows]
