#!/usr/bin/env python3
"""One-off migration to the multi-example layout.

Before there were several examples, the golden dataset of search intent lived in the
predictions table (version v1_test_data) and user feedback was laid over it at read
time. This moves it into golden_rows:

1. back up the database to data.db.bak-pre-tasks (once);
2. copy the golden rows into golden_rows as the search_intent example, with the kind
   of query and the provider's original labels from data/intent_labels.csv as metadata;
3. apply the recorded feedback on top, oldest first;
4. check every golden row arrived, then delete the old copies from predictions.

Live predictions stay in the predictions table. Running it again changes nothing.

    python backend/scripts/migrate_to_tasks.py
    python backend/scripts/migrate_to_tasks.py --dry-run
"""

import argparse
import csv
import os
import shutil
import sys
import uuid
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.config import DATA_DIR, DB_PATH, DEFAULT_TASK_ID
from app.database import FeedbackRecord, GoldenRow, PredictionRecord, SessionLocal
from app.golden import input_key, upsert_row
from app.tasks.registry import get_task

LEGACY_VERSION = "v1_test_data"
LABELS_FILE = DATA_DIR / "intent_labels.csv"


def load_meta() -> dict:
    """{keyword: {"kind", "original_label", "original_secondary_labels", "original_language"}} from the review file."""
    if not LABELS_FILE.exists():
        return {}
    meta = {}
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["keyword"]] = {
                "kind": row["kind"],
                "original_label": row["original_main_intent"],
                "original_secondary_labels": [s for s in row["original_secondary_intents"].split("|") if s],
                "original_language": row["original_language"],
            }
    return meta


def migrate(dry_run: bool) -> None:
    task = get_task(DEFAULT_TASK_ID)
    backup = DB_PATH.with_name(DB_PATH.name + ".bak-pre-tasks")
    if not dry_run and not backup.exists():
        shutil.copy2(DB_PATH, backup)
        print(f"Backed up the database to {backup}")

    db = SessionLocal()
    try:
        legacy = db.query(PredictionRecord).filter(PredictionRecord.version == LEGACY_VERSION).all()
        if not legacy:
            print("No golden rows left in the predictions table: nothing to migrate")
            return

        meta = load_meta()
        existing = {key for (key,) in db.query(GoldenRow.input_key).filter(GoldenRow.task_id == task.id)}
        copied = 0
        for record in legacy:
            key = input_key(record.query)
            if key in existing:
                continue
            existing.add(key)
            db.add(GoldenRow(
                id=str(uuid.uuid4()),
                task_id=task.id,
                input_text=record.query,
                input_key=key,
                label=record.predicted_intent,
                secondary_labels=record.secondary_intents or None,
                language=record.language,
                source="imported",
                meta=meta.get(record.query),
                created_at=record.created_at,
            ))
            copied += 1
        db.flush()
        print(f"Copied {copied} of {len(legacy)} golden rows into golden_rows")

        # What the golden dataset used to do at read time: feedback that settles the label overrides it.
        applied = 0
        feedbacks = db.query(FeedbackRecord).filter(FeedbackRecord.task_id == task.id).order_by(FeedbackRecord.created_at).all()
        for feedback in feedbacks:
            label = feedback.predicted_intent if feedback.feedback_type == "correct" else feedback.corrected_intent
            if label in task.labels:
                upsert_row(db, task, feedback.query, label, source="feedback")
                applied += 1
        db.flush()
        print(f"Applied {applied} of {len(feedbacks)} feedback entries")

        missing = [r.query for r in legacy if input_key(r.query) not in
                   {key for (key,) in db.query(GoldenRow.input_key).filter(GoldenRow.task_id == task.id)}]
        if missing:
            raise RuntimeError(f"{len(missing)} golden rows did not arrive, e.g. '{missing[0]}'. Nothing was changed.")

        deleted = db.query(PredictionRecord).filter(PredictionRecord.version == LEGACY_VERSION).delete()
        print(f"Removed the {deleted} old copies from the predictions table "
              f"({db.query(PredictionRecord).count()} live predictions kept)")

        if dry_run:
            db.rollback()
            print("\nDry run: nothing was written")
        else:
            db.commit()
            print(f"\n✅ Migrated. golden_rows now holds "
                  f"{db.query(GoldenRow).filter(GoldenRow.task_id == task.id).count()} search-intent rows")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    migrate(parser.parse_args().dry_run)
