#!/usr/bin/env python3
"""Import test data from test_db.json into the database."""

import json
import sys
import uuid
from pathlib import Path
import os

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.database import SessionLocal, PredictionRecord
from app.config import DATA_DIR


def import_test_data():
    """Read test_db.json and import intent data into the database."""
    test_db_path = DATA_DIR / "test_db.json"

    if not test_db_path.exists():
        print(f"Error: {test_db_path} not found")
        return

    print(f"Reading data from {test_db_path}...")

    with open(test_db_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    db = SessionLocal()
    imported_count = 0
    skipped_count = 0

    try:
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())

                keyword = data.get('keyword')
                search_intent_info = data.get('search_intent_info', {})
                main_intent = search_intent_info.get('main_intent')
                foreign_intents = search_intent_info.get('foreign_intent', [])
                language = (data.get('extra') or {}).get('detected_language')

                if not keyword or not main_intent:
                    skipped_count += 1
                    continue

                # Check if already exists
                existing = db.query(PredictionRecord).filter(
                    PredictionRecord.query == keyword
                ).first()

                if existing:
                    skipped_count += 1
                    continue

                # Create prediction record with main and secondary intents
                record = PredictionRecord(
                    id=str(uuid.uuid4()),
                    query=keyword,
                    predicted_intent=main_intent,
                    secondary_intents=foreign_intents if foreign_intents else None,
                    language=language,
                    confidence=0.95,  # Default high confidence for golden dataset
                    model="serp_classifier",
                    version="v1_test_data"
                )

                db.add(record)
                imported_count += 1

                # Commit every 100 records
                if imported_count % 100 == 0:
                    db.commit()
                    print(f"Imported {imported_count} records...")

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                skipped_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1

        # Final commit
        db.commit()

        print(f"\n✅ Import completed!")
        print(f"   Imported: {imported_count} records")
        print(f"   Skipped: {skipped_count} records")
        print(f"   Total lines: {len(lines)}")

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import_test_data()
