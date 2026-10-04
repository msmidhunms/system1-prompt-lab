#!/usr/bin/env python3
"""Import data/test_db.json as the golden dataset of the search-intent example.

Same as `python backend/scripts/import_dataset.py --task search_intent`.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.config import DEFAULT_TASK_ID
from app.database import SessionLocal
from app.tasks import sources
from app.tasks.registry import get_task


def import_test_data():
    """Read test_db.json and add its keywords, intents and languages to the golden dataset."""
    db = SessionLocal()
    try:
        result = sources.import_task(db, get_task(DEFAULT_TASK_ID))
    except ValueError as e:
        print(f"Error: {e}")
        return
    finally:
        db.close()

    print("\n✅ Import completed!")
    print(f"   Imported: {result['created']} rows")
    print(f"   Already present: {result['already_present']} rows")
    print(f"   Golden rows in total: {result['total_golden_rows']}")


if __name__ == "__main__":
    import_test_data()
