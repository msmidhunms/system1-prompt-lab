#!/usr/bin/env python3
"""Apply the reviewed intent labels in data/intent_labels.csv to the golden dataset.

The labels live in two places and both are updated:
- data/test_db.json: search_intent_info.main_intent / foreign_intent of each row
- the predictions table (version v1_test_data): predicted_intent / secondary_intents

    python backend/scripts/apply_intent_labels.py                      # apply the reviewed labels
    python backend/scripts/apply_intent_labels.py --restore-original   # put the provider's labels back
    python backend/scripts/apply_intent_labels.py --dry-run            # report what would change
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.database import SessionLocal, PredictionRecord
from app.config import DATA_DIR
from app.tasks.intent.labels import SearchIntent

LABELS_FILE = DATA_DIR / "intent_labels.csv"
TEST_DB_FILE = DATA_DIR / "test_db.json"

# Only this fragment of each line is rewritten, so every other byte of the file is left as it was.
INTENT_FRAGMENT = re.compile(r'"search_intent_info":\{"main_intent":"[a-z]+","foreign_intent":\[[^\]]*\]')


def load_labels(restore_original: bool) -> dict:
    """Read {keyword: (main_intent, [secondary intents])} from the labels file."""
    main_col = "original_main_intent" if restore_original else "main_intent"
    secondary_col = "original_secondary_intents" if restore_original else "secondary_intents"
    valid = set(SearchIntent.all_labels())
    labels = {}
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            main = row[main_col]
            secondary = [s for s in row[secondary_col].split("|") if s]
            if main not in valid or any(s not in valid for s in secondary) or main in secondary:
                raise ValueError(f"Invalid labels for '{row['keyword']}': {main} / {secondary}")
            labels[row["keyword"]] = (main, secondary)
    return labels


def update_json_file(path: Path, labels: dict, dry_run: bool) -> int:
    """Rewrite the intent fields in the JSONL file. Returns the number of rows changed."""
    changed = 0
    new_lines = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line)
            main, secondary = labels[row["keyword"]]
            fragment = (
                '"search_intent_info":{"main_intent":' + json.dumps(main)
                + ',"foreign_intent":' + json.dumps(secondary, separators=(",", ":"))
            )
            new_line, count = INTENT_FRAGMENT.subn(lambda _: fragment, line)
            if count != 1:
                raise ValueError(f"Line {line_num}: expected one search_intent_info fragment, found {count}")

            # The rewritten line must differ from the original only in the two intent fields.
            new_row = json.loads(new_line)
            expected = json.loads(line)
            expected["search_intent_info"]["main_intent"] = main
            expected["search_intent_info"]["foreign_intent"] = secondary
            if new_row != expected:
                raise ValueError(f"Line {line_num}: rewrite changed more than the intent fields")

            changed += new_line != line
            new_lines.append(new_line)

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        tmp.replace(path)
    return changed


def update_database(labels: dict, dry_run: bool) -> int:
    """Update the golden rows in the predictions table. Returns the number of rows changed."""
    db = SessionLocal()
    changed = 0
    try:
        records = db.query(PredictionRecord).filter(PredictionRecord.version == "v1_test_data").all()
        missing = [r.query for r in records if r.query not in labels]
        if missing:
            raise ValueError(f"{len(missing)} golden rows have no entry in {LABELS_FILE.name}, e.g. '{missing[0]}'")
        for record in records:
            main, secondary = labels[record.query]
            # Same convention as import_test_data.py: no secondary intents is stored as None.
            new_secondary = secondary if secondary else None
            if record.predicted_intent != main or (record.secondary_intents or None) != new_secondary:
                record.predicted_intent = main
                record.secondary_intents = new_secondary
                changed += 1
        if dry_run:
            db.rollback()
        else:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--restore-original", action="store_true", help="apply the original_* columns instead")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    args = parser.parse_args()

    if not LABELS_FILE.exists():
        print(f"Error: {LABELS_FILE} not found")
        return

    labels = load_labels(args.restore_original)
    which = "original" if args.restore_original else "reviewed"
    print(f"Loaded {len(labels)} {which} labels from {LABELS_FILE}")

    if TEST_DB_FILE.exists():
        file_changed = update_json_file(TEST_DB_FILE, labels, args.dry_run)
        print(f"   {TEST_DB_FILE.name}: {file_changed} rows changed")
    else:
        print(f"   {TEST_DB_FILE.name}: not found, skipped")

    db_changed = update_database(labels, args.dry_run)
    print(f"   predictions table: {db_changed} rows changed")

    if args.dry_run:
        print("\nDry run: nothing was written")
    else:
        print(f"\n✅ Applied the {which} labels")


if __name__ == "__main__":
    main()
