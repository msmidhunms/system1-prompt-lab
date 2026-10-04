#!/usr/bin/env python3
"""Apply the reviewed intent labels and query languages in data/intent_labels.csv to the golden dataset.

The values live in two places and both are updated:
- data/test_db.json: search_intent_info.main_intent / foreign_intent and extra.detected_language of each row
- the golden_rows table (search_intent example): label / secondary_labels / language

    python backend/scripts/apply_intent_labels.py                      # apply the reviewed values
    python backend/scripts/apply_intent_labels.py --restore-original   # put the provider's values back
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

from app.database import SessionLocal, GoldenRow
from app.config import DATA_DIR, DEFAULT_TASK_ID
from app.tasks.intent.labels import SearchIntent

LABELS_FILE = DATA_DIR / "intent_labels.csv"
TEST_DB_FILE = DATA_DIR / "test_db.json"

# Only these fragments of each line are rewritten, so every other byte of the file is left as it was.
INTENT_FRAGMENT = re.compile(r'"search_intent_info":\{"main_intent":"[a-z]+","foreign_intent":\[[^\]]*\]')
LANGUAGE_FRAGMENT = re.compile(r'"detected_language":"[a-z]+"')
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")


def load_labels(restore_original: bool) -> dict:
    """Read {keyword: (main_intent, [secondary intents], language)} from the labels file."""
    prefix = "original_" if restore_original else ""
    valid = set(SearchIntent.all_labels())
    labels = {}
    with open(LABELS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            main = row[prefix + "main_intent"]
            secondary = [s for s in row[prefix + "secondary_intents"].split("|") if s]
            language = row[prefix + "language"]
            if main not in valid or any(s not in valid for s in secondary) or main in secondary:
                raise ValueError(f"Invalid labels for '{row['keyword']}': {main} / {secondary}")
            if not LANGUAGE_CODE.match(language):
                raise ValueError(f"Invalid language for '{row['keyword']}': {language}")
            labels[row["keyword"]] = (main, secondary, language)
    return labels


def update_json_file(path: Path, labels: dict, dry_run: bool) -> int:
    """Rewrite the intent and language fields in the JSONL file. Returns the number of rows changed."""
    changed = 0
    new_lines = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line)
            main, secondary, language = labels[row["keyword"]]
            intent_fragment = (
                '"search_intent_info":{"main_intent":' + json.dumps(main)
                + ',"foreign_intent":' + json.dumps(secondary, separators=(",", ":"))
            )
            new_line, intent_count = INTENT_FRAGMENT.subn(lambda _: intent_fragment, line)
            new_line, language_count = LANGUAGE_FRAGMENT.subn(
                lambda _: '"detected_language":' + json.dumps(language), new_line
            )
            if intent_count != 1 or language_count != 1:
                raise ValueError(
                    f"Line {line_num}: expected one intent and one language fragment, "
                    f"found {intent_count} and {language_count}"
                )

            # The rewritten line must differ from the original only in the reviewed fields.
            new_row = json.loads(new_line)
            expected = json.loads(line)
            expected["search_intent_info"]["main_intent"] = main
            expected["search_intent_info"]["foreign_intent"] = secondary
            expected["extra"]["detected_language"] = language
            if new_row != expected:
                raise ValueError(f"Line {line_num}: rewrite changed more than the reviewed fields")

            changed += new_line != line
            new_lines.append(new_line)

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        tmp.replace(path)
    return changed


def update_database(labels: dict, dry_run: bool) -> int:
    """Update the imported search-intent rows in the golden_rows table. Returns the number of rows changed.

    Rows added by hand or through feedback are not in the review file and are left alone.
    """
    db = SessionLocal()
    changed = 0
    try:
        records = db.query(GoldenRow).filter(
            GoldenRow.task_id == DEFAULT_TASK_ID, GoldenRow.source == "imported"
        ).all()
        missing = [r.input_text for r in records if r.input_text not in labels]
        if missing:
            raise ValueError(f"{len(missing)} golden rows have no entry in {LABELS_FILE.name}, e.g. '{missing[0]}'")
        for record in records:
            main, secondary, language = labels[record.input_text]
            # Same convention as the import: no secondary intents is stored as None.
            new_secondary = secondary if secondary else None
            current = (record.label, record.secondary_labels or None, record.language)
            if current != (main, new_secondary, language):
                record.label = main
                record.secondary_labels = new_secondary
                record.language = language
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
    print(f"Loaded {len(labels)} {which} labels and languages from {LABELS_FILE}")

    if TEST_DB_FILE.exists():
        file_changed = update_json_file(TEST_DB_FILE, labels, args.dry_run)
        print(f"   {TEST_DB_FILE.name}: {file_changed} rows changed")
    else:
        print(f"   {TEST_DB_FILE.name}: not found, skipped")

    db_changed = update_database(labels, args.dry_run)
    print(f"   golden_rows table: {db_changed} rows changed")

    if args.dry_run:
        print("\nDry run: nothing was written")
    else:
        print(f"\n✅ Applied the {which} labels and languages")


if __name__ == "__main__":
    main()
