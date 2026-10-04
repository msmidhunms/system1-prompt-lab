#!/usr/bin/env python3
"""Import an example's golden dataset.

The new examples download a public dataset from Hugging Face (once: the sample is kept
under data/tasks/ and reused). Search intent is read from data/test_db.json. Inputs
that are already golden rows are left untouched.

    python backend/scripts/import_dataset.py --task all
    python backend/scripts/import_dataset.py --task support_routing --rows 3000
    python backend/scripts/import_dataset.py --task news_topic --refresh
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app import golden
from app.database import SessionLocal
from app.tasks import sources
from app.tasks.registry import TASKS, get_task


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, help=f"one of: all, {', '.join(TASKS)}")
    parser.add_argument("--rows", type=int, default=sources.DEFAULT_ROWS, help="rows to sample (default %(default)s)")
    parser.add_argument("--seed", type=int, default=sources.DEFAULT_SEED, help="sampling seed (default %(default)s)")
    parser.add_argument("--refresh", action="store_true", help="download again instead of reusing data/tasks/")
    args = parser.parse_args()

    tasks = [t for t in TASKS.values() if sources.importable(t)] if args.task == "all" else [get_task(args.task)]
    db = SessionLocal()
    try:
        for task in tasks:
            print(f"{task.id}: importing...")
            result = sources.import_task(db, task, rows=args.rows, seed=args.seed, refresh=args.refresh)
            by_label = golden.stats(db, task)["by_label"]
            print(f"   {result['created']} rows added, {result['already_present']} already there, "
                  f"{result['total_golden_rows']} golden rows in total")
            print("   " + ", ".join(f"{label} {count}" for label, count in by_label.items()))
    finally:
        db.close()


if __name__ == "__main__":
    main()
