"""SERP context for golden queries, read from data/test_db.json.

Only queries in the imported dataset have SERP results. Any other query (e.g. a
live /predict call) gets an empty list, and SERP placeholders render as "".
"""

import json
import threading
from typing import Dict, List

from app.config import DATA_DIR

SERP_FILE = DATA_DIR / "test_db.json"
MAX_RESULTS = 8
SNIPPET_CHARS = 160

_index: Dict[str, List[Dict[str, str]]] = {}
_loaded = False
_lock = threading.Lock()


def available() -> bool:
    return SERP_FILE.exists()


def _load() -> None:
    global _loaded
    with _lock:
        if _loaded:
            return
        if available():
            with open(SERP_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    items = (row.get("serp_info") or {}).get("serp") or []
                    _index[row.get("keyword")] = [
                        {
                            "site": (item.get("domain") or "").removeprefix("www."),
                            "title": item.get("title") or "",
                            "snippet": (item.get("description") or "")[:SNIPPET_CHARS],
                        }
                        for item in items[:MAX_RESULTS]
                    ]
        _loaded = True


def lookup(query: str) -> List[Dict[str, str]]:
    """Top organic results for a query, best first. Empty when the query has no stored SERP."""
    if not _loaded:
        _load()
    return _index.get(query, [])
