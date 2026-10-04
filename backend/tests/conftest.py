"""Test setup: a throwaway database and run folder, and a stub in place of the Laya model.

The paths have to be set before anything imports app.config, so they are set here at import time.
"""

import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="system1-tests-")
os.environ["DB_PATH"] = str(Path(_tmp) / "test.db")
os.environ["AUTORESEARCH_DIR"] = str(Path(_tmp) / "autoresearch")
os.environ["DATA_DIR"] = str(Path(_tmp) / "data")

import pytest  # noqa: E402

from app import laya_inference  # noqa: E402
from app.config import CHECKPOINTS_DIR, RUNS_DIR  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402


def _words(text) -> set:
    return set(re.findall(r"\w+", str(text).lower()))


class StubRouter:
    """A stand-in for Laya that answers by matching words.

    A choice goes to the label named in the state, else to the label whose description
    shares the most words with the state (so a better prompt scores better), else to the
    first label. A yes/no is yes when the state says YES.
    """

    def __init__(self):
        self.requests = []

    def predict_batch(self, requests, batch_size=32, sort_by_length=True):
        self.requests.extend(requests)
        results = []
        for request in requests:
            state = json.dumps(request["state"]) if not isinstance(request["state"], str) else request["state"]
            (qid, question), = request["questions"].items()
            if question["type"] == "noul":
                answer = {"type": "noul", "noul": 0.9 if "YES" in state else 0.2}
            else:
                labels = list(question["criteria"])
                hit = next((label for label in labels if label in state), None)
                if hit is None:
                    overlap = {label: len(_words(question["criteria"][label]) & _words(state)) for label in labels}
                    hit = max(labels, key=lambda label: overlap[label]) if any(overlap.values()) else labels[0]
                rest = 0.1 / (len(labels) - 1)
                answer = {"type": "choice", "probabilities": {label: 0.9 if label == hit else rest for label in labels}}
            results.append({"answers": {qid: answer}})
        return results


def wait_for_background_work(timeout: float = 30.0) -> None:
    """Let optimizer runs, evaluations and imports started by a test finish."""
    for thread in threading.enumerate():
        if thread.name.startswith(("karpathy-", "evaluation-", "dataset-import")):
            thread.join(timeout)


@pytest.fixture(autouse=True)
def clean_state():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    for folder in (CHECKPOINTS_DIR, RUNS_DIR):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    yield
    wait_for_background_work()


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def router(monkeypatch):
    stub = StubRouter()
    monkeypatch.setattr(laya_inference, "get_router", lambda: stub)
    return stub
