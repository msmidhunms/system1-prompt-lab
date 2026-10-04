"""Test setup: a throwaway database and a stub in place of the Laya model.

DB_PATH has to be set before anything imports app.config, so it is set here at import time.
"""

import json
import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="system1-tests-")
os.environ["DB_PATH"] = str(Path(_tmp) / "test.db")

import pytest  # noqa: E402

from app import laya_inference  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402


class StubRouter:
    """Answers from the text alone: a choice label named in the state wins, a yes/no is yes when the state says YES."""

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
                hit = next((label for label in labels if label in state), labels[0])
                rest = 0.1 / (len(labels) - 1)
                answer = {"type": "choice", "probabilities": {label: 0.9 if label == hit else rest for label in labels}}
            results.append({"answers": {qid: answer}})
        return results


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


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
