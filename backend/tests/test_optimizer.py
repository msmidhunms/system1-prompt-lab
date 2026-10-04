"""The optimizer loop end to end, with a stub Laya and a scripted LLM."""

import json

from fastapi.testclient import TestClient

from app import karpathy_loop, llm, versions
from app.database import SessionLocal
from app.main import app
from app.tasks.registry import get_task
from conftest import wait_for_background_work

client = TestClient(app)
NEWS = get_task("news_topic")
LLM_CONFIG = {"provider": "stub", "kind": "stub", "model": "scripted"}

# Each article is one code word. The stub Laya only gets it right when the label's description names that word.
CODE = {"world": "wcode", "sports": "scode", "business": "bcode", "sci_tech": "tcode"}
EXAMPLES = [{"text": f"{code} {n}", "label": label} for label, code in CODE.items() for n in range(30)]


def proposal(*labels):
    """A reply that names the code word for the given labels and leaves the others undescriptive."""
    return json.dumps({
        "hypothesis": f"describe {', '.join(labels)}",
        "state_template": {"article": "{article}"},
        "instructions": "What is `article` about?",
        "criteria": {label: (CODE[label] if label in labels else "nothing") for label in NEWS.labels},
    })


def run_loop(monkeypatch, router, replies, loops):
    replies = iter(replies)

    def complete(config, system, user):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete", complete)
    started = karpathy_loop.start_run(
        task=NEWS, examples=EXAMPLES, llm_config=LLM_CONFIG, loops=loops, sample_size=60, seed=1,
        start_config=NEWS.default_config, start_version="v1_baseline", metric="balanced", calibrate=False,
        use_serp=False, laya_model="auto",
    )
    wait_for_background_work()
    return karpathy_loop.get_run(started["run_id"])


def auto_versions(run_id):
    db = SessionLocal()
    try:
        return [v for v in versions.list_versions(db, NEWS) if v["run_id"] == run_id and v["source"] == "auto"]
    finally:
        db.close()


def test_each_better_prompt_replaces_the_runs_saved_version(monkeypatch, router):
    run = run_loop(monkeypatch, router, [
        proposal("world", "sports"),
        proposal("world"),                                  # worse: discarded
        proposal("world", "sports", "business", "sci_tech"),
    ], loops=3)

    assert run["status"] == "completed" and run["improved"]
    assert [it["status"] for it in run["iterations"]] == ["keep", "discard", "keep"]
    assert run["best_iteration"] == 3 and run["best_accuracy"] == 1.0

    saved = auto_versions(run["run_id"])
    assert len(saved) == 1, "one auto-saved version per run, overwritten by each better prompt"
    assert saved[0]["version"] == run["auto_version"]
    assert saved[0]["iteration"] == 3 and saved[0]["accuracy"] == 1.0
    assert saved[0]["holdout_accuracy"] == run["holdout"]["best"]["accuracy"] == 1.0
    assert saved[0]["config"]["criteria"]["business"] == "bcode"


def test_a_run_whose_llm_keeps_failing_still_keeps_and_scores_what_it_found(monkeypatch, router):
    quota = llm.LLMError("HTTP 429: quota exceeded")
    run = run_loop(monkeypatch, router, [proposal("world", "sports"), quota, quota, quota, proposal("world")], loops=6)

    assert run["status"] == "failed" and "quota exceeded" in run["error"]
    assert [it["status"] for it in run["iterations"]] == ["keep", "crash", "crash", "crash"]
    assert run["improved"] and run["holdout"] is not None
    assert run["holdout"]["best"]["accuracy"] > run["holdout"]["baseline"]["accuracy"]
    assert len(auto_versions(run["run_id"])) == 1


def test_any_round_can_be_saved_and_the_run_lists_its_versions(monkeypatch, router):
    run = run_loop(monkeypatch, router, [proposal("world", "sports"), proposal("world")], loops=2)
    auto = run["auto_version"]

    saved = client.post("/api/save-model", json={"model_name": "discarded_round", "run_id": run["run_id"], "iteration": 2}).json()
    assert saved["source"] == "optimizer" and saved["iteration"] == 2
    assert saved["config"]["criteria"]["sports"] == "nothing"
    assert client.post("/api/save-model", json={"model_name": "x", "run_id": run["run_id"], "iteration": 9}).status_code == 400

    # The auto-saved version can be renamed, and the run still finds it.
    assert client.patch(f"/api/models/{auto}", json={"name": "news_best"}).status_code == 200
    detail = client.get(f"/api/karpathy-loop/runs/{run['run_id']}").json()
    assert detail["saved_versions"] == [
        {"version": "news_best", "source": "auto", "iteration": 1},
        {"version": "discarded_round", "source": "optimizer", "iteration": 2},
    ]


def test_a_run_with_no_better_prompt_saves_nothing(monkeypatch, router):
    run = run_loop(monkeypatch, router, [proposal()], loops=1)
    assert run["status"] == "completed" and not run["improved"] and run["auto_version"] is None
    assert auto_versions(run["run_id"]) == []
