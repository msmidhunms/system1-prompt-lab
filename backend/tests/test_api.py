from fastapi.testclient import TestClient

from app.main import app
from conftest import wait_for_background_work

client = TestClient(app)


def evaluate(**body):
    """Start an evaluation, wait for it and return its full result."""
    started = client.post("/api/evaluate", json=body).json()
    assert started["status"] == "running"
    wait_for_background_work()
    return client.get(f"/api/evaluations/{started['eval_id']}").json()


def test_tasks_lists_every_example_with_its_labels():
    tasks = {t["id"]: t for t in client.get("/api/tasks").json()["tasks"]}
    assert len(tasks) == 8
    assert tasks["search_intent"]["labels"] == ["informational", "navigational", "commercial", "transactional"]
    assert tasks["toxicity"]["question_type"] == "noul"
    assert client.get("/api/golden-data/stats", params={"task": "nope"}).status_code == 400


def test_adding_a_row_twice_updates_it():
    body = {"task": "support_routing", "text": "Where is my card?", "label": "card"}
    first = client.post("/api/golden-data", json=body).json()
    assert first["status"] == "created" and first["previous_label"] is None

    again = client.post("/api/golden-data", json={**body, "text": "where is  MY card?", "label": "account"}).json()
    assert again["status"] == "updated" and again["previous_label"] == "card"
    assert again["row"]["id"] == first["row"]["id"]

    page = client.get("/api/golden-data", params={"task": "support_routing"}).json()
    assert page["pagination"]["total_items"] == 1 and page["items"][0]["label"] == "account"
    # The same input under another example is its own row.
    assert client.get("/api/golden-data", params={"task": "news_topic"}).json()["pagination"]["total_items"] == 0


def test_edit_and_delete_a_row():
    first = client.post("/api/golden-data", json={"task": "support_routing", "text": "a", "label": "card"}).json()["row"]
    second = client.post("/api/golden-data", json={"task": "support_routing", "text": "b", "label": "card"}).json()["row"]

    assert client.put(f"/api/golden-data/{second['id']}", json={"text": "A"}).status_code == 409
    assert client.put(f"/api/golden-data/{second['id']}", json={"label": "nope"}).status_code == 400
    edited = client.put(f"/api/golden-data/{second['id']}", json={"label": "top_up"}).json()
    assert edited["label"] == "top_up" and edited["text"] == "b"

    assert client.delete(f"/api/golden-data/{first['id']}").status_code == 200
    assert client.delete(f"/api/golden-data/{first['id']}").status_code == 404


def test_feedback_writes_the_golden_row(router):
    prediction = client.post("/api/predict", json={"task": "news_topic", "text": "a sports story"}).json()
    assert prediction["predicted_label"] == "sports" and prediction["golden_label"] is None

    reply = client.post("/api/feedback", json={
        "task": "news_topic", "text": "a sports story", "predicted_label": "sports",
        "feedback_type": "incorrect", "corrected_label": "business",
    }).json()
    assert reply["golden_status"] == "created"
    assert client.post("/api/predict", json={"task": "news_topic", "text": "A sports story"}).json()["golden_label"] == "business"


def test_evaluations_are_saved_per_example(router):
    for text, label in [("sports one", "sports"), ("business two", "business"), ("world three", "sports")]:
        client.post("/api/golden-data", json={"task": "news_topic", "text": text, "label": label})
    assert client.post("/api/evaluate", json={"task": "news_topic", "sample_size": 5}).status_code == 400

    result = evaluate(task="news_topic", sample_size=10)
    assert result["status"] == "completed" and result["progress_done"] == result["progress_total"] == 3
    assert result["total_cases"] == 3 and result["error_count"] == 1
    assert result["confusion_matrix"]["sports"]["world"] == 1

    history = client.get("/api/evaluations", params={"task": "news_topic"}).json()
    assert [(h["eval_id"], h["status"], h["accuracy"]) for h in history] == [(result["eval_id"], "completed", result["accuracy"])]
    assert client.get("/api/evaluations", params={"task": "toxicity"}).json() == []
    assert client.post("/api/evaluate", json={"task": "toxicity"}).status_code == 400


def test_a_failed_evaluation_is_recorded(router, monkeypatch):
    client.post("/api/golden-data", json={"task": "news_topic", "text": "sports one", "label": "sports"})

    def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(router, "predict_batch", broken)
    result = evaluate(task="news_topic", sample_size=10)
    assert result["status"] == "failed" and result["error"] == "model unavailable" and result["accuracy"] is None


def test_evaluations_left_running_are_marked_interrupted(db):
    from app import evaluations
    from app.database import EvaluationRun

    db.add(EvaluationRun(id="stale", task_id="news_topic", model="laya", version="v1_baseline", sample_size=5,
                         total_cases=5, accuracy=0.0, macro_f1=0.0, result={}, status="running"))
    db.commit()
    evaluations.mark_interrupted()
    assert client.get("/api/evaluations/stale").json()["status"] == "interrupted"


def test_import_runs_in_the_background_and_reports_status(monkeypatch):
    from app.tasks import sources

    monkeypatch.setitem(sources.LOADERS, "news_topic", lambda size, seed: [{"text": "a sports story", "label": "sports"}])
    monkeypatch.setitem(sources.LOADERS, "toxicity", lambda size, seed: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(sources, "_import_state", {})

    assert client.post("/api/setup/import", json={"tasks": ["news_topic", "toxicity"]}).json()["queued"] == ["news_topic", "toxicity"]
    wait_for_background_work()
    tasks = {t["id"]: t for t in client.get("/api/tasks").json()["tasks"]}
    assert tasks["news_topic"]["golden_rows"] == 1 and tasks["news_topic"]["import_status"]["status"] == "done"
    assert tasks["toxicity"]["import_status"] == {"status": "failed", "message": "offline"}
    assert "test_db.json" in tasks["search_intent"]["setup_hint"]
    assert client.get("/api/activity").json() == {"loop": None, "evaluations": [], "imports": []}
