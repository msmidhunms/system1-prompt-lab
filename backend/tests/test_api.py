from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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

    result = client.post("/api/evaluate", json={"task": "news_topic", "sample_size": 10}).json()
    assert result["total_cases"] == 3 and result["error_count"] == 1
    assert result["confusion_matrix"]["sports"]["world"] == 1

    history = client.get("/api/evaluations", params={"task": "news_topic"}).json()
    assert [h["eval_id"] for h in history] == [result["eval_id"]]
    assert client.get(f"/api/evaluations/{result['eval_id']}").json() == result
    assert client.get("/api/evaluations", params={"task": "toxicity"}).json() == []
    assert client.post("/api/evaluate", json={"task": "toxicity"}).status_code == 400
