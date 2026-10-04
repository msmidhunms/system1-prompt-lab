import pytest
from fastapi.testclient import TestClient

from app import versions
from app.config import CHECKPOINTS_DIR
from app.database import EvaluationRun, ModelVersion
from app.main import app
from app.tasks.registry import get_task

client = TestClient(app)
SUPPORT = get_task("support_routing")


def config_with(**criteria):
    return {**SUPPORT.default_config, "criteria": {**SUPPORT.default_config["criteria"], **criteria}}


def test_create_lists_and_loads(db):
    saved = versions.create(db, SUPPORT, "support_v2", config_with(card="anything about the card"), base_version="v1_baseline")
    assert saved["source"] == "manual" and saved["warnings"] == []
    assert versions.load_config("support_v2", SUPPORT)["criteria"]["card"] == "anything about the card"
    assert [v["version"] for v in versions.list_versions(db, SUPPORT)] == ["v1_baseline", "support_v2"]
    # A version belongs to one example.
    assert [v["version"] for v in versions.list_versions(db, get_task("news_topic"))] == ["v1_baseline"]
    with pytest.raises(ValueError, match="saved for the example"):
        versions.load_config("support_v2", get_task("news_topic"))


def test_create_rejects_bad_names_and_configs(db):
    versions.create(db, SUPPORT, "taken", SUPPORT.default_config)
    for name in ("taken", "v1_baseline", "has space", ""):
        with pytest.raises(ValueError):
            versions.create(db, SUPPORT, name, SUPPORT.default_config)
    with pytest.raises(ValueError, match="criteria"):
        versions.create(db, SUPPORT, "broken", {**SUPPORT.default_config, "criteria": {"card": "only one"}})
    assert not (CHECKPOINTS_DIR / "broken.json").exists()


def test_a_hand_written_version_keeps_its_label_bias(db):
    versions.create(db, SUPPORT, "biased", {**SUPPORT.default_config, "label_bias": {"card": 0.5}})
    assert versions.load_config("biased", SUPPORT)["label_bias"]["card"] == 0.5
    with pytest.raises(ValueError, match="label_bias"):
        versions.create(db, SUPPORT, "wrong_bias", {**SUPPORT.default_config, "label_bias": {"nope": 1}})


def test_rename_moves_the_file_and_the_references(db):
    versions.create(db, SUPPORT, "old_name", SUPPORT.default_config)
    versions.create(db, SUPPORT, "child", SUPPORT.default_config, base_version="old_name")
    db.add(EvaluationRun(id="e1", task_id=SUPPORT.id, model="laya", version="old_name", sample_size=1, total_cases=1,
                         accuracy=1.0, macro_f1=1.0, result={}, status="completed"))
    db.commit()

    renamed = versions.rename(db, "old_name", "new_name")
    assert renamed["version"] == "new_name" and renamed["latest_evaluation"]["eval_id"] == "e1"
    assert (CHECKPOINTS_DIR / "new_name.json").exists() and not (CHECKPOINTS_DIR / "old_name.json").exists()
    assert db.query(ModelVersion).filter(ModelVersion.version == "child").one().base_version == "new_name"
    assert versions.load_config("new_name", SUPPORT) == versions.load_config("child", SUPPORT)
    with pytest.raises(ValueError):
        versions.rename(db, "new_name", "child")
    with pytest.raises(ValueError):
        versions.rename(db, "v1_baseline", "something")


def test_delete(db):
    versions.create(db, SUPPORT, "gone", SUPPORT.default_config)
    versions.delete(db, "gone")
    assert not (CHECKPOINTS_DIR / "gone.json").exists()
    with pytest.raises(LookupError):
        versions.delete(db, "gone")


def test_models_api_duplicate_rename_delete():
    body = {"task": "support_routing", "name": "copy_1", "config": config_with(card="the card"), "base_version": "v1_baseline"}
    assert client.post("/api/models", json=body).json()["version"] == "copy_1"
    assert client.post("/api/models", json=body).status_code == 400
    assert client.post("/api/models", json={**body, "name": "bad", "config": {"instructions": "x"}}).status_code == 400

    assert client.patch("/api/models/copy_1", json={"name": "copy_renamed"}).json()["version"] == "copy_renamed"
    assert client.patch("/api/models/copy_1", json={"name": "x"}).status_code == 404
    listed = client.get("/api/models", params={"task": "support_routing"}).json()
    assert [(m["version"], m["source"]) for m in listed] == [("v1_baseline", "baseline"), ("copy_renamed", "manual")]

    assert client.delete("/api/models/copy_renamed").status_code == 200
    assert client.delete("/api/models/v1_baseline").status_code == 400
