"""Saved model versions of an example.

A version is a named prompt config for one task. Its config and how it came about live
in autoresearch/checkpoints/<name>.json; the model_versions table is the index the
lists are built from. Names are unique across examples, because each is one file.

Where a version comes from (`source`):
- "auto": the best prompt of an optimizer run, saved by the run itself. One per run,
  overwritten each time the run finds a better prompt.
- "optimizer": a round of a run that the user saved by hand.
- "manual": written or edited by hand, usually from a copy of another version.
"""

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app import engines
from app.config import CHECKPOINTS_DIR, DEFAULT_TASK_ID
from app.database import EvaluationRun, ModelVersion, PredictionRecord
from app.laya_inference import validate_config
from app.tasks.registry import Task

BASELINE_VERSION = "v1_baseline"
SOURCES = ("auto", "optimizer", "manual")
VERSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def checkpoint_path(name: str):
    if not VERSION_NAME_RE.match(name):
        raise ValueError("version names may only contain letters, digits, '_', '-' and '.'")
    return CHECKPOINTS_DIR / f"{name}.json"


def _read(name: str) -> Optional[Dict[str, Any]]:
    path = checkpoint_path(name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(name: str, payload: Dict[str, Any]) -> None:
    path = checkpoint_path(name)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_config(version: Optional[str], task: Task) -> Dict[str, Any]:
    """Return the prompt config saved under a version name for this task.

    Raises ValueError for a version that is neither the baseline nor a saved version of the task.
    """
    if not version or version == BASELINE_VERSION:
        return task.default_config
    checkpoint = _read(version)
    if checkpoint is None:
        raise ValueError(f"No saved model version named '{version}'")
    # Checkpoints saved before there were several examples have no task and belong to the default one.
    saved_for = checkpoint.get("task") or DEFAULT_TASK_ID
    if saved_for != task.id:
        raise ValueError(f"Model version '{version}' was saved for the example '{saved_for}', not '{task.id}'")
    return checkpoint["config"]


def _exists(db: Session, name: str) -> bool:
    return (
        name == BASELINE_VERSION
        or checkpoint_path(name).exists()
        or db.query(ModelVersion).filter(ModelVersion.version == name).first() is not None
    )


def _check_new_name(db: Session, name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("The model name cannot be empty")
    checkpoint_path(name)  # validates the characters
    if _exists(db, name):
        raise ValueError(f"Model version '{name}' already exists")
    return name


def create(
    db: Session,
    task: Task,
    name: str,
    config: Dict[str, Any],
    source: str = "manual",
    base_version: Optional[str] = None,
    description: Optional[str] = None,
    run_id: Optional[str] = None,
    iteration: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    llm: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a prompt config as a new version. Returns (config warnings are in the result's "warnings")."""
    name = _check_new_name(db, name)
    config, warnings = validate_config(config, task, keep_bias=True)
    metrics = metrics or {}
    _write(name, {
        "version": name,
        "task": task.id,
        "config": config,
        "source": source,
        "run_id": run_id,
        "iteration": iteration,
        "base_version": base_version,
        "dev_accuracy": metrics.get("dev_accuracy"),
        "dev_macro_f1": metrics.get("dev_macro_f1"),
        "holdout_accuracy": metrics.get("holdout_accuracy"),
        "llm": llm,
        "saved_at": datetime.utcnow().isoformat(),
    })
    db.add(ModelVersion(
        id=str(uuid.uuid4()),
        task_id=task.id,
        model_name="laya",
        version=name,
        # The checkpoint file holds the metrics; this column predates it and cannot be empty.
        accuracy=metrics.get("dev_accuracy") or 0.0,
        base_version=base_version,
        description=description,
        source=source,
        run_id=run_id,
        iteration=iteration,
    ))
    db.commit()
    return {**_describe(db, db.query(ModelVersion).filter(ModelVersion.version == name).first()), "warnings": warnings}


def _iteration_entry(run: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    entry = next((it for it in run["iterations"] if it["iteration"] == iteration), None)
    if entry is None or not entry.get("config"):
        raise ValueError(f"Round {iteration} of this run has no prompt to save")
    return entry


def save_from_run(
    db: Session,
    task: Task,
    run: Dict[str, Any],
    name: str,
    iteration: Optional[int] = None,
    description: Optional[str] = None,
    source: str = "optimizer",
) -> Dict[str, Any]:
    """Save a run's best prompt, or the prompt one of its rounds proposed (kept or not), as a version."""
    if iteration is None or iteration == run["best_iteration"]:
        if run.get("best_accuracy") is None:
            raise ValueError("Run has no evaluated config to save")
        iteration = run["best_iteration"]
        config = run["best_config"]
        metrics = {
            "dev_accuracy": run["best_accuracy"],
            "dev_macro_f1": run.get("best_macro_f1"),
            "holdout_accuracy": run["holdout"]["best"]["accuracy"] if run.get("holdout") else None,
        }
    else:
        entry = _iteration_entry(run, iteration)
        config = entry["config"]
        metrics = {"dev_accuracy": entry.get("accuracy"), "dev_macro_f1": entry.get("macro_f1")}
    return create(
        db, task, name, config, source=source, base_version=run["start_version"], description=description,
        run_id=run["run_id"], iteration=iteration, metrics=metrics, llm=run.get("llm"),
    )


def _auto_name(db: Session, task: Task, run_id: str) -> str:
    # run ids look like 20261004-125234-14a2d7
    stamp = f"{run_id[4:8]}-{run_id[9:13]}" if re.match(r"^\d{8}-\d{6}", run_id) else run_id[:12]
    base = f"{task.id}_{stamp}"
    name, n = base, 1
    while _exists(db, name):
        n += 1
        name = f"{base}-{n}"
    return name


def auto_version_of(db: Session, run_id: str) -> Optional[ModelVersion]:
    return db.query(ModelVersion).filter(ModelVersion.run_id == run_id, ModelVersion.source == "auto").first()


def autosave_run_best(db: Session, task: Task, run: Dict[str, Any]) -> str:
    """Save the run's current best prompt as the run's one auto-saved version, replacing its earlier best.

    The version is found by the run it belongs to, not by name, so it can be renamed while the run goes on.
    """
    row = auto_version_of(db, run["run_id"])
    if row is None:
        saved = save_from_run(
            db, task, run, _auto_name(db, task, run["run_id"]), source="auto",
            description=f"Best prompt of optimizer run {run['run_id']} (saved automatically)",
        )
        return saved["version"]
    checkpoint = _read(row.version) or {"version": row.version, "task": task.id, "source": "auto", "run_id": run["run_id"]}
    checkpoint.update({
        "config": run["best_config"],
        "iteration": run["best_iteration"],
        "dev_accuracy": run["best_accuracy"],
        "dev_macro_f1": run.get("best_macro_f1"),
        "holdout_accuracy": run["holdout"]["best"]["accuracy"] if run.get("holdout") else None,
        "saved_at": datetime.utcnow().isoformat(),
    })
    _write(row.version, checkpoint)
    row.accuracy = run["best_accuracy"]
    row.iteration = run["best_iteration"]
    db.commit()
    return row.version


def _row(db: Session, name: str) -> ModelVersion:
    if name == BASELINE_VERSION:
        raise ValueError("The baseline is the example's built-in prompt and cannot be changed")
    row = db.query(ModelVersion).filter(ModelVersion.version == name).first()
    if row is None:
        raise LookupError(f"No saved model version named '{name}'")
    return row


def rename(db: Session, old: str, new: str) -> Dict[str, Any]:
    """Rename a version, and the references to it in saved evaluations, predictions and other versions."""
    row = _row(db, old)
    new = _check_new_name(db, new)
    checkpoint = _read(old)
    if checkpoint is not None:
        _write(new, {**checkpoint, "version": new})
        checkpoint_path(old).unlink()
    row.version = new
    task_id = row.task_id
    db.query(EvaluationRun).filter(EvaluationRun.task_id == task_id, EvaluationRun.version == old).update({"version": new})
    db.query(PredictionRecord).filter(PredictionRecord.task_id == task_id, PredictionRecord.version == old).update({"version": new})
    db.query(ModelVersion).filter(ModelVersion.task_id == task_id, ModelVersion.base_version == old).update({"base_version": new})
    db.commit()
    return _describe(db, row)


def set_description(db: Session, name: str, description: Optional[str]) -> Dict[str, Any]:
    row = _row(db, name)
    row.description = (description or "").strip() or None
    db.commit()
    return _describe(db, row)


def delete(db: Session, name: str) -> None:
    row = _row(db, name)
    path = checkpoint_path(name)
    if path.exists():
        path.unlink()
    db.delete(row)
    db.commit()


def _latest_evaluation(db: Session, task_id: str, version: str, config: Optional[Dict[str, Any]]) -> Optional[EvaluationRun]:
    """The version's latest completed evaluation on the model it was saved for.

    A comparison (or an evaluation on another model) runs the same prompt elsewhere; that is not the version's score.
    """
    own_engine = (config or {}).get("engine") or engines.LAYA
    return (
        db.query(EvaluationRun)
        .filter(
            EvaluationRun.task_id == task_id, EvaluationRun.version == version,
            EvaluationRun.status == "completed", EvaluationRun.engine == own_engine,
        )
        .order_by(EvaluationRun.created_at.desc()).first()
    )


def _evaluation_summary(run: Optional[EvaluationRun]) -> Optional[Dict[str, Any]]:
    if run is None:
        return None
    return {"eval_id": run.id, "accuracy": run.accuracy, "macro_f1": run.macro_f1, "sample_size": run.sample_size}


def _describe(db: Session, row: ModelVersion) -> Dict[str, Any]:
    checkpoint = _read(row.version) or {}
    return {
        "version": row.version,
        "task": row.task_id,
        "source": row.source or checkpoint.get("source") or "optimizer",
        "base_version": row.base_version,
        "description": row.description,
        "run_id": row.run_id or checkpoint.get("run_id"),
        "iteration": row.iteration if row.iteration is not None else checkpoint.get("best_iteration", checkpoint.get("iteration")),
        "accuracy": checkpoint.get("dev_accuracy"),
        "macro_f1": checkpoint.get("dev_macro_f1"),
        "holdout_accuracy": checkpoint.get("holdout_accuracy"),
        "latest_evaluation": _evaluation_summary(_latest_evaluation(db, row.task_id, row.version, checkpoint.get("config"))),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "config": checkpoint.get("config"),
    }


def list_versions(db: Session, task: Task) -> List[Dict[str, Any]]:
    """The example's baseline followed by its saved versions, newest first."""
    versions = [{
        "version": BASELINE_VERSION,
        "task": task.id,
        "source": "baseline",
        "base_version": None,
        "description": "The example's default prompt",
        "run_id": None,
        "iteration": None,
        "accuracy": None,
        "macro_f1": None,
        "holdout_accuracy": None,
        "latest_evaluation": _evaluation_summary(_latest_evaluation(db, task.id, BASELINE_VERSION, task.default_config)),
        "created_at": None,
        "config": task.default_config,
    }]
    rows = db.query(ModelVersion).filter(ModelVersion.task_id == task.id).order_by(ModelVersion.created_at.desc()).all()
    for row in rows:
        described = _describe(db, row)
        if described["config"] is not None:  # a row without a checkpoint file was saved before prompts were stored
            versions.append(described)
    return versions


def of_run(db: Session, run_id: str) -> List[Dict[str, Any]]:
    """The versions saved from a run: its auto-saved best first, then any rounds saved by hand."""
    rows = db.query(ModelVersion).filter(ModelVersion.run_id == run_id).order_by(ModelVersion.created_at).all()
    rows.sort(key=lambda row: row.source != "auto")
    return [{"version": row.version, "source": row.source, "iteration": row.iteration} for row in rows]
