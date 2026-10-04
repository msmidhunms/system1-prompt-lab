"""Model evaluations: score a version on a sample of an example's golden dataset.

An evaluation runs on a background thread and is a row in evaluation_runs from the
moment it starts, so the page that started it can go away and come back: the row
carries the status, how far it is, and at the end the full result.

A comparison is several evaluations of the same prompt on the same sample, one per
model, run one after another and tied together by a comparison id.
"""

import random
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app import engines, golden, scoring, versions
from app.database import EvaluationRun, SessionLocal
from app.laya_inference import on_engine, predict_proba
from app.tasks.registry import Task

TOP_ERRORS = 10
ACTIVE = ("queued", "running")


def mark_interrupted() -> None:
    """Evaluations still marked running belonged to a server process that has gone away."""
    db = SessionLocal()
    try:
        db.query(EvaluationRun).filter(EvaluationRun.status.in_(ACTIVE)).update(
            {"status": "interrupted", "error": "The server restarted while this evaluation was running."},
            synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()


def summary(run: EvaluationRun) -> Dict[str, Any]:
    done = run.status == "completed"
    return {
        "eval_id": run.id,
        "task": run.task_id,
        "engine": run.engine or engines.LAYA,
        "version": run.version,
        "status": run.status,
        "progress_done": run.progress_done or 0,
        "progress_total": run.progress_total or 0,
        "error": run.error,
        "sample_size": run.sample_size,
        "accuracy": run.accuracy if done else None,
        "macro_f1": run.macro_f1 if done else None,
        "seconds": run.seconds,
        "ms_per_item": round(run.seconds * 1000 / run.sample_size, 1) if done and run.seconds is not None and run.sample_size else None,
        "comparison_id": run.comparison_id,
        "timestamp": run.created_at.isoformat() if run.created_at else None,
    }


def detail(run: EvaluationRun) -> Dict[str, Any]:
    """The summary plus, once completed, the full result."""
    return {**(run.result or {}), **summary(run)}


def _sample(db: Session, task: Task, sample_size: int, seed: int) -> List[Dict[str, str]]:
    examples = golden.examples_for(db, task)
    if not examples:
        in_language = f" in language '{task.language}'" if task.language else ""
        raise ValueError(f"No golden {task.items}{in_language} for this example. Add data in the Dataset tab.")
    # A seeded shuffle, not the first rows: imported datasets are not stored in random order.
    examples.sort(key=lambda e: e["text"])
    random.Random(seed).shuffle(examples)
    return examples[:sample_size]


def start(
    db: Session, task: Task, version: str, sample_size: int, seed: int, engine: Optional[str] = None
) -> Dict[str, Any]:
    """Start one evaluation and return its summary straight away. Raises ValueError if it cannot start."""
    return _start(db, task, version, sample_size, seed, [engine], comparison_id=None)[0]


def start_comparison(
    db: Session, task: Task, version: str, sample_size: int, seed: int, engine_ids: List[str]
) -> List[Dict[str, Any]]:
    """Start the same evaluation on several models, one after another. Returns their summaries."""
    if len(set(engine_ids)) < 2:
        raise ValueError("Choose at least two models to compare")
    return _start(db, task, version, sample_size, seed, list(dict.fromkeys(engine_ids)), comparison_id=str(uuid.uuid4()))


def _start(db, task, version, sample_size, seed, engine_ids, comparison_id) -> List[Dict[str, Any]]:
    base_config = versions.load_config(version, task)
    sample = _sample(db, task, sample_size, seed)
    stats = golden.stats(db, task)
    context = {
        "seed": seed,
        "total_golden_data": stats["usable"],
        "eval_language": task.language,
        "excluded_other_language": stats["excluded_other_language"],
    }
    jobs, started = [], []
    now = datetime.utcnow()
    for engine_id in engine_ids:
        config = on_engine(base_config, engine_id)
        run = EvaluationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            model="laya",
            engine=config.get("engine") or engines.LAYA,
            version=version,
            sample_size=len(sample),
            total_cases=len(sample),
            accuracy=0.0,
            macro_f1=0.0,
            result={},
            status="queued",
            progress_done=0,
            progress_total=len(sample),
            comparison_id=comparison_id,
            created_at=now,
        )
        db.add(run)
        jobs.append((run.id, config))
        started.append(run)
    db.commit()
    summaries = [summary(run) for run in started]
    threading.Thread(target=_run_all, args=(jobs, task, sample, context), name=f"evaluation-{jobs[0][0]}", daemon=True).start()
    return summaries


def _run_all(jobs, task: Task, sample: List[Dict[str, str]], context: Dict[str, Any]) -> None:
    for eval_id, config in jobs:
        _run(eval_id, task, config, sample, context)


def _run(eval_id: str, task: Task, config: Dict[str, Any], sample: List[Dict[str, str]], context: Dict[str, Any]) -> None:
    db = SessionLocal()

    def set_fields(**fields) -> None:
        db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).update(fields)
        db.commit()

    try:
        set_fields(status="running")
        # Load the model before the clock starts, so the time is the scoring and not the download.
        predict_proba([sample[0]["text"]], config, task)
        started = time.perf_counter()
        # The version's own label bias (if it has one) is applied as saved; nothing is fitted here.
        _, metrics = scoring.evaluate(
            config, sample, task, "balanced", fit_bias=False,
            on_progress=lambda done, total: set_fields(progress_done=done),
        )
        seconds = time.perf_counter() - started
        errors = sorted((c for c in metrics["cases"] if not c["correct"]), key=lambda c: c["confidence"], reverse=True)
        run = db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).first()
        if run is None:  # deleted while it ran
            return
        run.result = {
            "labels": list(task.labels),
            "total_cases": metrics["total"],
            "per_class_metrics": metrics["per_class"],
            # confusion_matrix[actual][predicted]
            "confusion_matrix": metrics["confusion"],
            "predicted_counts": metrics["predicted_counts"],
            "error_count": len(errors),
            "error_rate": round(len(errors) / metrics["total"], 4),
            "top_errors": errors[:TOP_ERRORS],
            **context,
        }
        run.accuracy = metrics["accuracy"]
        run.macro_f1 = metrics["macro_f1"]
        run.progress_done = metrics["total"]
        run.seconds = round(seconds, 3)
        run.status = "completed"
        db.commit()
    except Exception as e:  # noqa: BLE001 - a failed evaluation must be recorded, not lost with the thread
        db.rollback()
        set_fields(status="failed", error=str(e))
    finally:
        db.close()


def running(db: Session, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(EvaluationRun).filter(EvaluationRun.status.in_(ACTIVE))
    if task_id:
        query = query.filter(EvaluationRun.task_id == task_id)
    return [summary(run) for run in query.order_by(EvaluationRun.created_at).all()]


def comparisons(db: Session, task: Task, limit: int = 30) -> List[Dict[str, Any]]:
    """The example's comparisons, newest first, each with one row per model."""
    runs = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.task_id == task.id, EvaluationRun.comparison_id.isnot(None))
        .order_by(EvaluationRun.created_at.desc()).all()
    )
    grouped: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        group = grouped.setdefault(run.comparison_id, {
            "comparison_id": run.comparison_id,
            "version": run.version,
            "sample_size": run.sample_size,
            "timestamp": run.created_at.isoformat() if run.created_at else None,
            "runs": [],
        })
        group["runs"].append(summary(run))
    order = list(engines.ENGINES)
    for group in grouped.values():
        group["runs"].sort(key=lambda r: order.index(r["engine"]) if r["engine"] in order else len(order))
        group["active"] = any(r["status"] in ACTIVE for r in group["runs"])
    return list(grouped.values())[:limit]
