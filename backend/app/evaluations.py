"""Model evaluations: score a version on a sample of an example's golden dataset.

An evaluation runs on a background thread and is a row in evaluation_runs from the
moment it starts, so the page that started it can go away and come back: the row
carries the status, how far it is, and at the end the full result.
"""

import random
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app import golden, scoring, versions
from app.database import EvaluationRun, SessionLocal
from app.tasks.registry import Task

TOP_ERRORS = 10


def mark_interrupted() -> None:
    """Evaluations still marked running belonged to a server process that has gone away."""
    db = SessionLocal()
    try:
        db.query(EvaluationRun).filter(EvaluationRun.status == "running").update(
            {"status": "interrupted", "error": "The server restarted while this evaluation was running."}
        )
        db.commit()
    finally:
        db.close()


def summary(run: EvaluationRun) -> Dict[str, Any]:
    return {
        "eval_id": run.id,
        "task": run.task_id,
        "model": run.model,
        "version": run.version,
        "status": run.status,
        "progress_done": run.progress_done or 0,
        "progress_total": run.progress_total or 0,
        "error": run.error,
        "sample_size": run.sample_size,
        "accuracy": run.accuracy if run.status == "completed" else None,
        "macro_f1": run.macro_f1 if run.status == "completed" else None,
        "timestamp": run.created_at.isoformat() if run.created_at else None,
    }


def detail(run: EvaluationRun) -> Dict[str, Any]:
    """The summary plus, once completed, the full result."""
    return {**(run.result or {}), **summary(run)}


def start(db: Session, task: Task, version: str, sample_size: int, seed: int, model: str = "laya") -> Dict[str, Any]:
    """Start an evaluation and return its summary straight away. Raises ValueError if it cannot start."""
    examples = golden.examples_for(db, task)
    if not examples:
        in_language = f" in language '{task.language}'" if task.language else ""
        raise ValueError(f"No golden {task.items}{in_language} for this example. Import its dataset first.")
    config = versions.load_config(version, task)
    # A seeded shuffle, not the first rows: imported datasets are not stored in random order.
    examples.sort(key=lambda e: e["text"])
    random.Random(seed).shuffle(examples)
    sample = examples[:sample_size]
    stats = golden.stats(db, task)

    run = EvaluationRun(
        id=str(uuid.uuid4()),
        task_id=task.id,
        model=model,
        version=version,
        sample_size=len(sample),
        total_cases=len(sample),
        accuracy=0.0,
        macro_f1=0.0,
        result={},
        status="running",
        progress_done=0,
        progress_total=len(sample),
        created_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    started = summary(run)

    context = {
        "seed": seed,
        "total_golden_data": stats["usable"],
        "eval_language": task.language,
        "excluded_other_language": stats["excluded_other_language"],
    }
    threading.Thread(
        target=_run, args=(run.id, task, config, sample, context), name=f"evaluation-{run.id}", daemon=True
    ).start()
    return started


def _run(eval_id: str, task: Task, config: Dict[str, Any], sample: List[Dict[str, str]], context: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        def on_progress(done: int, total: int) -> None:
            db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).update({"progress_done": done})
            db.commit()

        # The version's own label bias (if it has one) is applied as saved; nothing is fitted here.
        _, metrics = scoring.evaluate(config, sample, task, "balanced", fit_bias=False, on_progress=on_progress)
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
        run.status = "completed"
        db.commit()
    except Exception as e:  # noqa: BLE001 - a failed evaluation must be recorded, not lost with the thread
        db.rollback()
        db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).update({"status": "failed", "error": str(e)})
        db.commit()
    finally:
        db.close()


def running(db: Session, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(EvaluationRun).filter(EvaluationRun.status == "running")
    if task_id:
        query = query.filter(EvaluationRun.task_id == task_id)
    return [summary(run) for run in query.order_by(EvaluationRun.created_at).all()]
