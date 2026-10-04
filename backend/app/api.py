"""API routes for System 1 experiments.

Every route works on one example (a task from app.tasks.registry), named by `task` in
the query string or the request body. Leaving it out means the search-intent example.
"""

import random
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import golden, karpathy_loop, llm, scoring, serp
from app.config import DEFAULT_TASK_ID
from app.database import EvaluationRun, FeedbackRecord, GoldenRow, ModelVersion, PredictionRecord, get_db
from app.inference import classify_intent
from app.laya_inference import (
    BASELINE_VERSION,
    LAYA_MODELS,
    checkpoint_path,
    load_version_config,
    predict_batch,
    save_checkpoint,
)
from app.tasks import sources
from app.tasks.registry import TASKS, Task, get_task

router = APIRouter(prefix="/api", tags=["api"])

TOP_ERRORS = 10


def _task(task_id: Optional[str]) -> Task:
    try:
        return get_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Request models
class PredictRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    text: str
    model: str = "laya"
    version: str = BASELINE_VERSION


class FeedbackRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    text: str
    predicted_label: str
    feedback_type: str
    corrected_label: Optional[str] = None
    confidence: float = 0.0
    notes: Optional[str] = None
    model: str = "laya"
    version: str = BASELINE_VERSION


class GoldenRowRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    text: str
    label: str
    secondary_labels: Optional[List[str]] = None
    language: Optional[str] = None


class GoldenRowUpdate(BaseModel):
    text: Optional[str] = None
    label: Optional[str] = None
    secondary_labels: Optional[List[str]] = None
    language: Optional[str] = None


class ImportRequest(BaseModel):
    rows: int = sources.DEFAULT_ROWS
    seed: int = sources.DEFAULT_SEED
    refresh: bool = False


class KarpathyLoopRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    loops: int = 10
    sample_size: int = 400
    seed: int = 42
    start_version: str = BASELINE_VERSION
    metric: str = "balanced"
    calibrate: bool = True
    use_serp: bool = False
    laya_model: str = "typed-decisions"


class SaveModelRequest(BaseModel):
    model_name: str
    run_id: str
    description: Optional[str] = None


class EvaluateRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    model: str = "laya"
    version: str = BASELINE_VERSION
    sample_size: int = 100
    seed: int = 42


class LLMConfigRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False


class LLMProviderRequest(BaseModel):
    provider: Optional[str] = None


# ---------------------------------------------------------------- examples

@router.get("/tasks")
def list_tasks(db: Session = Depends(get_db)):
    """The examples that can be selected, with their labels and how much golden data each has."""
    tasks = []
    for task in TASKS.values():
        info = task.public()
        info["golden_rows"] = db.query(GoldenRow).filter(GoldenRow.task_id == task.id).count()
        info["importable"] = sources.importable(task)
        tasks.append(info)
    return {"default": DEFAULT_TASK_ID, "tasks": tasks}


@router.post("/tasks/{task_id}/import")
def import_task_dataset(task_id: str, request: Optional[ImportRequest] = None, db: Session = Depends(get_db)):
    """Download the example's public dataset (or reuse its snapshot on disk) and add it to the golden dataset."""
    task = _task(task_id)
    request = request or ImportRequest()
    if not 10 <= request.rows <= 20000:
        raise HTTPException(status_code=400, detail="Rows must be between 10 and 20000")
    try:
        return sources.import_task(db, task, rows=request.rows, seed=request.seed, refresh=request.refresh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - a failed download must reach the user as a message
        raise HTTPException(status_code=502, detail=f"Could not import the dataset: {e}")


# ---------------------------------------------------------------- try it

@router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """Classify one input with a model version of the example."""
    task = _task(request.task)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{task.input_label} cannot be empty")

    try:
        config = load_version_config(request.version, task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        prediction = predict_batch([text], config, task)[0]
    except Exception as e:  # noqa: BLE001
        if task.id != DEFAULT_TASK_ID:
            raise HTTPException(status_code=500, detail=f"Laya could not classify this {task.item}: {e}")
        # Search intent has a keyword heuristic to fall back on.
        print(f"Laya classification failed: {e}, falling back to keyword-based")
        intent, confidence = classify_intent(text)
        prediction = {"label": intent.value, "confidence": confidence, "probabilities": {}}

    pred_id = str(uuid.uuid4())
    db.add(PredictionRecord(
        id=pred_id,
        task_id=task.id,
        query=text,
        predicted_intent=prediction["label"],
        confidence=prediction["confidence"],
        model=request.model,
        version=request.version,
    ))
    db.commit()

    golden_row = golden.find_row(db, task, text)
    return {
        "prediction_id": pred_id,
        "task": task.id,
        "text": text,
        "predicted_label": prediction["label"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "model": request.model,
        "version": request.version,
        # The label this input already has in the golden dataset, if it is there.
        "golden_label": golden_row.label if golden_row else None,
    }


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """Record feedback on a prediction. Feedback that settles the label also updates the golden dataset."""
    task = _task(request.task)
    text = request.text.strip()

    if request.feedback_type not in ["correct", "incorrect", "unsure"]:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    if request.corrected_label and request.corrected_label not in task.labels:
        raise HTTPException(status_code=400, detail="Invalid corrected label")

    feedback_id = str(uuid.uuid4())
    db.add(FeedbackRecord(
        id=feedback_id,
        task_id=task.id,
        query=text,
        predicted_intent=request.predicted_label,
        feedback_type=request.feedback_type,
        corrected_intent=request.corrected_label,
        confidence=request.confidence,
        notes=request.notes,
        model=request.model,
        version=request.version,
    ))

    label = request.predicted_label if request.feedback_type == "correct" else request.corrected_label
    golden_status = None
    if label:
        try:
            _, created, _ = golden.upsert_row(db, task, text, label, source="feedback")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        golden_status = "created" if created else "updated"
    db.commit()

    return {
        "feedback_id": feedback_id,
        "status": "success",
        "golden_status": golden_status,
        "message": "Feedback recorded successfully",
    }


# ---------------------------------------------------------------- golden dataset

@router.get("/golden-data")
def get_golden_data(
    task: str = DEFAULT_TASK_ID,
    page: int = 1,
    page_size: int = 20,
    q: Optional[str] = None,
    label: Optional[str] = None,
    source: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """One page of the example's golden dataset, most recently changed first."""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")
    if not 1 <= page_size <= 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")
    return golden.list_rows(db, _task(task), page=page, page_size=page_size, q=q, label=label, source=source)


@router.get("/golden-data/stats")
def get_golden_data_stats(task: str = DEFAULT_TASK_ID, db: Session = Depends(get_db)):
    """Counts of the example's golden rows by label, source and language."""
    return golden.stats(db, _task(task))


@router.post("/golden-data")
def add_golden_row(request: GoldenRowRequest, db: Session = Depends(get_db)):
    """Add a labelled input. If the input is already there (ignoring case and spacing), that row is updated instead."""
    task = _task(request.task)
    fields = {}
    if request.secondary_labels is not None:
        fields["secondary_labels"] = request.secondary_labels
    if request.language:
        fields["language"] = request.language.strip()
    try:
        row, created, previous = golden.upsert_row(db, task, request.text, request.label, source="manual", **fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    db.commit()
    return {
        "status": "created" if created else "updated",
        "previous_label": previous,
        "row": golden.serialise(row),
    }


def _golden_row(db: Session, row_id: str) -> GoldenRow:
    row = db.query(GoldenRow).filter(GoldenRow.id == row_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Golden row not found")
    return row


@router.put("/golden-data/{row_id}")
def update_golden_row(row_id: str, request: GoldenRowUpdate, db: Session = Depends(get_db)):
    """Edit a golden row's input, label, secondary labels or language."""
    row = _golden_row(db, row_id)
    try:
        golden.update_row(
            db, _task(row.task_id), row,
            text=request.text, label=request.label,
            secondary_labels=request.secondary_labels, language=request.language,
        )
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    db.commit()
    return golden.serialise(row)


@router.delete("/golden-data/{row_id}")
def delete_golden_row(row_id: str, db: Session = Depends(get_db)):
    db.delete(_golden_row(db, row_id))
    db.commit()
    return {"status": "deleted", "id": row_id}


# ---------------------------------------------------------------- karpathy loop

@router.post("/karpathy-loop")
def start_karpathy_loop(request: KarpathyLoopRequest, db: Session = Depends(get_db)):
    """Start a Karpathy loop: an LLM iteratively rewrites the Laya prompt, keeping what scores better."""
    task = _task(request.task)
    if request.loops < 1 or request.loops > 100:
        raise HTTPException(status_code=400, detail="Loops must be between 1 and 100")

    if request.sample_size < 20 or request.sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 20 and 10000")

    examples = golden.examples_for(db, task)
    if len(examples) < 20:
        in_language = f" in language '{task.language}'" if task.language else ""
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 labelled golden {task.items}{in_language}. Import the dataset in the Golden Dataset tab."
        )

    try:
        start_config = load_version_config(request.start_version, task)
        llm_config = llm.resolve_config(db)
    except (ValueError, llm.LLMError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        return karpathy_loop.start_run(
            task=task,
            examples=examples,
            llm_config=llm_config,
            loops=request.loops,
            sample_size=request.sample_size,
            seed=request.seed,
            start_config=start_config,
            start_version=request.start_version,
            metric=request.metric,
            calibrate=request.calibrate,
            use_serp=request.use_serp,
            laya_model=request.laya_model,
            excluded_other_language=golden.stats(db, task)["excluded_other_language"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/karpathy-loop/options")
def get_karpathy_options(task: str = DEFAULT_TASK_ID):
    """Choices for the run settings: objectives, Laya checkpoints, and whether SERP context can be used."""
    return {
        "metrics": scoring.METRICS,
        "laya_models": LAYA_MODELS,
        "serp_supported": _task(task).supports_serp,
        "serp_available": _task(task).supports_serp and serp.available(),
        "keep_confidence": scoring.KEEP_CONFIDENCE,
    }


@router.get("/karpathy-loop/runs")
def list_karpathy_runs(task: Optional[str] = None):
    """List an example's Karpathy loop runs (newest first) and the run currently in progress, if any.

    Only one run is in progress at a time across all examples, so the active run may belong to another one.
    """
    active = karpathy_loop.active_run()
    return {
        "active_run_id": active["run_id"] if active else None,
        "active_task": active["task"] if active else None,
        "runs": karpathy_loop.list_runs(_task(task).id if task else None),
    }


@router.get("/karpathy-loop/runs/{run_id}")
def get_karpathy_run(run_id: str):
    """Get the full state of a run: baseline, every iteration, best config, holdout scores, log."""
    run = karpathy_loop.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/karpathy-loop/runs/{run_id}/stop")
def stop_karpathy_run(run_id: str):
    """Ask a running loop to stop after the current round."""
    if not karpathy_loop.request_stop(run_id):
        raise HTTPException(status_code=404, detail="Run is not in progress")
    return {"status": "stopping"}


# ---------------------------------------------------------------- llm settings

@router.get("/llm/config")
def get_llm_config(db: Session = Depends(get_db)):
    """Get the LLM providers and the current settings (API keys are never returned)."""
    return llm.public_config(db)


@router.put("/llm/config")
def update_llm_config(request: LLMConfigRequest, db: Session = Depends(get_db)):
    """Save settings for a provider and make it the active one."""
    try:
        llm.update_config(
            db,
            provider=request.provider,
            model=request.model,
            base_url=request.base_url,
            api_key=request.api_key,
            clear_api_key=request.clear_api_key,
        )
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return llm.public_config(db)


@router.post("/llm/test")
def test_llm(request: LLMProviderRequest, db: Session = Depends(get_db)):
    """Send a tiny prompt to the saved LLM settings to check they work."""
    try:
        config = llm.resolve_config(db, request.provider)
        reply = llm.complete(config, "You are a connection test.", "Reply with the single word: ok")
    except llm.LLMError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "provider": config["provider"], "model": config["model"], "reply": reply.strip()[:200]}


@router.post("/llm/models")
def list_llm_models(request: LLMProviderRequest, db: Session = Depends(get_db)):
    """List the models the provider serves, using the saved settings."""
    try:
        config = llm.resolve_config(db, request.provider)
        return {"models": llm.list_models(config)}
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------- model versions

@router.get("/models")
def list_models(task: str = DEFAULT_TASK_ID, db: Session = Depends(get_db)):
    """List the example's baseline plus every model version saved for it."""
    task_ = _task(task)
    saved = (
        db.query(ModelVersion).filter(ModelVersion.task_id == task_.id)
        .order_by(ModelVersion.created_at.desc()).all()
    )
    versions = [{
        "version": BASELINE_VERSION,
        "accuracy": None,
        "base_version": None,
        "description": "Default Laya prompt",
        "created_at": None,
        "config": task_.default_config,
    }]
    for model in saved:
        try:
            config = load_version_config(model.version, task_)
        except ValueError:
            continue  # row without a checkpoint file (saved before prompts were stored)
        versions.append({
            "version": model.version,
            "accuracy": model.accuracy,
            "base_version": model.base_version,
            "description": model.description,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "config": config,
        })
    return versions


@router.post("/save-model")
def save_model(request: SaveModelRequest, db: Session = Depends(get_db)):
    """Save the best prompt found by a Karpathy loop run as a named model version of the run's example."""
    model_name = request.model_name.strip()

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name cannot be empty")

    run = karpathy_loop.get_run(request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run["status"] == "running":
        raise HTTPException(status_code=400, detail="Run is still in progress")

    if run["best_accuracy"] is None:
        raise HTTPException(status_code=400, detail="Run has no evaluated config to save")

    try:
        path = checkpoint_path(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Version names are unique across examples: each is one file under autoresearch/checkpoints/.
    existing = db.query(ModelVersion).filter(
        ModelVersion.version == model_name
    ).first()

    if existing or path.exists() or model_name == BASELINE_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"Model version '{model_name}' already exists"
        )

    save_checkpoint(model_name, {
        "version": model_name,
        "task": run["task"],
        "config": run["best_config"],
        "run_id": run["run_id"],
        "best_iteration": run["best_iteration"],
        "base_version": run["start_version"],
        "dev_accuracy": run["best_accuracy"],
        "dev_macro_f1": run.get("best_macro_f1"),
        "holdout_accuracy": run["holdout"]["best"]["accuracy"] if run.get("holdout") else None,
        "llm": run["llm"],
        "saved_at": datetime.utcnow().isoformat(),
    })

    model = ModelVersion(
        id=str(uuid.uuid4()),
        task_id=run["task"],
        model_name="laya",
        version=model_name,
        accuracy=run["best_accuracy"],
        base_version=run["start_version"],
        description=request.description,
    )
    db.add(model)
    db.commit()

    return {
        "model_id": model.id,
        "model_name": model_name,
        "accuracy": run["best_accuracy"],
        "status": "saved"
    }


# ---------------------------------------------------------------- evaluation

@router.post("/evaluate")
def run_evaluation(request: Optional[EvaluateRequest] = None, db: Session = Depends(get_db)):
    """Score a model version on a seeded sample of the example's golden dataset, and save the result."""
    request = request or EvaluateRequest()
    task = _task(request.task)

    if request.sample_size < 10 or request.sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 10 and 10000")

    examples = golden.examples_for(db, task)
    if not examples:
        in_language = f" in language '{task.language}'" if task.language else ""
        raise HTTPException(
            status_code=400,
            detail=f"No golden {task.items}{in_language} for this example. Import the dataset in the Golden Dataset tab."
        )
    # A seeded shuffle, not the first rows: imported datasets are not stored in random order.
    examples.sort(key=lambda e: e["text"])
    random.Random(request.seed).shuffle(examples)
    sample = examples[:request.sample_size]

    try:
        config = load_version_config(request.version, task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # The version's own label bias (if it has one) is applied as saved; nothing is fitted here.
    _, metrics = scoring.evaluate(config, sample, task, "balanced", fit_bias=False)

    errors = sorted((c for c in metrics["cases"] if not c["correct"]), key=lambda c: c["confidence"], reverse=True)
    stats = golden.stats(db, task)
    created_at = datetime.utcnow()
    result = {
        "eval_id": str(uuid.uuid4()),
        "task": task.id,
        "labels": list(task.labels),
        "model": request.model,
        "version": request.version,
        "seed": request.seed,
        "total_cases": metrics["total"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_metrics": metrics["per_class"],
        # confusion_matrix[actual][predicted]
        "confusion_matrix": metrics["confusion"],
        "predicted_counts": metrics["predicted_counts"],
        "error_count": len(errors),
        "error_rate": round(len(errors) / metrics["total"], 4),
        "top_errors": errors[:TOP_ERRORS],
        "timestamp": created_at.isoformat(),
        "sample_size": metrics["total"],
        "total_golden_data": stats["usable"],
        "eval_language": task.language,
        "excluded_other_language": stats["excluded_other_language"],
    }
    db.add(EvaluationRun(
        id=result["eval_id"],
        task_id=task.id,
        model=request.model,
        version=request.version,
        sample_size=metrics["total"],
        total_cases=metrics["total"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        result=result,
        created_at=created_at,
    ))
    db.commit()
    return result


@router.get("/evaluations")
def list_evaluations(task: str = DEFAULT_TASK_ID, limit: int = 100, db: Session = Depends(get_db)):
    """The example's saved evaluations, newest first."""
    runs = (
        db.query(EvaluationRun).filter(EvaluationRun.task_id == _task(task).id)
        .order_by(EvaluationRun.created_at.desc()).limit(max(1, min(limit, 500))).all()
    )
    return [{
        "eval_id": run.id,
        "task": run.task_id,
        "model": run.model,
        "version": run.version,
        "sample_size": run.sample_size,
        "accuracy": run.accuracy,
        "macro_f1": run.macro_f1,
        "timestamp": run.created_at.isoformat() if run.created_at else None,
    } for run in runs]


@router.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: str, db: Session = Depends(get_db)):
    """The full result of one saved evaluation."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return run.result
