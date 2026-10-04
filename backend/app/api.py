"""API routes for System 1 experiments.

Every route works on one example (a task from app.tasks.registry), named by `task` in
the query string or the request body. Leaving it out means the search-intent example.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import engines, evaluations, golden, karpathy_loop, llm, scoring, serp, versions
from app.config import DEFAULT_TASK_ID
from app.database import EvaluationRun, FeedbackRecord, GoldenRow, PredictionRecord, get_db
from app.inference import classify_intent
from app.laya_inference import LAYA_MODELS, on_engine, predict_batch
from app.tasks import sources
from app.tasks.registry import TASKS, Task, get_task
from app.versions import BASELINE_VERSION

router = APIRouter(prefix="/api", tags=["api"])

# Evaluations left running by a previous server process can no longer finish.
evaluations.mark_interrupted()


def _task(task_id: Optional[str]) -> Task:
    try:
        return get_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Request models
class PredictRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    text: str
    version: str = BASELINE_VERSION
    # The model to run the version's prompt on. Left out, the one the version was saved for (Laya for the baseline).
    engine: Optional[str] = None


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
    # The examples to import. Empty means every example that has no golden data yet.
    tasks: List[str] = []
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
    # The model whose prompt is optimized. Left out, the start version's own (Laya for the baseline).
    engine: Optional[str] = None


class SaveModelRequest(BaseModel):
    model_name: str
    run_id: str
    # The round whose prompt to save. Left out, the run's best prompt is saved.
    iteration: Optional[int] = None
    description: Optional[str] = None


class CreateModelRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    name: str
    config: Dict[str, Any]
    base_version: Optional[str] = None
    description: Optional[str] = None


class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class EvaluateRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    version: str = BASELINE_VERSION
    engine: Optional[str] = None
    sample_size: int = 100
    seed: int = 42


class CompareRequest(BaseModel):
    task: str = DEFAULT_TASK_ID
    version: str = BASELINE_VERSION
    # The models to run the version's prompt on. Empty means all of them.
    engines: List[str] = []
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

def _golden_counts(db: Session) -> Dict[str, int]:
    return dict(db.query(GoldenRow.task_id, func.count(GoldenRow.id)).group_by(GoldenRow.task_id).all())


@router.get("/tasks")
def list_tasks(db: Session = Depends(get_db)):
    """The examples that can be selected, with their labels, how much golden data each has and how to get it."""
    counts = _golden_counts(db)
    tasks = []
    for task in TASKS.values():
        info = task.public()
        info["golden_rows"] = counts.get(task.id, 0)
        info["importable"] = sources.importable(task)
        info["import_status"] = sources.import_status(task.id)
        info["setup_hint"] = sources.setup_hint(task)
        info["max_criterion_words"] = task.max_criterion_words
        info["max_instruction_words"] = task.max_instruction_words
        tasks.append(info)
    return {"default": DEFAULT_TASK_ID, "tasks": tasks}


@router.post("/setup/import")
def import_datasets(request: Optional[ImportRequest] = None, db: Session = Depends(get_db)):
    """Download and import examples' datasets in the background. GET /tasks reports each one's import_status."""
    request = request or ImportRequest()
    if not 10 <= request.rows <= 20000:
        raise HTTPException(status_code=400, detail="Rows must be between 10 and 20000")
    if request.tasks:
        tasks = [_task(task_id) for task_id in request.tasks]
        missing = [task.name for task in tasks if not sources.importable(task)]
        if missing:
            raise HTTPException(status_code=400, detail=f"No dataset to import for: {', '.join(missing)}")
    else:
        counts = _golden_counts(db)
        tasks = [task for task in TASKS.values() if sources.importable(task) and not counts.get(task.id)]
    return {"queued": sources.start_import(tasks, rows=request.rows, seed=request.seed, refresh=request.refresh)}


@router.get("/activity")
def get_activity(db: Session = Depends(get_db)):
    """Everything that is running in the background, across all examples."""
    run = karpathy_loop.active_run()
    loop = None
    if run:
        state = karpathy_loop.get_run(run["run_id"])
        loop = {
            "run_id": run["run_id"],
            "task": run["task"],
            "phase": state["phase"],
            "rounds_done": len(state["iterations"]),
            "rounds": state["loops"],
        }
    return {
        "loop": loop,
        "evaluations": evaluations.running(db),
        "imports": [{"task": task_id, **(sources.import_status(task_id) or {})} for task_id in sources.importing()],
    }


# ---------------------------------------------------------------- try it

@router.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """Classify one input with a model version of the example."""
    task = _task(request.task)
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail=f"{task.input_label} cannot be empty")

    try:
        config = on_engine(versions.load_config(request.version, task), request.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    engine = config.get("engine") or engines.LAYA
    try:
        prediction = predict_batch([text], config, task)[0]
    except Exception as e:  # noqa: BLE001
        if task.id != DEFAULT_TASK_ID or engine != engines.LAYA:
            raise HTTPException(status_code=500, detail=f"The model could not classify this {task.item}: {e}")
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
        model=engine,
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
        "model": engine,
        "engine": engine,
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
            detail=f"Need at least 20 labelled golden {task.items}{in_language}. Add data in the Dataset tab."
        )

    try:
        start_config = on_engine(versions.load_config(request.start_version, task), request.engine)
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
def get_karpathy_run(run_id: str, db: Session = Depends(get_db)):
    """Get the full state of a run: baseline, every iteration, best config, holdout scores, log."""
    run = karpathy_loop.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    # Looked up by run, not taken from the run's own record, so a renamed version shows its current name.
    run["saved_versions"] = versions.of_run(db, run_id)
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
        models = llm.list_models(config)
        # Kept with the provider's settings, so the list is still there after a reload.
        llm.update_config(db, config["provider"], models=models, make_active=False)
        return {"models": models}
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------- model versions

@router.get("/models")
def list_models(task: str = DEFAULT_TASK_ID, db: Session = Depends(get_db)):
    """List the example's baseline plus every model version saved for it."""
    return versions.list_versions(db, _task(task))


@router.post("/models")
def create_model(request: CreateModelRequest, db: Session = Depends(get_db)):
    """Save a prompt config as a new model version, e.g. an edited copy of another version."""
    try:
        return versions.create(
            db, _task(request.task), request.name, request.config, source="manual",
            base_version=request.base_version, description=request.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/models/{name}")
def update_model(name: str, request: UpdateModelRequest, db: Session = Depends(get_db)):
    """Rename a model version or change its description."""
    try:
        result = None
        if request.description is not None:
            result = versions.set_description(db, name, request.description)
        if request.name is not None and request.name.strip() != name:
            result = versions.rename(db, name, request.name)
        if result is None:
            raise ValueError("Nothing to change")
        return result
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/models/{name}")
def delete_model(name: str, db: Session = Depends(get_db)):
    try:
        versions.delete(db, name)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "version": name}


@router.post("/save-model")
def save_model(request: SaveModelRequest, db: Session = Depends(get_db)):
    """Save a prompt from an optimizer run as a named model version: its best, or the one a given round proposed."""
    run = karpathy_loop.get_run(request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        saved = versions.save_from_run(
            db, _task(run["task"]), run, request.model_name,
            iteration=request.iteration, description=request.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {**saved, "model_name": saved["version"], "status": "saved"}


# ---------------------------------------------------------------- evaluation

@router.post("/evaluate")
def run_evaluation(request: Optional[EvaluateRequest] = None, db: Session = Depends(get_db)):
    """Start scoring a model version on a seeded sample of the example's golden dataset.

    Returns at once with the evaluation's id and status "running". GET /evaluations/{id}
    reports progress and, when it has completed, the full result.
    """
    request = request or EvaluateRequest()
    task = _task(request.task)
    if request.sample_size < 10 or request.sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 10 and 10000")
    try:
        return evaluations.start(db, task, request.version, request.sample_size, request.seed, request.engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/engines")
def list_engines():
    """The models an example can be run on."""
    return engines.public()


@router.post("/compare")
def run_comparison(request: CompareRequest, db: Session = Depends(get_db)):
    """Evaluate one version's prompt on several models, on the same sample, one model after another."""
    task = _task(request.task)
    if request.sample_size < 10 or request.sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 10 and 10000")
    try:
        chosen = [engines.get_engine(e)["id"] for e in request.engines] or list(engines.ENGINES)
        return evaluations.start_comparison(db, task, request.version, request.sample_size, request.seed, chosen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/comparisons")
def list_comparisons(task: str = DEFAULT_TASK_ID, db: Session = Depends(get_db)):
    """The example's model comparisons, newest first, each with one evaluation per model."""
    return evaluations.comparisons(db, _task(task))


@router.get("/evaluations")
def list_evaluations(task: str = DEFAULT_TASK_ID, limit: int = 100, db: Session = Depends(get_db)):
    """The example's evaluations, newest first, including any still running."""
    runs = (
        db.query(EvaluationRun).filter(EvaluationRun.task_id == _task(task).id)
        .order_by(EvaluationRun.created_at.desc()).limit(max(1, min(limit, 500))).all()
    )
    return [evaluations.summary(run) for run in runs]


@router.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: str, db: Session = Depends(get_db)):
    """One evaluation: its status and progress, and the full result once it has completed."""
    run = db.query(EvaluationRun).filter(EvaluationRun.id == eval_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluations.detail(run)
