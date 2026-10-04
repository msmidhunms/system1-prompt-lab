"""API routes for System 1 experiments."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import uuid

from app import karpathy_loop, llm
from app.database import get_db, PredictionRecord, FeedbackRecord, ModelVersion
from app.tasks.intent.labels import SearchIntent
from app.inference import classify_intent, classify_intent_v2
from app.laya_inference import (
    BASELINE_VERSION,
    DEFAULT_CONFIG,
    checkpoint_path,
    classify_with_laya,
    load_version_config,
    predict_batch,
    save_checkpoint,
)

router = APIRouter(prefix="/api", tags=["api"])


# Request models
class PredictRequest(BaseModel):
    query: str
    model: str = "laya"
    version: str = "v1_baseline"


class FeedbackRequest(BaseModel):
    query: str
    predicted_intent: str
    feedback_type: str
    corrected_intent: Optional[str] = None
    confidence: float = 0.0
    notes: Optional[str] = None
    model: str = "laya"
    version: str = "v1_baseline"


class KarpathyLoopRequest(BaseModel):
    loops: int = 10
    sample_size: int = 200
    seed: int = 42
    start_version: str = BASELINE_VERSION


class SaveModelRequest(BaseModel):
    model_name: str
    run_id: str
    description: Optional[str] = None


class EvaluateRequest(BaseModel):
    model: str = "laya"
    version: str = "v1_baseline"


class LLMConfigRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False


class LLMProviderRequest(BaseModel):
    provider: Optional[str] = None


@router.post("/predict")
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """Get intent prediction for a query."""
    query = request.query
    model = request.model
    version = request.version

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Use Laya-based classifier if model is 'laya'
    if model == "laya":
        try:
            config = load_version_config(version)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            predicted_intent, confidence = classify_with_laya(query, config)
        except Exception as e:
            print(f"Laya classification failed: {e}, falling back to keyword-based")
            predicted_intent, confidence = classify_intent(query)
    elif version == "v1_baseline":
        predicted_intent, confidence = classify_intent(query)
    elif version.startswith("v2"):
        predicted_intent, confidence = classify_intent_v2(query)
    else:
        predicted_intent, confidence = classify_intent(query)

    # Store prediction record
    pred_id = str(uuid.uuid4())
    record = PredictionRecord(
        id=pred_id,
        query=query.strip(),
        predicted_intent=predicted_intent.value,
        confidence=confidence,
        model=model,
        version=version
    )
    db.add(record)
    db.commit()

    return {
        "prediction_id": pred_id,
        "query": query.strip(),
        "predicted_intent": predicted_intent.value,
        "confidence": confidence,
        "model": model,
        "version": version,
    }


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit user feedback for a prediction."""
    query = request.query
    predicted_intent = request.predicted_intent
    feedback_type = request.feedback_type
    corrected_intent = request.corrected_intent
    confidence = request.confidence
    notes = request.notes
    model = request.model
    version = request.version

    # Validate feedback type
    if feedback_type not in ["correct", "incorrect", "unsure"]:
        raise HTTPException(status_code=400, detail="Invalid feedback type")

    # Validate corrected intent if provided
    if corrected_intent and corrected_intent not in SearchIntent.all_labels():
        raise HTTPException(status_code=400, detail="Invalid corrected intent")

    # Store feedback
    feedback_id = str(uuid.uuid4())
    record = FeedbackRecord(
        id=feedback_id,
        query=query.strip(),
        predicted_intent=predicted_intent,
        feedback_type=feedback_type,
        corrected_intent=corrected_intent,
        confidence=confidence,
        notes=notes,
        model=model,
        version=version
    )
    db.add(record)
    db.commit()

    return {
        "feedback_id": feedback_id,
        "status": "success",
        "message": "Feedback recorded successfully"
    }


def build_golden_data(db: Session) -> list:
    """Golden dataset: imported keywords, with user feedback corrections applied on top."""
    # Get imported test data
    test_data = db.query(PredictionRecord).filter(
        PredictionRecord.version == "v1_test_data"
    ).all()

    # Get user feedback
    feedbacks = db.query(FeedbackRecord).all()

    # Create golden dataset combining both
    golden_data = {}

    # Add imported test data
    for record in test_data:
        if record.query not in golden_data:
            golden_data[record.query] = {
                "id": record.id,
                "query": record.query,
                "correct_intent": record.predicted_intent,
                "source": "imported",
                "feedback_entries": [],
            }

    # Add user feedback
    for feedback in feedbacks:
        if feedback.query not in golden_data:
            golden_data[feedback.query] = {
                "id": str(uuid.uuid4()),
                "query": feedback.query,
                "correct_intent": None,
                "source": "feedback",
                "feedback_entries": [],
            }

        golden_data[feedback.query]["feedback_entries"].append({
            "feedback_type": feedback.feedback_type,
            "corrected_intent": feedback.corrected_intent,
        })

        # If feedback provides a corrected intent, use that
        if feedback.feedback_type == "correct":
            golden_data[feedback.query]["correct_intent"] = feedback.predicted_intent
        elif feedback.corrected_intent:
            golden_data[feedback.query]["correct_intent"] = feedback.corrected_intent

    # Convert to list
    result = []
    for data in golden_data.values():
        result.append({
            "id": data["id"],
            "query": data["query"],
            "correct_intent": data["correct_intent"] or "unknown",
            "source": data["source"],
            "feedback_count": len(data["feedback_entries"]),
        })

    return result


@router.get("/golden-data")
async def get_golden_data(db: Session = Depends(get_db)):
    """Get all golden dataset (imported keywords + user feedback)."""
    return build_golden_data(db)


@router.get("/golden-data/stats")
async def get_golden_data_stats(db: Session = Depends(get_db)):
    """Get statistics about the golden dataset."""
    # Count imported test data
    imported_count = db.query(PredictionRecord).filter(
        PredictionRecord.version == "v1_test_data"
    ).count()

    # Count user feedback
    feedback_count = db.query(FeedbackRecord).count()

    # Count unique queries in feedback
    from sqlalchemy import func
    unique_feedback_queries = db.query(func.count(func.distinct(FeedbackRecord.query))).scalar() or 0

    return {
        "imported_data": imported_count,
        "user_feedback_count": feedback_count,
        "unique_feedback_queries": unique_feedback_queries,
        "total_golden_data": imported_count + unique_feedback_queries,
    }


@router.post("/karpathy-loop")
def start_karpathy_loop(request: KarpathyLoopRequest, db: Session = Depends(get_db)):
    """Start a Karpathy loop: an LLM iteratively rewrites the Laya prompt, keeping what scores better."""
    if request.loops < 1 or request.loops > 100:
        raise HTTPException(status_code=400, detail="Loops must be between 1 and 100")

    if request.sample_size < 20 or request.sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 20 and 10000")

    examples = [
        {"query": d["query"], "intent": d["correct_intent"]}
        for d in build_golden_data(db)
        if d["correct_intent"] in SearchIntent.all_labels()
    ]
    if len(examples) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 labelled golden examples")

    try:
        start_config = load_version_config(request.start_version)
        llm_config = llm.resolve_config(db)
    except (ValueError, llm.LLMError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        return karpathy_loop.start_run(
            examples=examples,
            llm_config=llm_config,
            loops=request.loops,
            sample_size=request.sample_size,
            seed=request.seed,
            start_config=start_config,
            start_version=request.start_version,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/karpathy-loop/runs")
def list_karpathy_runs():
    """List all Karpathy loop runs (newest first) and the one currently running, if any."""
    return {"active_run_id": karpathy_loop.active_run_id(), "runs": karpathy_loop.list_runs()}


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


@router.get("/models")
async def list_models(db: Session = Depends(get_db)):
    """List the baseline plus every saved model version."""
    saved = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    versions = [{
        "version": BASELINE_VERSION,
        "accuracy": None,
        "base_version": None,
        "description": "Default Laya prompt",
        "created_at": None,
        "config": DEFAULT_CONFIG,
    }]
    for model in saved:
        try:
            config = load_version_config(model.version)
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
async def save_model(
    request: SaveModelRequest,
    db: Session = Depends(get_db)
):
    """Save the best prompt found by a Karpathy loop run as a named model version."""
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
        "config": run["best_config"],
        "run_id": run["run_id"],
        "best_iteration": run["best_iteration"],
        "base_version": run["start_version"],
        "dev_accuracy": run["best_accuracy"],
        "holdout_accuracy": run["holdout"]["best"]["accuracy"] if run.get("holdout") else None,
        "llm": run["llm"],
        "saved_at": datetime.utcnow().isoformat(),
    })

    model = ModelVersion(
        id=str(uuid.uuid4()),
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


@router.get("/keywords")
async def get_keywords(
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    """Get paginated list of keywords with their intents."""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")

    if not 1 <= page_size <= 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")

    # Get total count
    total_count = db.query(PredictionRecord).count()

    # Calculate pagination
    skip = (page - 1) * page_size

    # Get paginated results
    records = db.query(PredictionRecord).offset(skip).limit(page_size).all()

    # Format response
    items = []
    for record in records:
        secondary_intents = record.secondary_intents or []
        items.append({
            "id": record.id,
            "keyword": record.query,
            "main_intent": record.predicted_intent,
            "secondary_intents": secondary_intents,
            "confidence": record.confidence,
            "model": record.model,
            "version": record.version,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        })

    return {
        "items": items,
        "pagination": {
            "current_page": page,
            "page_size": page_size,
            "total_items": total_count,
            "total_pages": (total_count + page_size - 1) // page_size,
        }
    }


@router.get("/keywords/stats")
async def get_keywords_stats(db: Session = Depends(get_db)):
    """Get statistics about keywords and intents."""
    from sqlalchemy import func

    total_keywords = db.query(PredictionRecord).count()

    # Count by main intent
    intent_counts = db.query(
        PredictionRecord.predicted_intent,
        func.count(PredictionRecord.id).label('count')
    ).group_by(PredictionRecord.predicted_intent).all()

    intent_distribution = {
        intent: count for intent, count in intent_counts
    }

    return {
        "total_keywords": total_keywords,
        "intent_distribution": intent_distribution,
    }


@router.post("/evaluate")
def run_evaluation(
    request: Optional[EvaluateRequest] = None,
    model: str = "laya",
    version: str = "v1_baseline",
    sample_size: int = 100,
    db: Session = Depends(get_db)
):
    """Run evaluation on keywords in the database with configurable sample size."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    if request is not None:
        model, version = request.model, request.version

    if sample_size < 10 or sample_size > 10000:
        raise HTTPException(status_code=400, detail="Sample size must be between 10 and 10000")

    # Get keywords from database with sample size limit
    records = db.query(PredictionRecord).filter(
        PredictionRecord.version == "v1_test_data"
    ).limit(sample_size).all()

    if not records:
        raise HTTPException(status_code=400, detail="No test data found in database")

    # Run prediction on every keyword based on model
    if model == "laya":
        try:
            config = load_version_config(version)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        predictions = [
            (p["intent"], p["confidence"])
            for p in predict_batch([record.query for record in records], config)
        ]
    else:
        predictions = [
            (intent.value, confidence)
            for intent, confidence in (classify_intent(record.query) for record in records)
        ]

    true_labels = []
    predicted_labels = []
    per_case_results = []

    for record, (pred_intent, confidence) in zip(records, predictions):
        # Get ground truth (main intent is our ground truth)
        true_intent = record.predicted_intent
        true_labels.append(true_intent)
        predicted_labels.append(pred_intent)

        per_case_results.append({
            "keyword": record.query,
            "predicted": pred_intent,
            "actual": true_intent,
            "confidence": confidence,
            "correct": pred_intent == true_intent
        })

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Get all unique intents
    all_intents = sorted(set(true_labels))

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=all_intents, zero_division=0
    )

    per_class_metrics = {}
    for intent, p, r, f, s in zip(all_intents, precision, recall, f1, support):
        per_class_metrics[intent] = {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f), 3),
            "support": int(s)
        }

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=all_intents)
    conf_matrix_dict = {}
    for i, predicted_intent in enumerate(all_intents):
        conf_matrix_dict[predicted_intent] = {}
        for j, actual_intent in enumerate(all_intents):
            conf_matrix_dict[predicted_intent][actual_intent] = int(conf_matrix[j, i])

    # Find errors
    errors = [r for r in per_case_results if not r["correct"]]
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    eval_id = str(uuid.uuid4())

    # Get total golden data count
    total_golden = db.query(PredictionRecord).filter(
        PredictionRecord.version == "v1_test_data"
    ).count()

    return {
        "eval_id": eval_id,
        "model": model,
        "version": version,
        "total_cases": len(records),
        "accuracy": round(accuracy, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix_dict,
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(records), 4),
        "top_errors": errors[:10],
        "timestamp": datetime.utcnow().isoformat(),
        "sample_size": len(records),
        "total_golden_data": total_golden,
    }
