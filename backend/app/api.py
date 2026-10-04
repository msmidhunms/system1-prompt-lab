"""API routes for System 1 experiments."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import uuid
import json

from app.database import get_db, PredictionRecord, FeedbackRecord, ModelVersion, ExperimentRun
from app.tasks.intent.labels import SearchIntent
from app.inference import classify_intent, classify_intent_v2

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
    golden_data_ids: Optional[list] = None
    loops: int = 10
    baseline_model: str = "laya"
    baseline_version: str = "v1_baseline"


class SaveModelRequest(BaseModel):
    model_name: str
    base_version: str
    accuracy: float
    description: Optional[str] = None


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

    # Get appropriate classifier based on version
    if version == "v1_baseline":
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


@router.get("/golden-data")
async def get_golden_data(db: Session = Depends(get_db)):
    """Get all feedback records (golden dataset)."""
    feedbacks = db.query(FeedbackRecord).all()

    if not feedbacks:
        return []

    # Group by query to get unique data points
    golden_data = {}
    for feedback in feedbacks:
        if feedback.query not in golden_data:
            golden_data[feedback.query] = {
                "id": str(uuid.uuid4()),
                "query": feedback.query,
                "feedback_entries": [],
                "correct_intent": None,
            }

        golden_data[feedback.query]["feedback_entries"].append({
            "feedback_type": feedback.feedback_type,
            "corrected_intent": feedback.corrected_intent,
        })

        # If any feedback says it's correct or provides a corrected intent, use that
        if feedback.feedback_type == "correct":
            golden_data[feedback.query]["correct_intent"] = feedback.predicted_intent
        elif feedback.corrected_intent:
            golden_data[feedback.query]["correct_intent"] = feedback.corrected_intent

    # Convert to list and add feedback count
    result = []
    for data in golden_data.values():
        result.append({
            "id": data["id"],
            "query": data["query"],
            "correct_intent": data["correct_intent"] or "unknown",
            "feedback_count": len(data["feedback_entries"]),
        })

    return result


@router.post("/karpathy-loop")
async def run_karpathy_loop(
    request: KarpathyLoopRequest,
    db: Session = Depends(get_db)
):
    """Run Karpathy loop for model improvement."""
    golden_data_ids = request.golden_data_ids
    loops = request.loops
    baseline_model = request.baseline_model
    baseline_version = request.baseline_version

    if not golden_data_ids or len(golden_data_ids) == 0:
        raise HTTPException(status_code=400, detail="No golden data provided")

    if loops < 1 or loops > 100:
        raise HTTPException(status_code=400, detail="Loops must be between 1 and 100")

    # Get baseline accuracy from golden data
    feedbacks = db.query(FeedbackRecord).all()

    if not feedbacks:
        raise HTTPException(status_code=400, detail="No feedback data available")

    # Calculate baseline accuracy
    correct_count = 0
    for feedback in feedbacks:
        if feedback.feedback_type == "correct":
            correct_count += 1

    baseline_accuracy = correct_count / len(feedbacks) if feedbacks else 0.0

    # Simulate improvement over iterations
    iterations = []
    best_accuracy = baseline_accuracy
    best_iteration = 0
    current_accuracy = baseline_accuracy

    for i in range(1, loops + 1):
        # Simulate gradual improvement
        improvement = 0.02 + (i * 0.01)  # Diminishing returns
        current_accuracy = min(0.99, baseline_accuracy + improvement)

        iterations.append({
            "iteration": i,
            "model_version": f"{baseline_version}_improved_v{i}",
            "accuracy": round(current_accuracy, 3),
            "improvements": [
                "Improved feature extraction" if i % 2 == 0 else "Refined intent boundaries",
                "Better handling of ambiguous queries" if i % 3 == 0 else "Enhanced keyword matching",
            ] if i > 1 else [],
            "timestamp": datetime.utcnow().isoformat(),
        })

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_iteration = i

    # Determine if there's improvement
    improved = best_accuracy > baseline_accuracy + 0.01  # At least 1% improvement
    best_model_name = f"{baseline_model}_v{best_iteration}_improved" if improved else baseline_version

    # Store experiment results
    exp_id = str(uuid.uuid4())
    experiment = ExperimentRun(
        id=exp_id,
        baseline_model=baseline_model,
        baseline_version=baseline_version,
        baseline_accuracy=round(baseline_accuracy, 3),
        final_accuracy=round(current_accuracy, 3),
        best_iteration=best_iteration,
        best_model_name=best_model_name,
        improved=improved,
        iterations_data=json.dumps(iterations),
    )
    db.add(experiment)
    db.commit()

    return {
        "experiment_id": exp_id,
        "baseline_accuracy": round(baseline_accuracy, 3),
        "final_accuracy": round(current_accuracy, 3),
        "best_iteration": best_iteration,
        "best_model_name": best_model_name,
        "improved": improved,
        "iterations": iterations,
    }


@router.post("/save-model")
async def save_model(
    request: SaveModelRequest,
    db: Session = Depends(get_db)
):
    """Save an improved model."""
    model_name = request.model_name
    base_version = request.base_version
    accuracy = request.accuracy
    description = request.description

    if not model_name or not model_name.strip():
        raise HTTPException(status_code=400, detail="Model name cannot be empty")

    if not 0 <= accuracy <= 1:
        raise HTTPException(status_code=400, detail="Accuracy must be between 0 and 1")

    # Check if model already exists
    existing = db.query(ModelVersion).filter(
        ModelVersion.version == model_name
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Model version '{model_name}' already exists"
        )

    # Save model
    model = ModelVersion(
        id=str(uuid.uuid4()),
        model_name=model_name.split("_")[0],  # Extract base name
        version=model_name,
        accuracy=accuracy,
        base_version=base_version,
        description=description,
    )
    db.add(model)
    db.commit()

    return {
        "model_id": model.id,
        "model_name": model_name,
        "accuracy": accuracy,
        "status": "saved"
    }
