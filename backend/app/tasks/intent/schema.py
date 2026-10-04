"""Pydantic schemas for intent classification task."""

from pydantic import BaseModel, Field
from typing import Optional, List
from .labels import SearchIntent


class SERPResult(BaseModel):
    """A single search result in the SERP."""

    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Page URL")
    snippet: str = Field(..., description="Search snippet/preview text")


class TestCase(BaseModel):
    """A test case for intent classification."""

    id: str = Field(..., description="Unique case identifier")
    query: str = Field(..., description="Search query")
    serp: List[SERPResult] = Field(..., description="SERP results for this query")
    intent: SearchIntent = Field(..., description="Ground truth intent label")

    class Config:
        use_enum_values = True


class Prediction(BaseModel):
    """Model prediction for a test case."""

    case_id: str = Field(..., description="Test case ID")
    predicted_intent: SearchIntent = Field(..., description="Predicted intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")
    model: str = Field(..., description="Model name (e.g., 'laya')")
    version: str = Field(..., description="Version/variant ID (e.g., 'v1_baseline')")

    class Config:
        use_enum_values = True


class FeedbackEntry(BaseModel):
    """User feedback on a prediction."""

    case_id: str = Field(..., description="Test case ID")
    prediction_id: Optional[str] = Field(None, description="Prediction ID if from inference")
    model: str = Field(..., description="Model name")
    version: str = Field(..., description="Version ID")
    feedback: str = Field(..., description="'correct', 'wrong', or corrected label (enum value)")
    corrected_intent: Optional[SearchIntent] = Field(None, description="If feedback is a corrected label")
    notes: Optional[str] = Field(None, description="User notes")

    class Config:
        use_enum_values = True


class EvalRun(BaseModel):
    """Results of an eval run."""

    run_id: str = Field(..., description="Unique run ID (timestamp or uuid)")
    model: str = Field(..., description="Model name")
    version: str = Field(..., description="Version ID")
    git_commit: str = Field(..., description="Git commit SHA when this version was evaluated")
    total_cases: int = Field(..., description="Total test cases evaluated")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy")
    per_class_metrics: dict = Field(..., description="Precision/Recall/F1 per intent label")
    confusion_matrix: dict = Field(..., description="Confusion matrix as {predicted: {actual: count}}")
    case_results: List[dict] = Field(..., description="Per-case results with pred, actual, confidence")
    created_at: str = Field(..., description="ISO timestamp")

    class Config:
        use_enum_values = True
