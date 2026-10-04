"""Database setup and models."""

from sqlalchemy import (
    create_engine, inspect, text, Column, String, Float, Integer, DateTime, Text, Boolean, JSON, UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import DEFAULT_TASK_ID, SQLALCHEMY_DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class GoldenRow(Base):
    """One labelled input of an example's golden dataset."""
    __tablename__ = "golden_rows"
    __table_args__ = (UniqueConstraint("task_id", "input_key", name="uq_golden_rows_task_input"),)

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, index=True)
    input_text = Column(Text, nullable=False)
    # Hash of the input with case and spacing normalised: what decides whether an added row is new.
    input_key = Column(String, nullable=False)
    label = Column(String, nullable=False, index=True)
    secondary_labels = Column(JSON, nullable=True)
    language = Column(String, nullable=True, index=True)  # None when the task has no language filter
    source = Column(String, nullable=False, default="manual")  # 'imported', 'manual' or 'feedback'
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EvaluationRun(Base):
    """A saved model evaluation: its summary columns plus the full result as returned by the API."""
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    version = Column(String, nullable=False)
    sample_size = Column(Integer, nullable=False)
    total_cases = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)
    macro_f1 = Column(Float, nullable=False)
    result = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PredictionRecord(Base):
    """Store live model predictions (the log of what was asked in the Try It tab)."""
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, default=DEFAULT_TASK_ID, index=True)
    # The input text and predicted label. The column names date from when search intent was the only example.
    query = Column(Text, nullable=False, index=True)
    predicted_intent = Column(String, nullable=False)
    secondary_intents = Column(JSON, nullable=True)  # Store list of secondary intents
    language = Column(String, nullable=True, index=True)  # Language of the query wording (e.g. 'en'); None if unknown
    confidence = Column(Float, nullable=False)
    model = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class FeedbackRecord(Base):
    """Store user feedback on predictions."""
    __tablename__ = "feedback"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, default=DEFAULT_TASK_ID, index=True)
    query = Column(Text, nullable=False, index=True)
    predicted_intent = Column(String, nullable=False)
    feedback_type = Column(String, nullable=False)  # 'correct', 'incorrect', 'unsure'
    corrected_intent = Column(String, nullable=True)  # Only if feedback_type != 'correct'
    confidence = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    model = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ModelVersion(Base):
    """Store model versions and their performance."""
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, default=DEFAULT_TASK_ID, index=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False, unique=True)
    accuracy = Column(Float, nullable=False)
    base_version = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_baseline = Column(Boolean, default=False)


class ExperimentRun(Base):
    """Store Karpathy loop experiment runs."""
    __tablename__ = "experiment_runs"

    id = Column(String, primary_key=True)
    task_id = Column(String, nullable=False, default=DEFAULT_TASK_ID, index=True)
    baseline_model = Column(String, nullable=False)
    baseline_version = Column(String, nullable=False)
    baseline_accuracy = Column(Float, nullable=False)
    final_accuracy = Column(Float, nullable=False)
    best_iteration = Column(Integer, nullable=False)
    best_model_name = Column(String, nullable=True)
    improved = Column(Boolean, default=False)
    iterations_data = Column(JSON, nullable=False)  # Store all iteration details
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class AppSetting(Base):
    """Key/value store for app settings (e.g. the LLM used by the Karpathy loop)."""
    __tablename__ = "app_settings"

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def _add_missing_columns():
    """create_all does not alter existing tables, so add columns introduced after a table was created."""
    added = {
        "predictions": {"language": "VARCHAR", "task_id": f"VARCHAR NOT NULL DEFAULT '{DEFAULT_TASK_ID}'"},
        "feedback": {"task_id": f"VARCHAR NOT NULL DEFAULT '{DEFAULT_TASK_ID}'"},
        "model_versions": {"task_id": f"VARCHAR NOT NULL DEFAULT '{DEFAULT_TASK_ID}'"},
        "experiment_runs": {"task_id": f"VARCHAR NOT NULL DEFAULT '{DEFAULT_TASK_ID}'"},
    }
    for table, columns in added.items():
        existing = {column["name"] for column in inspect(engine).get_columns(table)}
        with engine.begin() as conn:
            for name, definition in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_{name} ON {table} ({name})"))


_add_missing_columns()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
