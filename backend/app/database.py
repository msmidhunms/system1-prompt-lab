"""Database setup and models."""

from sqlalchemy import create_engine, inspect, text, Column, String, Float, Integer, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import SQLALCHEMY_DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    """Store model predictions."""
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    query = Column(String, nullable=False, index=True)
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
    query = Column(String, nullable=False, index=True)
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
    existing = {column["name"] for column in inspect(engine).get_columns("predictions")}
    if "language" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN language VARCHAR"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_predictions_language ON predictions (language)"))


_add_missing_columns()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
