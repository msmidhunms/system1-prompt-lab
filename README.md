# System-1 Model Experiments: SERP Intent Classification

A reusable experiment harness for evaluating System-1 models on the task of classifying search intent from SERP (Search Engine Results Page) results. Starting with Laya, extensible to other models.

## Overview

This project implements a complete ML experiment loop:

1. **Inference**: Run a model (Laya, and later others) on test queries to classify search intent.
2. **Feedback**: Interactively mark predictions as correct/wrong, with top-N confidence-ranked results.
3. **Evaluation**: Run eval across multiple model versions and compare accuracy/metrics.
4. **Karpathy Loop**: Automated iterative improvement (edit classifier → eval → keep if accuracy ↑ → revert else).
5. **Frontend**: Visualize results, compare model versions, inspect per-case differences.

## Architecture

```
backend/              Python FastAPI + SQLite
  app/
    main.py           FastAPI app, routes
    config.py         Settings (env-based)
    models/           Model adapters (abstract + implementations)
    tasks/
      intent/         Intent labels, schemas, task-specific logic
    classifiers/      Versioned classifier files (v1_baseline.py, v2_*.py, ...)
    eval/             Eval runner, metrics, storage
    api/              API routes
  scripts/            CLI tools (classify.py, eval.py)
  tests/              Unit tests
  requirements.txt    Python dependencies

frontend/             React + Vite + TypeScript
  src/
    pages/
      Inference.tsx   Run model, mark right/wrong, show top-N
      Results.tsx     Display top-N correct/wrong, drill into cases
      Eval.tsx        Trigger eval runs
      Compare.tsx     Version/model comparison dashboard
    api.ts            API client
    types.ts          TS types (mirror backend schemas)

data/
  test_cases.jsonl    Test dataset (user-provided)
  
autoresearch/
  program.md          Agent instructions for Karpathy loop
  loop.py             Loop driver
  results.tsv         Iteration history
```

## Intent Labels

Search intent is classified into 4 categories:

- **Informational**: User seeks knowledge/education (e.g., "how to", "what is").
- **Navigational**: User seeks a specific website (e.g., "facebook login", "gmail").
- **Commercial**: User researches products before purchase (e.g., "best laptop", "product reviews").
- **Transactional**: User intends to complete a purchase/action (e.g., "buy laptop", "download pdf").

## Data Format

Test cases are JSONL (one JSON object per line):

```json
{
  "id": "case_001",
  "query": "how to make chocolate cake",
  "serp": [
    {
      "title": "Easy Chocolate Cake Recipe - Tasty",
      "url": "https://www.tastyrecipes.com/chocolate-cake",
      "snippet": "Learn how to make a delicious chocolate cake..."
    },
    ...
  ],
  "intent": "informational"
}
```

Use `data/test_cases_sample.jsonl` as reference. **You should provide your own test cases in `data/test_cases.jsonl`**.

## Key Concepts

### Model-Agnostic Adapter Pattern
Each model (Laya, future ones) has an adapter in `backend/app/models/` implementing:
- `load()`: Load model weights.
- `predict(query, serp) -> (label, confidence)`: Infer intent + score.

New models are registered in a central registry.

### Versioned Classifiers
Each classifier variant is a **single file** (e.g., `v1_baseline.py`, `v2_improved.py`) containing the core logic. This allows:
- Git tracking per version.
- Easy diff/comparison between versions.
- Eval to record git commit SHA for reproducibility.
- Karpathy loop to edit ONE file and keep/revert atomically.

### Top-N Feedback
After inference, extract:
- **Top N correct**: Highest confidence predictions that match ground truth.
- **Top N wrong**: Highest confidence predictions that don't match (most confident mistakes).

This focuses feedback collection on edge cases and hard examples. Configurable via `TOP_N_FEEDBACK` env var (default: 5).

### Eval Harness
For each (model, version, test_set) triple, compute:
- Overall accuracy.
- Per-class precision, recall, F1.
- Confusion matrix.
- Per-case results (query, predicted, actual, confidence).

Store in SQLite with git commit SHA for reproducibility.

### Karpathy Autoresearch Loop
An agent (run separately, configured in `autoresearch/program.md`) iteratively:
1. Edits the current classifier file (e.g., v1_baseline.py).
2. Runs eval.
3. If accuracy on dev set improves → git commit, advance.
4. Else → git revert, try again.
5. Never sees holdout set (10% reserved for honest final metrics).

Logs each iteration to `autoresearch/results.tsv`.

### Cross-Model Comparison
Eval can be run for any (model, version) combination. Frontend compare page groups results by model OR version, showing:
- Accuracy table (rows: versions, cols: models).
- Per-class metrics.
- Case-level diffs (which cases did model A get right but model B got wrong?).

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PyTorch (CPU or GPU; we default to CPU)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
DEBUG=false
TOP_N_FEEDBACK=5
LAYA_MODEL_NAME=layalm/laya1
LAYA_DEVICE=cpu
EVAL_HOLDOUT_RATIO=0.1
```

### Run Backend

```bash
cd backend
export PYTHONPATH=.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Docs: `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:5173`.

## Workflows

### 1. Inference + Feedback

1. Open frontend → **Inference** page.
2. Select model and version.
3. Run inference on your test set.
4. Inspect predictions.
5. Mark results as correct/wrong or correct the label.
6. System extracts top-N correct and top-N wrong by confidence.
7. Feedback stored in SQLite.

**CLI alternative**:
```bash
python scripts/classify.py --model laya --version v1_baseline --input data/test_cases.jsonl --output results.jsonl
```

### 2. Evaluate a Single Version

```bash
python scripts/eval.py --model laya --version v1_baseline --test-file data/test_cases.jsonl
```

Produces metrics JSON and stores in DB.

### 3. Compare Versions (Same Model)

Frontend → **Compare** page → Select model "laya" → See accuracy, confusion matrices across versions.

### 4. Compare Models (Same Version)

Frontend → **Compare** page → Select version "v1_baseline" → See accuracy, confusion matrices across all models.

### 5. Karpathy Loop (Autoresearch)

Once you've created at least 2 classifier versions (e.g., v1_baseline, v2_improved), run:

```bash
cd autoresearch
python loop.py --model laya --version v1_baseline --iterations 10 --eval-budget 10
```

The loop:
- Edits classifier file.
- Runs eval on **dev set only** (90% of data).
- Keeps edits if dev accuracy improves.
- Logs results to `results.tsv`.
- Reports holdout (10%) accuracy at the end.

## Project Structure in Detail

### `backend/app/models/`

- `base.py`: Abstract `ModelAdapter` class.
- `registry.py`: Model registry (mapping name → adapter class).
- `laya.py`: Laya-specific adapter (loads HF model, tokenizes, infers).

### `backend/app/tasks/intent/`

- `labels.py`: `SearchIntent` enum with all 4 labels.
- `schema.py`: Pydantic schemas for test cases, predictions, feedback, eval runs.
- `__init__.py`: Package exports.

### `backend/app/classifiers/`

- `v1_baseline.py`: Initial classifier (e.g., wrap Laya model directly).
- `v2_*.py`: Subsequent versions (prompt engineering, fine-tuning config, etc.).

Each file should export a `classify(query: str, serp: List[SERPResult]) -> Tuple[SearchIntent, float]` function.

### `backend/app/eval/`

- `runner.py`: `EvalRunner` class; computes accuracy, per-class metrics, confusion matrix, case results.
- `store.py`: SQLAlchemy models for runs, predictions, feedback; CRUD functions.

### `backend/app/api/`

- `cases.py`: Endpoints for `/cases` (list, upload, get by ID).
- `infer.py`: Endpoints for `/infer` (run inference on a case or set).
- `feedback.py`: Endpoints for `/feedback` (submit user feedback).
- `eval.py`: Endpoints for `/eval`, `/runs`, `/compare` (trigger eval, fetch results).

### `backend/scripts/`

- `classify.py`: CLI to classify a JSONL file with a given model+version.
- `eval.py`: CLI to eval a single (model, version) pair.

### `autoresearch/`

- `program.md`: Instructions for the autoresearch agent (Karpathy loop logic).
- `loop.py`: Driver script that orchestrates git edits, eval, revert logic, logging.
- `results.tsv`: Tab-separated log of iterations (timestamp, model, version, dev_accuracy, holdout_accuracy, action).

### `frontend/`

Vite + React + TypeScript. Pages:

- **Inference.tsx**: Form to select model+version, input query/SERP, display prediction, mark feedback.
- **Results.tsx**: Show top-N correct/wrong from latest run, drill into individual cases.
- **Eval.tsx**: Trigger eval run for selected model+version, stream results.
- **Compare.tsx**: Multi-version or multi-model comparison dashboard (accuracy table, confusion matrices, per-case diffs).

## API Endpoints (Summary)

### Cases
- `GET /cases` — List all test cases.
- `POST /cases` — Upload test cases (JSONL).
- `GET /cases/{case_id}` — Get a specific case.

### Inference
- `POST /infer` — Run inference on a case or batch.
  - Request: `{model: str, version: str, case_id: str}` or `case_ids: List[str]`.
  - Response: `[{case_id, predicted_intent, confidence, ...}]`.

### Feedback
- `POST /feedback` — Submit user feedback.
  - Request: `{case_id, model, version, feedback: "correct" | "wrong" | "<label>", ...}`.
  - Response: `{success: bool}`.

### Evaluation
- `POST /eval` — Trigger eval run.
  - Request: `{model: str, version: str}`.
  - Response: `{run_id: str, ...metrics...}`.
- `GET /runs` — List eval runs.
- `GET /runs/{run_id}` — Get specific run results.
- `GET /compare` — Compare results across versions or models.
  - Query: `?model=laya&versions=v1_baseline,v2_improved` OR `?version=v1_baseline&models=laya,gpt4`.
  - Response: Grouped metrics for comparison.

## Testing

Run pytest:

```bash
cd backend
pytest tests/ -v
```

Tests should cover:
- Model adapters (mock inference).
- Schema validation.
- Eval metrics (accuracy, confusion matrix computation).
- Top-N selection logic.

## Next Steps (Phases)

1. ✅ **Phase 1: Scaffold + data contract** (this document).
   - You: Review schema, layout, labels. Provide sample test cases if different from `test_cases_sample.jsonl`.

2. **Phase 2: Laya adapter + v1_baseline** (next).
   - We: Implement Laya model adapter and first classifier version.
   - You: Test with sample queries; verify inference works.

3. **Phase 3: Inference loop + feedback + top-N API** (phase 3).
   - We: Implement FastAPI endpoints, SQLite storage.
   - You: curl/CLI walkthrough.

4. **Phase 4: Eval harness + multi-version** (phase 4).
   - We: Implement eval runner and comparison endpoints.
   - You: Create 2nd version (v2), run eval, verify metrics.

5. **Phase 5: Frontend** (phase 5).
   - We: Build React pages (inference, results, eval, compare).
   - You: Click through UI.

6. **Phase 6: Karpathy loop** (phase 6).
   - We: Implement autoresearch loop driver.
   - You: Run dry-run (3 iterations); review kept/reverted changes.

7. **Phase 7: Second model stub** (phase 7).
   - We: Add stub adapter for a second model.
   - You: Verify cross-model comparison works.

## Contributing

When adding a new model:

1. Create `backend/app/models/<model_name>.py` with a class inheriting from `ModelAdapter`.
2. Implement `load()` and `predict(query, serp) -> (label, confidence)`.
3. Register in `backend/app/models/registry.py`.
4. Provide test cases or adapt existing ones for that model's input format.
5. Run eval to benchmark against Laya.

When improving the classifier:

1. Create a new file `backend/app/classifiers/v<N>_<description>.py`.
2. Implement the classify function with your improvements (e.g., prompt engineering, post-processing).
3. Optionally use the autoresearch loop to iterate.

## License

(To be determined)

## Questions?

See `docs/` (to be added) or reach out!
