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

## Golden Dataset Labels

The golden dataset is `data/test_db.json` (1,000 search queries with their top results), imported into the `predictions` table with `backend/scripts/import_test_data.py`. Each row carries a primary intent (`main_intent`), optional secondary intents (`foreign_intent`) and the language of the query (`extra.detected_language`, stored in the `language` column).

The provider's intent labels were unreliable: an audit of 100 rows (`data/intent_label_audit_100.csv`) found about half the primary intents wrong or debatable, mostly plain information lookups labelled commercial or transactional. All 1,000 rows were therefore relabelled on 2026-10-04, each judged from the query and its top six results against one rubric. `data/intent_labels.csv` holds, for every row, the reviewed labels and language, the provider's original values, and the kind of query the ruling was based on.

**Only English queries are evaluated or trained on.** The provider's language field was as unreliable as its intents (101 rows flagged non-English, 11 really are), so the language of each query was reviewed too: 989 rows are `en`, 8 `es`, and one each `pt`, `fr` and `am`. A query made only of proper names counts as English, the dataset's search language. `/api/evaluate` and the Karpathy loop use the rows whose `language` equals `EVAL_LANGUAGE` (default `en`). A feedback-only query has no reviewed language, so it is included unless Laya's detector sees another script or accented text.

```bash
python backend/scripts/apply_intent_labels.py                      # apply the reviewed labels and languages (already done)
python backend/scripts/apply_intent_labels.py --restore-original   # put the provider's values back
```

Both commands update `data/test_db.json` and the `predictions` table together. Only the two intent fields and the language field change.

### Labelling rubric

Judge from the query wording first and the top results second. One primary intent, up to two secondary intents.

| Kind of query | Primary | Secondary |
|---|---|---|
| Questions, facts, how-to, meanings, news, weather, health, recipes, homework | informational | |
| Lyrics, cast, recaps, game guides, memes, sports scores and stats, image lookups | informational | |
| Film, show or game title; title + episode or chapter | informational | transactional (watch or buy), navigational |
| Person name, with or without a disambiguator | informational | navigational |
| Organisation + information it hosts (calendar, schedule, programme page, data table) | informational | navigational |
| Ideas and trends for things people pay for; stock quotes; pre-purchase questions | informational | commercial |
| Named business, venue, hotel or attraction, with or without a location | navigational | commercial |
| Church, school, agency, public facility | navigational | informational |
| Business + menu | navigational | commercial, informational |
| Login, portal, a query containing a domain, organisation + jobs, phone number, address of a venue | navigational | informational or transactional |
| Named event | navigational | transactional, informational |
| Explicit destination site in the query (imdb, tiktok, a subreddit) | navigational | by the underlying need |
| Reviews, "vs" for products, best, comparisons | commercial | |
| Product category, branded category, merchandise, "[category] near me / in [city]" | commercial | transactional |
| Specific product where results are mostly info, reviews or community; trading cards | commercial | informational, transactional |
| Photos of a commercial venue; a menu item at a named chain | commercial | navigational, informational |
| Specific product where results are mostly shops; part numbers; product at a named store | transactional | commercial, navigational |
| Buy, for sale, where to buy, tickets, booking, appointment, subscription, coupon | transactional | commercial or navigational |
| Download, install, printable, design assets, online tools; "watch / stream / full" | transactional | informational |

"X reddit" keeps the underlying intent as primary with navigational as secondary, because Reddit is being used as a preferred source rather than a destination page.

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
Modelled on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but the thing being edited is the **prompt Laya is given**, not code. A prompt config (`backend/app/laya_inference.py`) has these parts the LLM may rewrite:

- `state_template`: how the query is rendered into the state passed to `Router.predict`. A string, or an object of field → string that becomes a JSON state (the form Laya's presets use, with the instructions naming the field in backticks). With SERP context enabled it may also use `{serp_sites}`, `{serp_titles}` and `{serp_snippets}`, drawn from the top `serp_results` results in `data/test_db.json`.
- `instructions`: the question text.
- `criteria`: one description per intent label (the labels themselves are fixed).

Two more parts are set by the run, not the LLM: `model` (the Laya checkpoint: `typed-decisions`, `english`, `multilingual` or `auto`) and `label_bias` (per-label offsets fitted on the dev set).

Each round (`backend/app/karpathy_loop.py`):
1. An LLM is shown the current best config, its dev metrics, predicted-label counts, the confusion matrix, gold examples per label, a sample of misclassified queries and the experiment history, and proposes a new config.
2. Laya is run with that config on the dev set.
3. The config is scored by the run's objective. The default is the mean of accuracy and macro-F1, because on this imbalanced dataset plain accuracy rewards putting every query in the majority label.
4. With calibration on, a per-label bias is fitted on the dev set for every config, so a prompt that skews towards one label is judged by how well it separates the labels rather than by which label it favours. The bias is scored cross-fitted (fitted on one half, scored on the other) and is only used when it beats the plain argmax by more than noise.
5. **Keep** only if the score is higher and a paired bootstrap over the dev queries gives at least 0.8 probability that the gain is real. Otherwise **discard**. An unusable LLM reply is a **crash**.
6. Held-out queries (`EVAL_HOLDOUT_RATIO` of the golden data plus whatever the dev sample did not use, never shown to the LLM) are scored once at the end for the baseline and the best config.

Laya keeps only the first 48 tokens of each label description, so the LLM is told to keep them short.

Every run writes to `autoresearch/runs/<run_id>/` (gitignored):

- `run.json` / `run.log`: live run state and log.
- `iter_000_baseline.json`, `iter_NNN.json`: config, metrics, per-query results and the raw LLM reply for each round.
- `iter_NNN_prompt.txt`: the exact prompt sent to the LLM.
- `best_config.json`, `holdout_results.json`, `dev_set.json`, `holdout_set.json`.

One line per experiment is also appended to `autoresearch/results.tsv`. Saving a run's best prompt from the UI writes `autoresearch/checkpoints/<name>.json`; that name can then be used as `version` in `/api/predict` and `/api/evaluate`, or as the starting point of the next run. A saved prompt that uses SERP placeholders only has that context for queries in `data/test_db.json`.

### LLM Configuration
The LLM is chosen on the Karpathy Loop page and stored in the local SQLite DB (`backend/app/llm.py`). Supported: Claude (Anthropic API), OpenAI, Ollama, Google Gemini, any OpenAI-compatible endpoint (OpenRouter, Groq, LM Studio, vLLM, ...), and the Claude Code and Codex CLIs (which use their own login, no API key). API keys can be entered in the UI or supplied through `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` in the backend environment.

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
EVAL_LANGUAGE=en
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

1. Open frontend → **Karpathy Loop** page.
2. Pick the LLM provider and model, then **Save and test**.
3. Set the dev set size, the number of rounds, the version to start from, the Laya checkpoint and the objective, then start the loop.
4. Watch each round arrive as keep / discard / crash. Click a round to see the prompt it proposed.
5. When the run ends, compare dev and holdout accuracy and save the best prompt as a named version.

**API alternative**:
```bash
curl -X POST localhost:8000/api/karpathy-loop -H 'Content-Type: application/json' \
  -d '{"loops": 10, "sample_size": 400, "start_version": "v1_baseline", "laya_model": "typed-decisions", "metric": "balanced", "calibrate": true, "use_serp": false}'
curl localhost:8000/api/karpathy-loop/runs/<run_id>
```

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

- `import_test_data.py`: Import `data/test_db.json` into the `predictions` table as the golden dataset.
- `apply_intent_labels.py`: Apply (or restore) the reviewed intent labels and languages in `data/intent_labels.csv`.
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
