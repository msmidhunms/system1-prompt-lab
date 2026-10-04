# System-1 Model Experiments

An experiment harness for Laya, a fast non-autoregressive "System 1" classifier. Each **example** is a classification task with a golden dataset. For any example you can try single inputs, edit the golden dataset, evaluate a prompt version, and run a Karpathy loop in which an LLM rewrites the prompt Laya is given and keeps only what scores better.

## Examples

Pick the example in the header of the UI; every tab then works on that example. They are defined in `backend/app/tasks/registry.py`.

| Example | Input | Labels | Golden data |
|---|---|---|---|
| `search_intent` | search query | informational, navigational, commercial, transactional | `data/test_db.json`, 1,000 queries, labels reviewed by hand |
| `support_routing` | customer message | card, card_payment, transfer, top_up, cash_withdrawal, account | Banking77 (`mteb/banking77`), 77 intents grouped into 6 |
| `prompt_injection` | prompt | benign, injection (yes/no) | `xTRam1/safe-guard-prompt-injection` |
| `question_type` | question | abbreviation, description, entity, human, location, number | TREC coarse classes (`SetFit/TREC-QC`) |
| `news_topic` | news article | world, sports, business, sci_tech | AG News (`fancyzhx/ag_news`) |
| `email_triage` | email subject and body | legitimate, spam (yes/no) | seven email corpora (`puyang2025/seven-phishing-email-datasets`) |
| `toxicity` | comment | not_toxic, toxic (yes/no) | Jigsaw Wikipedia comments (`OxAISH-AL-LLM/wiki_toxic`, balanced) |
| `request_domain` | request to an LLM | code, math_or_logic, writing, factual_lookup, data_analysis, chitchat | six public datasets, one per label |

The yes/no examples use Laya's native `noul` question type; the others use `choice`.

Two of the datasets are weaker than the rest. `request_domain` is labelled by which source a request came from (MBPP, GSM8K, Dolly, SQL questions, Persona-Chat), so each label has its own writing style and the task is easy. `email_triage` has no separate phishing label, because the public "phishing" sets that were checked turned out to be ordinary spam.

### Importing the datasets

The seven new examples download their data from Hugging Face. 2,000 rows are sampled per example (seeded), kept under `data/tasks/` (gitignored) and reused on later imports.

```bash
python backend/scripts/import_dataset.py --task all
python backend/scripts/import_dataset.py --task support_routing --rows 3000 --refresh
```

The Golden Dataset tab has an **Import dataset** button that does the same for an example with no rows. An import never overwrites a row that is already in the golden dataset.

### Adding an example

1. Add a `Task` to `backend/app/tasks/registry.py`: its state field, labels, default prompt and the paragraph the Karpathy loop is told about the task.
2. Add a loader to `backend/app/tasks/sources.py` if the data is downloadable, or add rows by hand in the Golden Dataset tab.

Nothing else is example-specific: inference, scoring, the loop, the API and the UI all read the registry.

## Architecture

```
backend/                  Python FastAPI + SQLite (backend/data.db)
  app/
    main.py               FastAPI app
    api.py                All routes; each takes the example as `task`
    config.py             Settings (env-based)
    database.py           Tables: golden_rows, evaluation_runs, predictions, feedback,
                          model_versions, experiment_runs, app_settings
    tasks/
      registry.py         The examples
      sources.py          Dataset loaders and the import
      intent/             Search-intent label enum and schemas
    laya_inference.py     Prompt configs, Laya calls, saved versions
    scoring.py            Accuracy, macro-F1, label-bias fit, paired bootstrap
    golden.py             Golden rows: list, add-or-update, edit
    karpathy_loop.py      The autoresearch loop
    llm.py                The LLM that proposes prompts
    serp.py               Search-result context for search intent
  scripts/
    import_dataset.py     Import an example's golden dataset
    import_test_data.py   Same, for search intent only
    apply_intent_labels.py  Apply the reviewed search-intent labels
    migrate_to_tasks.py   One-off migration to the multi-example layout
  tests/                  pytest

frontend/                 React + Vite + TypeScript
  src/
    api.ts                API client and shared types
    TaskContext.tsx       The selected example
    components/           TryIt, GoldenDataset, Evaluation, KarpathyLoop, LLMSettings

data/
  test_db.json            Search-intent queries with their search results (gitignored)
  tasks/                  Downloaded samples of the other examples (gitignored)

autoresearch/
  runs/<run_id>/          Everything a Karpathy loop run produced
  checkpoints/            Saved prompt versions
  results.tsv             One line per experiment
```

## The four tabs

- **Try It**: classify one input with any saved version and mark the prediction right or wrong. Feedback that settles the label is written to the golden dataset.
- **Golden Dataset**: browse, search, filter, edit and delete the example's labelled rows, and add new ones.
- **Model Evaluation**: score a version on a seeded random sample of the golden dataset. Every evaluation is saved and listed under Past Evaluations.
- **Karpathy Loop**: run the prompt autoresearch loop and open past runs.

### Adding a golden row

A row is identified by its example and its input. When a row is added, the input is compared with the existing ones ignoring upper/lower case and extra spaces: if it is already there, that row's label is updated; otherwise a new row is created. The response says which happened (`status: "created"` or `"updated"`, with `previous_label`).

```bash
curl -X POST localhost:8000/api/golden-data -H 'Content-Type: application/json' \
  -d '{"task": "search_intent", "text": "best laptop 2024", "label": "commercial"}'
```

Edits made in the UI change the database only. For search intent they are not written back to `data/test_db.json` or `data/intent_labels.csv`.

## Search Intent: Golden Dataset Labels

The search-intent golden dataset is `data/test_db.json` (1,000 search queries with their top results), imported into the `golden_rows` table with `backend/scripts/import_test_data.py`. Each row carries a primary intent (`main_intent`), optional secondary intents (`foreign_intent`) and the language of the query (`extra.detected_language`, stored in the `language` column).

The provider's intent labels were unreliable: an audit of 100 rows (`data/intent_label_audit_100.csv`) found about half the primary intents wrong or debatable, mostly plain information lookups labelled commercial or transactional. All 1,000 rows were therefore relabelled on 2026-10-04, each judged from the query and its top six results against one rubric. `data/intent_labels.csv` holds, for every row, the reviewed labels and language, the provider's original values, and the kind of query the ruling was based on.

**Only English queries are evaluated or trained on.** The provider's language field was as unreliable as its intents (101 rows flagged non-English, 11 really are), so the language of each query was reviewed too: 989 rows are `en`, 8 `es`, and one each `pt`, `fr` and `am`. A query made only of proper names counts as English, the dataset's search language. `/api/evaluate` and the Karpathy loop use the rows whose `language` equals `EVAL_LANGUAGE` (default `en`). A feedback-only query has no reviewed language, so it is included unless Laya's detector sees another script or accented text.

```bash
python backend/scripts/apply_intent_labels.py                      # apply the reviewed labels and languages (already done)
python backend/scripts/apply_intent_labels.py --restore-original   # put the provider's values back
```

Both commands update `data/test_db.json` and the imported search-intent rows of the `golden_rows` table together. Only the two intent fields and the language field change. Rows added by hand or through feedback are left alone.

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

### Karpathy Autoresearch Loop
Modelled on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but the thing being edited is the **prompt Laya is given**, not code. It works the same way for every example. A prompt config (`backend/app/laya_inference.py`) has these parts the LLM may rewrite:

- `state_template`: how the input is rendered into the state passed to `Router.predict`. A string, or an object of field → string that becomes a JSON state (the form Laya's presets use, with the instructions naming the field in backticks). The input goes where the example's placeholder is: `{query}` for search intent, `{message}` for support routing, and so on. For search intent with SERP context enabled it may also use `{serp_sites}`, `{serp_titles}` and `{serp_snippets}`, drawn from the top `serp_results` results in `data/test_db.json`.
- `instructions`: the question text.
- `criteria`: one description per label (the labels themselves are fixed). For a yes/no example these are the descriptions of the "no" and the "yes" side.

Two more parts are set by the run, not the LLM: `model` (the Laya checkpoint: `typed-decisions`, `english`, `multilingual` or `auto`) and `label_bias` (per-label offsets fitted on the dev set).

Each round (`backend/app/karpathy_loop.py`):
1. An LLM is shown the current best config, its dev metrics, predicted-label counts, the confusion matrix, gold examples per label, a sample of misclassified queries and the experiment history, and proposes a new config.
2. Laya is run with that config on the dev set.
3. The config is scored by the run's objective (`backend/app/scoring.py`). The default is the mean of accuracy and macro-F1, because on an imbalanced dataset plain accuracy rewards putting every input in the majority label.
4. With calibration on, a per-label bias is fitted on the dev set for every config, so a prompt that skews towards one label is judged by how well it separates the labels rather than by which label it favours. The bias is scored cross-fitted (fitted on one half, scored on the other) and is only used when it beats the plain argmax by more than noise.
5. **Keep** only if the score is higher and a paired bootstrap over the dev queries gives at least 0.8 probability that the gain is real. Otherwise **discard**. An unusable LLM reply is a **crash**.
6. Held-out queries (`EVAL_HOLDOUT_RATIO` of the golden data plus whatever the dev sample did not use, never shown to the LLM) are scored once at the end for the baseline and the best config.

Laya keeps only the first 48 tokens of each label description, and the instructions and all descriptions share about 190 tokens, so the LLM is told to keep them short (shorter still for six-label examples). Inputs longer than about 1,500 characters are cut before they reach Laya.

Only one loop runs at a time, across all examples.

Every run writes to `autoresearch/runs/<run_id>/` (gitignored):

- `run.json` / `run.log`: live run state and log.
- `iter_000_baseline.json`, `iter_NNN.json`: config, metrics, per-query results and the raw LLM reply for each round.
- `iter_NNN_prompt.txt`: the exact prompt sent to the LLM.
- `best_config.json`, `holdout_results.json`, `dev_set.json`, `holdout_set.json`.

One line per experiment is also appended to `autoresearch/results.tsv`. Saving a run's best prompt from the UI writes `autoresearch/checkpoints/<name>.json`; that name can then be used as `version` in `/api/predict` and `/api/evaluate` for the same example, or as the starting point of its next run. Version names are unique across examples. A saved prompt that uses SERP placeholders only has that context for queries in `data/test_db.json`.

### LLM Configuration
The LLM is chosen on the Karpathy Loop page and stored in the local SQLite DB (`backend/app/llm.py`). Supported: Claude (Anthropic API), OpenAI, Ollama, Google Gemini, any OpenAI-compatible endpoint (OpenRouter, Groq, LM Studio, vLLM, ...), and the Claude Code and Codex CLIs (which use their own login, no API key). API keys can be entered in the UI or supplied through `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` in the backend environment.

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

### Load the data

```bash
python backend/scripts/import_dataset.py --task all
```

A database created before the examples existed is upgraded once with `python backend/scripts/migrate_to_tasks.py`. It backs the database up to `backend/data.db.bak-pre-tasks`, moves the search-intent golden rows from the `predictions` table into `golden_rows`, applies the recorded feedback on top, and leaves live predictions where they are. Runs under `autoresearch/runs/` and saved versions need no migration: anything without an example recorded belongs to `search_intent`.

## Workflows

### Evaluate a version

Model Evaluation tab, or:

```bash
curl -X POST localhost:8000/api/evaluate -H 'Content-Type: application/json' \
  -d '{"task": "support_routing", "version": "v1_baseline", "sample_size": 200}'
curl 'localhost:8000/api/evaluations?task=support_routing'      # past evaluations
```

### Karpathy loop

1. Open the **Karpathy Loop** tab for the example.
2. Pick the LLM provider and model, then **Save and test**.
3. Set the dev set size, the number of rounds, the version to start from, the Laya checkpoint and the objective, then start the loop.
4. Watch each round arrive as keep / discard / crash. Click a round to see the prompt it proposed.
5. When the run ends, compare dev and holdout accuracy and save the best prompt as a named version.

```bash
curl -X POST localhost:8000/api/karpathy-loop -H 'Content-Type: application/json' \
  -d '{"task": "support_routing", "loops": 10, "sample_size": 400, "start_version": "v1_baseline", "laya_model": "typed-decisions", "metric": "balanced", "calibrate": true}'
curl localhost:8000/api/karpathy-loop/runs/<run_id>
```

## API Endpoints (Summary)

Every endpoint takes the example as `task` (query parameter on GET, body field on POST). Leaving it out means `search_intent`. Full docs: `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /api/tasks` | The examples, their labels and golden row counts |
| `POST /api/tasks/{task}/import` | Import the example's dataset |
| `POST /api/predict` | Classify one input |
| `POST /api/feedback` | Record feedback; updates the golden dataset when it settles the label |
| `GET /api/golden-data` | One page of golden rows (`page`, `page_size`, `q`, `label`, `source`) |
| `GET /api/golden-data/stats` | Row counts by label, source and language |
| `POST /api/golden-data` | Add a row, or update the row with the same input |
| `PUT /api/golden-data/{id}`, `DELETE /api/golden-data/{id}` | Edit or delete a row |
| `POST /api/evaluate` | Evaluate a version and save the result |
| `GET /api/evaluations`, `GET /api/evaluations/{id}` | Past evaluations |
| `POST /api/karpathy-loop` | Start a loop |
| `GET /api/karpathy-loop/runs`, `GET /api/karpathy-loop/runs/{id}` | Past and current runs |
| `POST /api/karpathy-loop/runs/{id}/stop` | Stop after the current round |
| `GET /api/models`, `POST /api/save-model` | Saved prompt versions |
| `GET/PUT /api/llm/config`, `POST /api/llm/test`, `POST /api/llm/models` | LLM settings |

## Testing

```bash
cd backend
pytest tests/ -v
```

The tests use a throwaway database (`DB_PATH`) and a stub in place of the Laya model.

## License

(To be determined)
