# System 1 Prompt Lab

A workbench for small, fast "System 1" classifiers: models that make a decision in one forward pass instead of generating text. It runs seven of them (Laya, Verdict, GLiClass, three DeBERTa NLI classifiers and ModernBERT-Instruct), all on a laptop CPU.

These models are steered by a short prompt: a question and one description per label. The lab is for finding out how good a prompt is, on which model, and for making it better:

- **Golden datasets** for eight classification tasks (support-ticket routing, prompt-injection detection, spam, toxicity, and more), editable in the app.
- **Evaluation** of any prompt version, with per-label metrics, a confusion matrix and the most confident errors.
- **Model comparison**: one prompt, the same sample, several models side by side on accuracy and speed.
- **Prompt Optimizer**: an LLM rewrites the prompt, each proposal is scored, and only gains that are larger than noise are kept (Karpathy's autoresearch loop, applied to prompts instead of code).
- **Model versions**: every improvement is saved, and prompts can be copied, edited and renamed.

FastAPI and SQLite on the back, React on the front. The classifiers run locally; the only outside call is to the LLM that proposes prompts in the optimizer, and that can be a local one too (Ollama).

Each **example** is a classification task with a golden dataset. For any example you can try single inputs, edit the golden dataset, manage prompt versions ("models"), evaluate them, compare models, and run the Prompt Optimizer.

## Examples

Pick the example in the sidebar; every tab then works on that example. They are defined in `backend/app/tasks/registry.py`.

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

Two of the datasets are weaker than the rest. `request_domain` is labelled by which source a request came from (MBPP, GSM8K, Dolly, SQL questions, Persona-Chat), so a label can be recognised from its source's writing style rather than from what is being asked. `email_triage` has no separate phishing label, because the public "phishing" sets that were checked turned out to be ordinary spam.

### Importing the datasets

The seven new examples download their data from Hugging Face. 2,000 rows are sampled per example (seeded), kept under `data/tasks/` (gitignored) and reused on later imports.

```bash
python backend/scripts/import_dataset.py --task all
python backend/scripts/import_dataset.py --task support_routing --rows 3000 --refresh
```

In the app, **Set up data** in the sidebar lists every example with its status and downloads the missing public datasets with one button; an empty example's Dataset tab has the same button. An import never overwrites a row that is already in the golden dataset.

Search intent is the exception: its data (`data/test_db.json`) is not in the repository and cannot be downloaded. Put the file in `data/` and run `python backend/scripts/import_dataset.py --task search_intent`. Without it the example still works with rows added by hand.

### Adding an example

1. Add a `Task` to `backend/app/tasks/registry.py`: its state field, labels, default prompt and the paragraph the Karpathy loop is told about the task.
2. Add a loader to `backend/app/tasks/sources.py` if the data is downloadable, or add rows by hand in the Dataset tab.

Nothing else is example-specific: inference, scoring, the loop, the API and the UI all read the registry.

## Models

Every example can be run on seven models. Laya is the default; the others are open zero-shot classifiers that take the same prompt (the input, a question, one description per label) in their own input format (`backend/app/engines.py`).

| Model | Size | How it reads the prompt |
|---|---|---|
| Laya | – | Native: JSON state, instructions, option descriptions; yes/no questions as `noul` |
| GLiClass modern-base v2.0 (`knowledgator/gliclass-modern-base-v2.0`) | 151M | All descriptions and the text in one pass; the instructions go before the text |
| Verdict 1.4 / OpenJev (`heman10x/rlcd-modernbert-151m`) | 151M | GLiClass backbone; each description becomes "It is …", the instructions become "Question: …", and the model's own temperature calibration and abstention option are applied |
| DeBERTa-v3 xsmall zero-shot v1.1 (`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`) | 70.8M | NLI: one pass per label; instructions containing `{}` are the hypothesis template, otherwise "This text is about {}." |
| DeBERTa small long NLI (`tasksource/deberta-small-long-nli`) | 142M | NLI, as above |
| DeBERTa-v3 large zero-shot v2.0 (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) | 435M | NLI, as above |
| ModernBERT-Large-Instruct (`answerdotai/ModernBERT-Large-Instruct`) | 396M | A multiple-choice prompt answered by the masked-language head |

A model's weights are downloaded from Hugging Face the first time it is used, and only one of the six non-Laya models is kept in memory at a time. They run on the CPU unless `ENGINE_DEVICE` says otherwise.

A model version is saved for one model. The Playground, Evaluation and the Prompt Optimizer can each run a version's prompt on another model; a label bias or Laya checkpoint does not carry over when they do. The optimizer is told how the chosen model reads a prompt, so it can be used to tune a prompt for any of the seven.

The **Compare Models** tab runs one prompt on several models over the same sample and shows accuracy, macro-F1 and time per input side by side.

```bash
curl -X POST localhost:8000/api/compare -H 'Content-Type: application/json' \
  -d '{"task": "support_routing", "version": "v1_baseline", "engines": ["laya", "gliclass", "verdict"], "sample_size": 100}'
curl 'localhost:8000/api/comparisons?task=support_routing'
```

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
      sources.py          Dataset loaders and the (background) import
      intent/             Search-intent label enum and schemas
    laya_inference.py     Prompt configs and Laya calls
    engines.py            The other models: GLiClass, Verdict, NLI classifiers, ModernBERT-Instruct
    versions.py           Saved model versions: create, rename, delete, auto-save
    scoring.py            Accuracy, macro-F1, label-bias fit, paired bootstrap
    evaluations.py        Evaluations as background jobs
    golden.py             Golden rows: list, add-or-update, edit
    karpathy_loop.py      The optimizer loop
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
    context.tsx           App state (examples, LLM settings, background activity) and routing
    ui/                   Shared components (Button, Card, Menu, Dialog, ...)
    styles/               Design tokens (light and dark) and base styles
    components/           Playground, Dataset, Models, Evaluation, optimizer/, dialogs

data/
  test_db.json            Search-intent queries with their search results (gitignored)
  tasks/                  Downloaded samples of the other examples (gitignored)

autoresearch/
  runs/<run_id>/          Everything an optimizer run produced
  checkpoints/            Saved model versions
  results.tsv             One line per experiment
```

## The six tabs

- **Playground**: classify one input with any model version and mark the prediction right or wrong. Feedback that settles the label is written to the golden dataset.
- **Dataset**: browse, search, filter, edit and delete the example's labelled rows, and add new ones.
- **Models**: the example's prompt versions. View a prompt, duplicate and edit it into a new version, rename, delete, or send it to Evaluation or the Playground.
- **Evaluation**: score a version on a seeded random sample of the golden dataset. Every evaluation is saved and listed.
- **Compare Models**: one prompt, the same sample, several models side by side.
- **Prompt Optimizer**: run the prompt autoresearch loop and open past runs.

Evaluations, optimizer runs and dataset imports run on the server. You can switch tab or example, or reload the page, and find them where they were; the header shows what is running. The place in the app is kept in the URL (`#/support_routing/optimizer`).

The LLM used by the optimizer is one setting for the whole app (**Settings** in the header), stored on the server.

### Model versions

A model version is a named prompt for one example: the state template, the instructions, one description per label, the Laya checkpoint and an optional label bias. Versions come from three places:

- **Auto-saved**: each optimizer run saves its best prompt as one version (`<example>_<MMDD-HHMM>`) the moment a round is kept, and overwrites it when a later round is better. It is there even if the run is stopped or fails, and can be renamed at any time.
- **Optimizer**: any round of a run, kept or discarded, saved by hand from the round's menu.
- **Manual**: **Duplicate and edit** on any version opens an editor; saving creates a new version and leaves the original untouched.

Renaming a version also renames it in saved evaluations and in versions based on it.

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

### The Prompt Optimizer (Karpathy autoresearch loop)
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
5. **Keep** only if the score is higher and a paired bootstrap over the dev queries gives at least 0.8 probability that the gain is real. Otherwise **discard**. An unusable LLM reply or a failed LLM call is a failed round; three in a row end the proposing early, and the run finishes as `failed` with what it found still scored and saved.
6. Held-out queries (`EVAL_HOLDOUT_RATIO` of the golden data plus whatever the dev sample did not use, never shown to the LLM) are scored once at the end for the baseline and the best config.

Laya keeps only the first 48 tokens of each label description, and the instructions and all descriptions share about 190 tokens, so the LLM is told to keep them short (shorter still for six-label examples). Inputs longer than about 1,500 characters are cut before they reach Laya.

Only one loop runs at a time, across all examples.

Every run writes to `autoresearch/runs/<run_id>/` (gitignored):

- `run.json` / `run.log`: live run state and log.
- `iter_000_baseline.json`, `iter_NNN.json`: config, metrics, per-query results and the raw LLM reply for each round.
- `iter_NNN_prompt.txt`: the exact prompt sent to the LLM.
- `best_config.json`, `holdout_results.json`, `dev_set.json`, `holdout_set.json`.

One line per experiment is also appended to `autoresearch/results.tsv`. A saved version (the run's auto-saved best, or a round saved from the UI) is written to `autoresearch/checkpoints/<name>.json`; that name can then be used as `version` in `/api/predict` and `/api/evaluate` for the same example, or as the starting point of its next run. Version names are unique across examples. A saved prompt that uses SERP placeholders only has that context for queries in `data/test_db.json`.

### LLM Configuration
The LLM is chosen in **Settings** and stored in the local SQLite DB (`backend/app/llm.py`). Supported: Claude (Anthropic API), OpenAI, Ollama, Google Gemini, any OpenAI-compatible endpoint (OpenRouter, Groq, LM Studio, vLLM, ...), and the Claude Code and Codex CLIs (which use their own login, no API key). API keys can be entered in the UI or supplied through `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` in the backend environment.

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

Optional: copy `backend/.env.example` to `backend/.env` and adjust. Everything has a default, so the app runs without it. The Laya weights are downloaded from Hugging Face the first time a prediction is made.

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

The frontend talks to `http://localhost:8000/api`; set `VITE_API_URL` to point it elsewhere.

### Load the data

A fresh clone has no golden data: the database and the datasets are not in the repository. Open the app and use **Set up data** in the sidebar, or:

```bash
python backend/scripts/import_dataset.py --task all
```

A database created before the examples existed is upgraded once with `python backend/scripts/migrate_to_tasks.py`. It backs the database up to `backend/data.db.bak-pre-tasks`, moves the search-intent golden rows from the `predictions` table into `golden_rows`, applies the recorded feedback on top, and leaves live predictions where they are. Runs under `autoresearch/runs/` and saved versions need no migration: anything without an example recorded belongs to `search_intent`.

## Workflows

### Evaluate a version

Evaluation tab, or:

```bash
curl -X POST localhost:8000/api/evaluate -H 'Content-Type: application/json' \
  -d '{"task": "support_routing", "version": "v1_baseline", "sample_size": 200}'   # returns the id, status "running"
curl localhost:8000/api/evaluations/<eval_id>                    # progress, then the result
curl 'localhost:8000/api/evaluations?task=support_routing'      # past evaluations
```

### Optimize a prompt

1. Choose the LLM once in **Settings** (provider, model, key), then **Save and test**.
2. Open the example's **Prompt Optimizer** tab, set the rounds, the dev set size, the version to start from, the Laya checkpoint and the objective, and start the run.
3. Watch each round arrive as kept / discarded / failed, with the score chart. Click a round to see the prompt it proposed.
4. The best prompt is saved automatically; rename it, evaluate it, or save any other round from its menu.

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
| `POST /api/setup/import` | Download and import datasets in the background (`tasks`: the examples; empty means all that have no data) |
| `GET /api/activity` | What is running: the optimizer run, evaluations, imports |
| `POST /api/predict` | Classify one input |
| `POST /api/feedback` | Record feedback; updates the golden dataset when it settles the label |
| `GET /api/golden-data` | One page of golden rows (`page`, `page_size`, `q`, `label`, `source`) |
| `GET /api/golden-data/stats` | Row counts by label, source and language |
| `POST /api/golden-data` | Add a row, or update the row with the same input |
| `PUT /api/golden-data/{id}`, `DELETE /api/golden-data/{id}` | Edit or delete a row |
| `POST /api/evaluate` | Start an evaluation in the background |
| `GET /api/engines` | The models an example can be run on |
| `POST /api/compare`, `GET /api/comparisons` | Evaluate one prompt on several models; past comparisons |
| `GET /api/evaluations`, `GET /api/evaluations/{id}` | Evaluations with status and progress; the full result once completed |
| `POST /api/karpathy-loop` | Start a loop |
| `GET /api/karpathy-loop/runs`, `GET /api/karpathy-loop/runs/{id}` | Past and current runs |
| `POST /api/karpathy-loop/runs/{id}/stop` | Stop after the current round |
| `GET /api/models`, `POST /api/models` | List model versions; create one from a prompt config |
| `PATCH /api/models/{name}`, `DELETE /api/models/{name}` | Rename or delete a version |
| `POST /api/save-model` | Save a run's best prompt, or the prompt of one round (`iteration`), as a version |
| `GET/PUT /api/llm/config`, `POST /api/llm/test`, `POST /api/llm/models` | LLM settings |

## Testing

```bash
cd backend
pytest tests/ -v
```

The tests use a throwaway database and run folder (`DB_PATH`, `DATA_DIR`, `AUTORESEARCH_DIR`), a stub in place of the Laya model and a scripted LLM.

## License

(To be determined)
