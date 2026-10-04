"""Karpathy-style autoresearch loop over Laya's prompt, for any example in app.tasks.registry.

Same shape as https://github.com/karpathy/autoresearch, but the thing being
edited is the Laya prompt config (state template, instructions, label
descriptions) instead of a training script:

    propose a change (LLM) -> evaluate on the dev set -> keep if the score
    improved by more than noise, otherwise discard -> repeat.

Three things keep the loop honest on a small, imbalanced dataset:

- The objective defaults to the mean of accuracy and macro-F1, so a prompt that
  puts every query in the majority label does not look like progress.
- With calibration on, a per-label bias is fitted on the dev set for every
  candidate, so a prompt that skews towards one label is judged by how well its
  probabilities separate the labels and not by which label it happens to favour.
  The bias is scored cross-fitted (fitted on one half, scored on the other) and
  is only used when it beats the plain argmax by more than noise.
- A candidate replaces the best only when a paired bootstrap over the dev examples
  says it is better with KEEP_CONFIDENCE probability.

Each time a run finds a better prompt it saves it as the run's own model version, so the
best prompt so far is never lost to a stopped, failed or interrupted run. A run whose LLM
keeps failing stops proposing, but still scores what it found on the holdout.

For a task with a language (search intent: EVAL_LANGUAGE, English by default) only golden
rows in that language take part. The LLM only ever sees dev examples. Held-out examples (EVAL_HOLDOUT_RATIO of all
golden data plus whatever the dev sample did not use) are scored once at the end
for the baseline and the best config. Every run writes its state, per-iteration
checkpoints and prompts under autoresearch/runs/<run_id>/, and appends one line
per experiment to autoresearch/results.tsv.
"""

import json
import random
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app import engines, llm, serp, versions
from app.config import EVAL_HOLDOUT_RATIO, RESULTS_TSV, RUNS_DIR
from app.database import ExperimentRun, SessionLocal
from app.laya_inference import validate_config
from app.scoring import KEEP_CONFIDENCE, METRICS, evaluate, gold_array, pred_array, prob_better, summarise
from app.tasks.registry import DEFAULT_TASK_ID, TASKS, Task

MAX_HOLDOUT_SIZE = 500
MAX_CONSECUTIVE_CRASHES = 3
ERRORS_SHOWN = 40
GOLD_SHOWN_PER_LABEL = 8
HISTORY_SHOWN = 15
# Inputs are shown to the LLM cut to this many characters, so long emails and comments do not swamp the prompt.
EXAMPLE_CHARS = 240


LAYA_NOTES = """- It was trained on states that are JSON objects with named fields, and instructions that name the field in backticks. Example from its presets: state {{"message": "..."}} with instructions "What does the customer want in `message`?". On a dataset of search queries a bare-string state made the model put about 95% of them into a single label, while the same wording with a JSON state and the field named in the instructions produced a real spread of predictions.
- Its presets describe options in short plain language, for example "money returned or a duplicate charge reversed" or "a bug, outage or integration problem", not as keyword lists.
- Each criteria description is cut off after about 48 tokens, and the instructions and all {n} descriptions together share a budget of about 190 tokens. Stay under {criterion_words} words each and put the most discriminating words first.
- Keep instructions under {instruction_words} words.
- Only about the first 300 tokens of a long {item} are read."""

LAYA_INTRO = (
    'The classifier is Laya, a fast non-autoregressive "System 1" model. It is not a chat LLM: in one forward pass it '
    "reads a state, a question's instructions and a short description of each option, and outputs a probability per "
    "option. It cannot reason step by step or follow long rule lists. It responds to the wording and the format of "
    "what it is given."
)
OTHER_INTRO = (
    "The classifier is {name}, a small zero-shot classification model. It is not a chat LLM: it scores each label's "
    "description against the text and outputs a probability per label. It cannot reason step by step or follow long "
    "rule lists. It responds to the wording of what it is given."
)


def build_system_prompt(task: Task, metric: str, calibrate: bool, use_serp: bool, engine_id: str = engines.LAYA) -> str:
    labels = task.labels
    engine = engines.get_engine(engine_id)
    if engine_id == engines.LAYA:
        intro = LAYA_INTRO
        notes = LAYA_NOTES.format(n=len(labels), criterion_words=task.max_criterion_words,
                                  instruction_words=task.max_instruction_words, item=task.item)
    else:
        intro = OTHER_INTRO.format(name=engine["name"])
        notes = (
            engines.PROMPT_NOTES[engine["family"]]
            + f"\n- The state template only wraps the text: field names are not shown to this model, so leave it as it is."
            + f"\n- Keep each description under {task.max_criterion_words} words and the instructions under "
              f"{task.max_instruction_words} words. Inputs are cut to 512 tokens."
        )
    field, placeholder = task.input_field, task.placeholder
    state_lines = [
        f'- "state_template": how each {task.item} is presented to the model. Either a plain string, or a JSON object '
        f"of field name -> string, which is passed to the model as a JSON state. It must contain {placeholder} exactly once.",
    ]
    if use_serp:
        state_lines.append(
            "  You may also use {serp_sites} (domains of the top-ranking results), {serp_titles} (their titles) and "
            f'{{serp_snippets}} (their snippets), and set "serp_results" (1 to {serp.MAX_RESULTS}) for how many top '
            "results they draw on. The gold labels were produced from search results, so this context can carry signal, "
            "but long context can also drown out the query. Whether it helps is an empirical question for the loop."
        )
    question_lines = [
        '- "instructions": the question the model answers.',
        '- "criteria": one description per label. The label names and their number are fixed.',
    ]
    if task.question_type == "noul":
        no, yes = labels
        question_lines.append(
            f'  This is a yes/no question for Laya: the instructions must be a question whose answer is yes for "{yes}" '
            f'and no for "{no}", and the two descriptions say what the yes side and the no side look like.'
        )
    calibration = (
        "When a config skews towards one label, a per-label bias is fitted automatically on the dev set to correct "
        "it, so how often each label gets predicted is largely taken care of. Do not spend rounds trying to make a "
        "label more or less frequent. What moves the score is how well the wording separates the labels from each other."
        if calibrate else
        "No calibration is applied: the label with the highest probability wins. Wording changes can swing which label "
        "the model favours, so watch the predicted-label counts for collapse onto one label."
    )
    serp_field = '\n  "serp_results": 4,' if use_serp else ""
    return f"""You are running an autoresearch loop that improves a small classifier by rewriting the text it is given.

{intro}

The task is {task.loop_task} into exactly these labels: {", ".join(labels)}. {task.loop_conventions}

You may change these things, and nothing else:
{chr(10).join(state_lines)}
{chr(10).join(question_lines)}

What is known about the model:
{notes}

How the loop scores a config: it is run on a dev set and scored by {METRICS[metric]}. {calibration}

A config replaces the current best only if it scores higher AND a paired bootstrap over the dev {task.items} says the gain is real with probability {KEEP_CONFIDENCE}. A one-or-two-{task.item} gain is noise and is discarded. A separate held-out set you never see is scored at the end, so describing kinds of {task.items} generalises and pasting specific dev {task.items} does not.

Make one clear, testable change per round and say what you expect it to fix. Use the experiment history: it shows, for every past round, how the predicted-label counts and per-label recall moved. Do not repeat a discarded idea, and try a different direction when several rounds in a row were discarded.

Reply with a single JSON object and nothing else:
{{
  "hypothesis": "one or two sentences: what you are changing and why it should help",
  "state_template": {{"{field}": "{placeholder}"}},{serp_field}
  "instructions": "...",
  "criteria": {{{", ".join(f'"{label}": "..."' for label in labels)}}}
}}"""


# ---------------------------------------------------------------- data

def split_examples(
    examples: List[Dict[str, str]], sample_size: int, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Seeded shuffle into (dev, holdout).

    The holdout share of all data is reserved first, dev is sampled from the rest, and
    whatever dev did not use joins the holdout (capped), since it costs nothing to keep unseen.
    """
    shuffled = sorted(examples, key=lambda e: e["text"])
    random.Random(seed).shuffle(shuffled)
    reserved_size = max(1, round(len(shuffled) * EVAL_HOLDOUT_RATIO))
    reserved, rest = shuffled[:reserved_size], shuffled[reserved_size:]
    dev, unused = rest[:sample_size], rest[sample_size:]
    return dev, (reserved + unused)[:MAX_HOLDOUT_SIZE]


# ---------------------------------------------------------------- prompt

def _proposal_view(config: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a config the LLM writes (no checkpoint name, no fitted bias)."""
    return {k: v for k, v in config.items() if k not in ("model", "label_bias")}


def _sample_errors(cases: List[Dict[str, Any]], rng: random.Random, limit: int) -> List[Dict[str, Any]]:
    """Pick errors round-robin across (actual, predicted) pairs so every confusion is represented."""
    by_pair = defaultdict(list)
    for case in cases:
        if not case["correct"]:
            by_pair[(case["actual"], case["predicted"])].append(case)
    for group in by_pair.values():
        rng.shuffle(group)
    # Largest confusions first.
    groups = sorted(by_pair.values(), key=len, reverse=True)
    picked = []
    while len(picked) < limit and any(groups):
        for group in groups:
            if group and len(picked) < limit:
                picked.append(group.pop())
    return picked


def _counts_line(counts: Dict[str, int], labels: List[str]) -> str:
    return ", ".join(f"{label} {counts[label]}" for label in labels)


def _shown(text: str) -> str:
    """An input as the LLM sees it: on one line, and cut when long."""
    text = " ".join(text.split())
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS].rstrip() + "…"


def build_user_prompt(
    task: Task,
    best_config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    history: List[Dict[str, Any]],
    iteration: int,
    total_iterations: int,
    seed: int,
    metric: str,
    use_serp: bool,
) -> str:
    labels = task.labels
    rng = random.Random(seed * 1000 + iteration)
    cases = best_metrics["cases"]

    def with_sites(query: str) -> str:
        if not use_serp:
            return ""
        sites = ", ".join(r["site"] for r in serp.lookup(query)[:3])
        return f"  top sites: {sites}" if sites else ""

    lines = [
        f"Round {iteration} of {total_iterations}. Objective: {METRICS[metric]}.",
        "",
        "## Current best config",
        json.dumps(_proposal_view(best_config), indent=2, ensure_ascii=False),
        "",
        f"## Its dev-set results ({best_metrics['total']} {task.items})",
        f"score: {best_metrics['score']:.4f}   accuracy: {best_metrics['accuracy']:.4f}   macro-F1: {best_metrics['macro_f1']:.4f}",
        f"gold label counts:      {_counts_line(best_metrics['gold_counts'], labels)}",
        f"predicted label counts: {_counts_line(best_metrics['predicted_counts'], labels)}",
        "",
        "Per class (precision / recall / f1 / support):",
    ]
    for label in labels:
        m = best_metrics["per_class"][label]
        lines.append(f"- {label}: {m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f} / {m['support']}")

    lines += ["", "Confusion matrix (rows = gold label, columns = predicted):",
              "gold \\ predicted | " + " | ".join(labels)]
    for actual in labels:
        row = best_metrics["confusion"][actual]
        lines.append(f"{actual} | " + " | ".join(str(row[p]) for p in labels))

    lines += ["", f"## What the gold labels look like (random dev {task.items} per label)"]
    for label in labels:
        gold = [c for c in cases if c["actual"] == label]
        rng.shuffle(gold)
        lines.append(f"{label}:")
        for case in gold[:GOLD_SHOWN_PER_LABEL]:
            lines.append(f'- "{_shown(case["text"])}"{with_sites(case["text"])}')

    errors = _sample_errors(cases, rng, ERRORS_SHOWN)
    total_errors = best_metrics["total"] - best_metrics["correct"]
    lines += ["", f"## Misclassified dev {task.items} ({len(errors)} of {total_errors} shown)"]
    for case in errors:
        lines.append(
            f'- "{_shown(case["text"])}"  gold={case["actual"]}  predicted={case["predicted"]} '
            f'({case["confidence"]:.2f}){with_sites(case["text"])}'
        )

    lines += ["", "## Experiment history (oldest first)"]
    if not history:
        lines.append("No experiments yet. This is the first round.")
    for item in history[-HISTORY_SHOWN:]:
        if item["status"] == "crash":
            lines.append(f"- round {item['iteration']}: CRASH ({item['error']})")
            continue
        recall = ", ".join(f"{label} {item['per_class'][label]['recall']:.2f}" for label in labels)
        lines += [
            f"- round {item['iteration']}: {item['status'].upper()}  score {item['score']:.4f} "
            f"({item['delta_vs_best']:+.4f} vs best at the time, P(better)={item['p_better']:.2f})  "
            f"accuracy {item['accuracy']:.4f}  macro-F1 {item['macro_f1']:.4f}",
            f"  hypothesis: {item['hypothesis']}",
            f"  predicted counts: {_counts_line(item['predicted_counts'], labels)}   recall: {recall}",
        ]
        if item["status"] == "discard":
            lines.append(f"  discarded config: {json.dumps(_proposal_view(item['config']), ensure_ascii=False)}")

    lines += ["", "Propose the next config as a single JSON object."]
    return "\n".join(lines)


def parse_proposal(
    raw_reply: str, task: Task, use_serp: bool, laya_model: Optional[str], engine_id: str = engines.LAYA
) -> Tuple[Dict[str, Any], str, List[str]]:
    """Turn the LLM's reply into (config, hypothesis, warnings)."""
    proposal = llm.extract_json(raw_reply)
    # The model and checkpoint are run settings and the bias is fitted, so none of them is taken from the LLM.
    proposal = {**proposal, "model": laya_model, "engine": engine_id}
    config, warnings = validate_config(proposal, task, allow_serp=use_serp)
    hypothesis = str(proposal.get("hypothesis", "")).strip() or "(no hypothesis given)"
    return config, hypothesis, warnings


# ---------------------------------------------------------------- run state

_runs: Dict[str, Dict[str, Any]] = {}
_stop_events: Dict[str, threading.Event] = {}
_state_lock = threading.Lock()
_active_run_id: Optional[str] = None


def _now() -> str:
    return datetime.utcnow().isoformat()


def _write_json(path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _persist(run: Dict[str, Any]) -> None:
    _write_json(RUNS_DIR / run["run_id"] / "run.json", run)


def _log(run: Dict[str, Any], message: str) -> None:
    line = f"{_now()}  {message}"
    with _state_lock:
        run["log"].append(line)
    with open(RUNS_DIR / run["run_id"] / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_results_tsv(run: Dict[str, Any], iteration: int, metrics: Optional[Dict[str, Any]], status: str, description: str) -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "timestamp\trun_id\titeration\tdev_accuracy\tmacro_f1\tstatus\tdescription\tscore\ttask\n", encoding="utf-8"
        )
    accuracy, macro_f1, score = (f"{metrics[k]:.4f}" if metrics else "0.0000" for k in ("accuracy", "macro_f1", "score"))
    description = " ".join(description.split())
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        # score and task are the last columns so lines written before they existed still line up.
        # A line without a task belongs to the search-intent example.
        f.write(f"{_now()}\t{run['run_id']}\t{iteration}\t{accuracy}\t{macro_f1}\t{status}\t{description}\t{score}\t{run['task']}\n")


def _with_task(run: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in the task fields of a run saved before there were several examples."""
    run.setdefault("task", DEFAULT_TASK_ID)
    run.setdefault("engine", engines.LAYA)
    run.setdefault("engine_name", engines.ENGINES[engines.LAYA]["name"])
    # Failed runs used to leave `improved` unset even when a round had been kept.
    if run.get("status") != "running" and (run.get("best_iteration") or 0) > 0:
        run["improved"] = True
    if "labels" not in run:
        task = TASKS.get(run["task"])
        run["labels"] = list(task.labels) if task else list(run["best_config"]["criteria"])
    return run


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a run's state: live from memory, otherwise from its run.json."""
    with _state_lock:
        if run_id in _runs:
            return json.loads(json.dumps(_runs[run_id]))
    path = RUNS_DIR / run_id / "run.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    # A run still marked running on disk belonged to a server process that has gone away.
    if run["status"] == "running":
        run["status"] = "interrupted"
    return _with_task(run)


def list_runs(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Summaries of the runs on disk, newest first. With task_id, only that example's runs."""
    summaries = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        run = get_run(run_dir.name) if run_dir.is_dir() else None
        if not run or (task_id and run["task"] != task_id):
            continue
        baseline = run.get("baseline") or {}
        summaries.append({
            "run_id": run["run_id"],
            "task": run["task"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run.get("finished_at"),
            "llm": run["llm"],
            "loops": run["loops"],
            "completed_iterations": len(run["iterations"]),
            "dev_size": run["dev_size"],
            "baseline_accuracy": baseline.get("accuracy"),
            "baseline_macro_f1": baseline.get("macro_f1"),
            "best_accuracy": run.get("best_accuracy"),
            "best_macro_f1": run.get("best_macro_f1"),
            "best_iteration": run.get("best_iteration"),
            "improved": run.get("improved", False),
        })
    return summaries


def active_run_id() -> Optional[str]:
    with _state_lock:
        return _active_run_id


def active_run() -> Optional[Dict[str, str]]:
    """The run in progress, if any, and the example it belongs to."""
    with _state_lock:
        if _active_run_id is None:
            return None
        return {"run_id": _active_run_id, "task": _runs[_active_run_id]["task"]}


def request_stop(run_id: str) -> bool:
    event = _stop_events.get(run_id)
    if not event:
        return False
    event.set()
    return True


def start_run(
    task: Task,
    examples: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    loops: int,
    sample_size: int,
    seed: int,
    start_config: Dict[str, Any],
    start_version: str,
    metric: str,
    calibrate: bool,
    use_serp: bool,
    laya_model: str,
    excluded_other_language: int = 0,
) -> Dict[str, Any]:
    # The model being optimized comes with the start config (see laya_inference.on_engine).
    engine_id = start_config.get("engine") or engines.LAYA
    """Create a run and start it on a background thread. Only one run at a time, across all examples."""
    global _active_run_id
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}")
    if use_serp and not task.supports_serp:
        raise ValueError(f"The example '{task.name}' has no search-result context")
    if use_serp and not serp.available():
        raise ValueError(f"SERP context needs {serp.SERP_FILE}, which was not found")
    dev, holdout = split_examples(examples, sample_size, seed)
    if len(dev) < 20:
        raise ValueError("Not enough labelled data for a dev set")

    # The run's checkpoint replaces the start version's, and any saved bias is refitted or dropped.
    baseline_config, _ = validate_config({**start_config, "model": laya_model}, task)

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run = {
        "run_id": run_id,
        "task": task.id,
        # Kept with the run so it still renders if the example's definition changes later.
        "labels": list(task.labels),
        "items": task.items,
        "status": "running",
        "phase": "baseline",
        "started_at": _now(),
        "finished_at": None,
        "llm": {"provider": llm_config["provider"], "model": llm_config["model"]},
        "loops": loops,
        "seed": seed,
        "start_version": start_version,
        "metric": metric,
        "calibrate": calibrate,
        "use_serp": use_serp,
        "laya_model": laya_model if engine_id == engines.LAYA else None,
        "engine": engine_id,
        "engine_name": engines.get_engine(engine_id)["name"],
        "language": task.language,
        "excluded_other_language": excluded_other_language,
        "keep_confidence": KEEP_CONFIDENCE,
        "dev_size": len(dev),
        "holdout_size": len(holdout),
        "baseline_config": baseline_config,
        "baseline": None,
        "best_config": baseline_config,
        "best_score": None,
        "best_accuracy": None,
        "best_macro_f1": None,
        "best_iteration": 0,
        "improved": False,
        # The version this run's best prompt is saved under, once a round has been kept.
        "auto_version": None,
        "iterations": [],
        "holdout": None,
        "error": None,
        "log": [],
    }
    with _state_lock:
        if _active_run_id is not None:
            raise RuntimeError(f"Run {_active_run_id} is still in progress")
        _active_run_id = run_id
        _runs[run_id] = run
        _stop_events[run_id] = threading.Event()

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "dev_set.json", dev)
    _write_json(run_dir / "holdout_set.json", holdout)
    _persist(run)

    thread = threading.Thread(
        target=_run_loop, args=(run, task, dev, holdout, llm_config), name=f"karpathy-{run_id}", daemon=True
    )
    thread.start()
    return get_run(run_id)


def _run_loop(
    run: Dict[str, Any], task: Task, dev: List[Dict[str, str]], holdout: List[Dict[str, str]], llm_config: Dict[str, Any]
) -> None:
    global _active_run_id
    run_id = run["run_id"]
    run_dir = RUNS_DIR / run_id
    stop = _stop_events[run_id]
    metric, calibrate, use_serp = run["metric"], run["calibrate"], run["use_serp"]
    labels, k = task.labels, len(task.labels)
    engine_id = run.get("engine") or engines.LAYA
    system_prompt = build_system_prompt(task, metric, calibrate, use_serp, engine_id)

    def update(**fields):
        with _state_lock:
            run.update(fields)
        _persist(run)

    def progress(label: str):
        """A callback that puts "<label>: <done> / <total>" in the run's phase as Laya works through a set."""
        def report(done: int, total: int) -> None:
            with _state_lock:
                run["phase"] = f"{label}: {done} / {total}"
            if done == total:
                _persist(run)
        return report

    def autosave() -> None:
        """Save the run's best prompt as its model version. A failure here must not cost the run."""
        db = SessionLocal()
        try:
            name = versions.autosave_run_best(db, task, run)
            if run["auto_version"] != name:
                update(auto_version=name)
                _log(run, f"Best prompt saved as model version '{name}'")
        except Exception as e:  # noqa: BLE001
            _log(run, f"Could not save the best prompt as a model version: {e}")
        finally:
            db.close()

    try:
        if task.language:
            _log(run, f"Golden {task.items} in language '{task.language}' only: "
                      f"{run['excluded_other_language']} in other languages excluded")
        _log(run, f"Run started for '{task.name}': {len(dev)} dev / {len(holdout)} holdout {task.items}, {run['loops']} rounds, "
                  f"LLM {llm_config['provider']}:{llm_config['model'] or 'default'}, model {run['engine_name']}"
                  f"{' (' + run['laya_model'] + ')' if run['laya_model'] else ''}, "
                  f"objective {metric}, calibration {'on' if calibrate else 'off'}, SERP {'on' if use_serp else 'off'}")

        baseline_config, best_metrics = evaluate(run["baseline_config"], dev, task, metric, fit_bias=calibrate,
                                                 on_progress=progress("baseline"))
        best_config = baseline_config
        y = gold_array(best_metrics, labels)
        _write_json(run_dir / "iter_000_baseline.json", {
            "iteration": 0, "status": "baseline", "config": best_config, "metrics": best_metrics,
        })
        _write_json(run_dir / "best_config.json", {"iteration": 0, "score": best_metrics["score"], "config": best_config})
        _append_results_tsv(run, 0, best_metrics, "baseline", f"baseline ({run['start_version']})")
        update(baseline=summarise(best_metrics), baseline_config=baseline_config, best_config=best_config,
               best_score=best_metrics["score"], best_accuracy=best_metrics["accuracy"],
               best_macro_f1=best_metrics["macro_f1"], phase="looping")
        _log(run, f"Baseline: score {best_metrics['score']:.4f}, accuracy {best_metrics['accuracy']:.4f}, "
                  f"macro-F1 {best_metrics['macro_f1']:.4f}, predicted {_counts_line(best_metrics['predicted_counts'], labels)}")

        history: List[Dict[str, Any]] = []
        consecutive_crashes = 0
        failure: Optional[str] = None

        for i in range(1, run["loops"] + 1):
            if stop.is_set():
                _log(run, "Stop requested; ending the loop early")
                break

            update(phase=f"round {i}: asking the LLM")
            user_prompt = build_user_prompt(task, best_config, best_metrics, history, i, run["loops"], run["seed"], metric, use_serp)
            with open(run_dir / f"iter_{i:03d}_prompt.txt", "w", encoding="utf-8") as f:
                f.write(system_prompt + "\n\n-----\n\n" + user_prompt)

            entry: Dict[str, Any] = {"iteration": i, "timestamp": _now()}
            raw_reply = None
            try:
                raw_reply = llm.complete(llm_config, system_prompt, user_prompt)
                candidate, hypothesis, warnings = parse_proposal(raw_reply, task, use_serp, run["laya_model"], engine_id)
            except (llm.LLMError, ValueError) as e:
                consecutive_crashes += 1
                entry.update(status="crash", error=str(e), hypothesis=None, config=None, score=None,
                             accuracy=None, macro_f1=None, delta_vs_best=None, p_better=None,
                             best_score=best_metrics["score"], warnings=[])
                _write_json(run_dir / f"iter_{i:03d}.json", {**entry, "llm_reply": raw_reply})
                _append_results_tsv(run, i, None, "crash", str(e))
                history.append(entry)
                with _state_lock:
                    run["iterations"].append(entry)
                _persist(run)
                _log(run, f"Round {i}: CRASH - {e}")
                if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                    # Stop proposing, but keep what the run found: the holdout below needs only Laya.
                    failure = f"{MAX_CONSECUTIVE_CRASHES} rounds in a row failed. Last error: {e}"
                    _log(run, f"Stopping early: {failure}")
                    break
                continue
            consecutive_crashes = 0

            update(phase=f"round {i}: evaluating on {len(dev)} dev {task.items}")
            candidate, metrics = evaluate(candidate, dev, task, metric, fit_bias=calibrate,
                                          on_progress=progress(f"round {i}: evaluating"))
            delta = round(metrics["score"] - best_metrics["score"], 4)
            p_better = prob_better(y, pred_array(metrics, labels), pred_array(best_metrics, labels), metric, k, run["seed"] + i)
            kept = delta > 0 and p_better >= KEEP_CONFIDENCE
            status = "keep" if kept else "discard"

            if kept:
                best_config, best_metrics = candidate, metrics
                _write_json(run_dir / "best_config.json", {"iteration": i, "score": metrics["score"], "config": candidate})

            entry.update(status=status, error=None, hypothesis=hypothesis, config=candidate,
                         score=metrics["score"], accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                         raw_accuracy=metrics["raw_accuracy"], delta_vs_best=delta, p_better=round(p_better, 3),
                         best_score=best_metrics["score"], predicted_counts=metrics["predicted_counts"],
                         per_class=metrics["per_class"], warnings=warnings)
            _write_json(run_dir / f"iter_{i:03d}.json", {**entry, "metrics": metrics, "llm_reply": raw_reply})
            _append_results_tsv(run, i, metrics, status, hypothesis)
            history.append(entry)
            with _state_lock:
                run["iterations"].append(entry)
                if kept:
                    run.update(best_config=candidate, best_score=metrics["score"], best_accuracy=metrics["accuracy"],
                               best_macro_f1=metrics["macro_f1"], best_iteration=i)
            _persist(run)
            _log(run, f"Round {i}: {status.upper()} score {metrics['score']:.4f} ({delta:+.4f}, P(better)={p_better:.2f}), "
                      f"accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, "
                      f"predicted {_counts_line(metrics['predicted_counts'], labels)} - {hypothesis}")
            if kept:
                autosave()

        update(phase=f"scoring {len(holdout)} holdout {task.items}")
        improved = run["best_iteration"] > 0
        # The bias fitted on dev is applied as is: nothing is fitted on the holdout.
        _, holdout_baseline = evaluate(baseline_config, holdout, task, metric, fit_bias=False,
                                       on_progress=progress("holdout, starting prompt"))
        holdout_best = evaluate(best_config, holdout, task, metric, fit_bias=False,
                                on_progress=progress("holdout, best prompt"))[1] if improved else holdout_baseline
        holdout_p = prob_better(gold_array(holdout_best, labels), pred_array(holdout_best, labels),
                                pred_array(holdout_baseline, labels), metric, k, run["seed"]) if improved else None
        _write_json(run_dir / "holdout_results.json", {"baseline": holdout_baseline, "best": holdout_best, "p_better": holdout_p})
        _log(run, f"Holdout: baseline score {holdout_baseline['score']:.4f} (accuracy {holdout_baseline['accuracy']:.4f}, "
                  f"macro-F1 {holdout_baseline['macro_f1']:.4f}), best score {holdout_best['score']:.4f} "
                  f"(accuracy {holdout_best['accuracy']:.4f}, macro-F1 {holdout_best['macro_f1']:.4f})"
                  + (f", P(better)={holdout_p:.2f}" if holdout_p is not None else ""))

        update(
            holdout={"baseline": summarise(holdout_baseline), "best": summarise(holdout_best), "p_better": holdout_p},
            improved=improved,
            status="failed" if failure else "stopped" if stop.is_set() else "completed",
            error=failure,
            phase="done",
            finished_at=_now(),
        )
        if improved:
            autosave()  # adds the holdout accuracy to the saved version
        _log(run, f"Run finished: dev score {run['baseline']['score']:.4f} -> {run['best_score']:.4f} "
                  f"(best round {run['best_iteration']})")
        _record_experiment(run)
    except Exception as e:  # noqa: BLE001 - a failed run must be recorded, not lost with the thread
        _log(run, f"Run failed: {e}")
        update(status="failed", phase="done", error=str(e), finished_at=_now(), improved=run["best_iteration"] > 0)
    finally:
        with _state_lock:
            _active_run_id = None
        _stop_events.pop(run_id, None)


def _record_experiment(run: Dict[str, Any]) -> None:
    """Mirror the finished run into the experiment_runs table."""
    db = SessionLocal()
    try:
        db.add(ExperimentRun(
            id=run["run_id"],
            task_id=run["task"],
            baseline_model="laya",
            baseline_version=run["start_version"],
            baseline_accuracy=run["baseline"]["accuracy"],
            final_accuracy=run["best_accuracy"],
            best_iteration=run["best_iteration"],
            best_model_name=None,
            improved=run["improved"],
            iterations_data=run["iterations"],
        ))
        db.commit()
    finally:
        db.close()
