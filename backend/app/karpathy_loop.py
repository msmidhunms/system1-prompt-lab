"""Karpathy-style autoresearch loop over Laya's prompt.

Same shape as https://github.com/karpathy/autoresearch, but the thing being
edited is the Laya prompt config (state template, instructions, label
descriptions) instead of a training script:

    propose a change (LLM) -> evaluate on the dev set -> keep if accuracy
    improved, otherwise discard -> repeat.

The LLM only ever sees dev examples. A holdout split is scored once at the end
for the baseline and the best config. Every run writes its state, per-iteration
checkpoints and prompts under autoresearch/runs/<run_id>/, and appends one line
per experiment to autoresearch/results.tsv.
"""

import json
import random
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import precision_recall_fscore_support

from app import llm
from app.config import EVAL_HOLDOUT_RATIO, RESULTS_TSV, RUNS_DIR
from app.database import ExperimentRun, SessionLocal
from app.laya_inference import (
    MAX_CRITERION_WORDS,
    MAX_INSTRUCTION_WORDS,
    predict_batch,
    validate_config,
)
from app.tasks.intent.labels import SearchIntent

LABELS = SearchIntent.all_labels()
MAX_CONSECUTIVE_CRASHES = 3
ERRORS_SHOWN = 40
CORRECT_SHOWN = 12
HISTORY_SHOWN = 15

SYSTEM_PROMPT = f"""You are running an autoresearch loop that improves a small classifier by rewriting its prompt.

The classifier is Laya, a fast non-autoregressive "System 1" model. It is not a chat LLM: in one forward pass it reads a state, a question's instructions and a short description of each option, and outputs a probability per option. It cannot reason step by step, follow long rule lists, or use examples the way you can. It responds to the wording of the text it is given.

The task is search-intent classification of search queries into exactly these labels: {", ".join(LABELS)}. The gold labels come from an SEO data provider, so their conventions may differ from the textbook definitions. Infer the real conventions from the data you are shown.

You may change three things, and nothing else:
- "state_template": how the raw query is presented to the model. It must contain {{query}} exactly once.
- "instructions": the question the model answers.
- "criteria": one description per label. The label names and their number are fixed.

Hard limits of the model:
- Each criteria description is cut off after about 48 tokens. Stay under {MAX_CRITERION_WORDS} words each and put the most discriminating words first.
- Keep instructions under {MAX_INSTRUCTION_WORDS} words.

How the loop works: each round you propose one new config. It is scored on a dev set. If its accuracy beats the current best it becomes the new best, otherwise it is discarded and you continue from the current best. A separate holdout set you never see is scored at the end, so describing kinds of queries generalises and pasting specific dev queries does not.

Make one clear, testable change per round and say what you expect it to fix. Use the experiment history: do not repeat a discarded idea, and try a different direction when several rounds in a row were discarded.

Reply with a single JSON object and nothing else:
{{
  "hypothesis": "one or two sentences: what you are changing and why it should help",
  "state_template": "...",
  "instructions": "...",
  "criteria": {{{", ".join(f'"{label}": "..."' for label in LABELS)}}}
}}"""


# ---------------------------------------------------------------- evaluation

def split_examples(
    examples: List[Dict[str, str]], sample_size: int, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Seeded shuffle, take sample_size, and split off the holdout set."""
    shuffled = sorted(examples, key=lambda e: e["query"])
    random.Random(seed).shuffle(shuffled)
    sample = shuffled[:sample_size]
    holdout_size = max(1, round(len(sample) * EVAL_HOLDOUT_RATIO))
    return sample[holdout_size:], sample[:holdout_size]


def evaluate(config: Dict[str, Any], examples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run Laya with a prompt config over labelled examples and score it."""
    predictions = predict_batch([e["query"] for e in examples], config)
    true = [e["intent"] for e in examples]
    pred = [p["intent"] for p in predictions]

    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f), 3),
            "support": int(s),
        }
        for label, p, r, f, s in zip(LABELS, precision, recall, f1, support)
    }
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    cases = []
    for example, prediction in zip(examples, predictions):
        if example["intent"] in confusion:
            confusion[example["intent"]][prediction["intent"]] += 1
        cases.append({
            "query": example["query"],
            "actual": example["intent"],
            "predicted": prediction["intent"],
            "confidence": prediction["confidence"],
            "correct": example["intent"] == prediction["intent"],
        })

    correct = sum(c["correct"] for c in cases)
    return {
        "accuracy": round(correct / len(cases), 4),
        "macro_f1": round(float(sum(f1) / len(LABELS)), 4),
        "correct": correct,
        "total": len(cases),
        "per_class": per_class,
        "confusion": confusion,  # confusion[actual][predicted]
        "cases": cases,
    }


def summarise(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Metrics without the per-case list (for run state and API responses)."""
    return {k: v for k, v in metrics.items() if k != "cases"}


# ---------------------------------------------------------------- prompt

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


def build_user_prompt(
    best_config: Dict[str, Any],
    best_metrics: Dict[str, Any],
    history: List[Dict[str, Any]],
    iteration: int,
    total_iterations: int,
    seed: int,
) -> str:
    rng = random.Random(seed * 1000 + iteration)
    errors = _sample_errors(best_metrics["cases"], rng, ERRORS_SHOWN)
    correct = [c for c in best_metrics["cases"] if c["correct"]]
    rng.shuffle(correct)

    lines = [
        f"Round {iteration} of {total_iterations}.",
        "",
        "## Current best config",
        json.dumps(best_config, indent=2, ensure_ascii=False),
        "",
        f"## Its dev-set results ({best_metrics['total']} queries)",
        f"accuracy: {best_metrics['accuracy']:.4f}   macro-F1: {best_metrics['macro_f1']:.4f}",
        "",
        "Per class (precision / recall / f1 / support):",
    ]
    for label in LABELS:
        m = best_metrics["per_class"][label]
        lines.append(f"- {label}: {m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f} / {m['support']}")

    lines += ["", "Confusion matrix (rows = gold label, columns = predicted):",
              "gold \\ predicted | " + " | ".join(LABELS)]
    for actual in LABELS:
        row = best_metrics["confusion"][actual]
        lines.append(f"{actual} | " + " | ".join(str(row[p]) for p in LABELS))

    total_errors = best_metrics["total"] - best_metrics["correct"]
    lines += ["", f"## Misclassified dev queries ({len(errors)} of {total_errors} shown)"]
    for case in errors:
        lines.append(f'- "{case["query"]}"  gold={case["actual"]}  predicted={case["predicted"]} ({case["confidence"]:.2f})')

    lines += ["", "## Some correctly classified dev queries"]
    for case in correct[:CORRECT_SHOWN]:
        lines.append(f'- "{case["query"]}"  {case["actual"]}')

    lines += ["", "## Experiment history (oldest first)"]
    if not history:
        lines.append("No experiments yet. This is the first round.")
    for item in history[-HISTORY_SHOWN:]:
        if item["status"] == "crash":
            lines.append(f"- round {item['iteration']}: CRASH ({item['error']})")
            continue
        lines.append(
            f"- round {item['iteration']}: {item['status'].upper()}  "
            f"accuracy {item['accuracy']:.4f} ({item['delta_vs_best']:+.4f} vs best at the time)  "
            f"hypothesis: {item['hypothesis']}"
        )
        if item["status"] == "discard":
            lines.append(f"  discarded config: {json.dumps(item['config'], ensure_ascii=False)}")

    lines += ["", "Propose the next config as a single JSON object."]
    return "\n".join(lines)


def propose(llm_config: Dict[str, Any], user_prompt: str) -> Tuple[Dict[str, Any], str, List[str], str]:
    """Ask the LLM for the next config. Returns (config, hypothesis, warnings, raw reply)."""
    raw = llm.complete(llm_config, SYSTEM_PROMPT, user_prompt)
    proposal = llm.extract_json(raw)
    config, warnings = validate_config(proposal)
    hypothesis = str(proposal.get("hypothesis", "")).strip() or "(no hypothesis given)"
    return config, hypothesis, warnings, raw


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


def _append_results_tsv(run_id: str, iteration: int, metrics: Optional[Dict[str, Any]], status: str, description: str) -> None:
    new_file = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp\trun_id\titeration\tdev_accuracy\tmacro_f1\tstatus\tdescription\n")
        accuracy = f"{metrics['accuracy']:.4f}" if metrics else "0.0000"
        macro_f1 = f"{metrics['macro_f1']:.4f}" if metrics else "0.0000"
        description = " ".join(description.split())
        f.write(f"{_now()}\t{run_id}\t{iteration}\t{accuracy}\t{macro_f1}\t{status}\t{description}\n")


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
    return run


def list_runs() -> List[Dict[str, Any]]:
    """Summaries of all runs on disk, newest first."""
    summaries = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        run = get_run(run_dir.name) if run_dir.is_dir() else None
        if not run:
            continue
        summaries.append({
            "run_id": run["run_id"],
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run.get("finished_at"),
            "llm": run["llm"],
            "loops": run["loops"],
            "completed_iterations": len(run["iterations"]),
            "dev_size": run["dev_size"],
            "baseline_accuracy": run["baseline"]["accuracy"] if run.get("baseline") else None,
            "best_accuracy": run.get("best_accuracy"),
            "best_iteration": run.get("best_iteration"),
            "improved": run.get("improved", False),
        })
    return summaries


def active_run_id() -> Optional[str]:
    with _state_lock:
        return _active_run_id


def request_stop(run_id: str) -> bool:
    event = _stop_events.get(run_id)
    if not event:
        return False
    event.set()
    return True


def start_run(
    examples: List[Dict[str, str]],
    llm_config: Dict[str, Any],
    loops: int,
    sample_size: int,
    seed: int,
    start_config: Dict[str, Any],
    start_version: str,
) -> Dict[str, Any]:
    """Create a run and start it on a background thread. Only one run at a time."""
    global _active_run_id
    dev, holdout = split_examples(examples, sample_size, seed)
    if len(dev) < 5:
        raise ValueError("Not enough labelled data for a dev set")

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run = {
        "run_id": run_id,
        "status": "running",
        "phase": "baseline",
        "started_at": _now(),
        "finished_at": None,
        "llm": {"provider": llm_config["provider"], "model": llm_config["model"]},
        "loops": loops,
        "seed": seed,
        "start_version": start_version,
        "dev_size": len(dev),
        "holdout_size": len(holdout),
        "baseline_config": start_config,
        "baseline": None,
        "best_config": start_config,
        "best_accuracy": None,
        "best_iteration": 0,
        "improved": False,
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
        target=_run_loop, args=(run, dev, holdout, llm_config), name=f"karpathy-{run_id}", daemon=True
    )
    thread.start()
    return get_run(run_id)


def _run_loop(run: Dict[str, Any], dev: List[Dict[str, str]], holdout: List[Dict[str, str]], llm_config: Dict[str, Any]) -> None:
    global _active_run_id
    run_id = run["run_id"]
    run_dir = RUNS_DIR / run_id
    stop = _stop_events[run_id]

    def update(**fields):
        with _state_lock:
            run.update(fields)
        _persist(run)

    try:
        _log(run, f"Run started: {len(dev)} dev / {len(holdout)} holdout queries, "
                  f"{run['loops']} rounds, LLM {llm_config['provider']}:{llm_config['model'] or 'default'}")

        baseline_config = run["baseline_config"]
        best_config = baseline_config
        best_metrics = evaluate(best_config, dev)
        _write_json(run_dir / "iter_000_baseline.json", {
            "iteration": 0, "status": "baseline", "config": best_config, "metrics": best_metrics,
        })
        _write_json(run_dir / "best_config.json", {"iteration": 0, "accuracy": best_metrics["accuracy"], "config": best_config})
        _append_results_tsv(run_id, 0, best_metrics, "baseline", f"baseline ({run['start_version']})")
        update(baseline=summarise(best_metrics), best_accuracy=best_metrics["accuracy"], phase="looping")
        _log(run, f"Baseline dev accuracy {best_metrics['accuracy']:.4f}")

        history: List[Dict[str, Any]] = []
        consecutive_crashes = 0

        for i in range(1, run["loops"] + 1):
            if stop.is_set():
                _log(run, "Stop requested; ending the loop early")
                break

            update(phase=f"round {i}: asking the LLM")
            user_prompt = build_user_prompt(best_config, best_metrics, history, i, run["loops"], run["seed"])
            with open(run_dir / f"iter_{i:03d}_prompt.txt", "w", encoding="utf-8") as f:
                f.write(SYSTEM_PROMPT + "\n\n-----\n\n" + user_prompt)

            entry: Dict[str, Any] = {"iteration": i, "timestamp": _now()}
            raw_reply = None
            try:
                candidate, hypothesis, warnings, raw_reply = propose(llm_config, user_prompt)
            except (llm.LLMError, ValueError) as e:
                consecutive_crashes += 1
                entry.update(status="crash", error=str(e), hypothesis=None, config=None,
                             accuracy=None, macro_f1=None, delta_vs_best=None,
                             best_accuracy=best_metrics["accuracy"], warnings=[])
                _write_json(run_dir / f"iter_{i:03d}.json", {**entry, "llm_reply": raw_reply})
                _append_results_tsv(run_id, i, None, "crash", str(e))
                history.append(entry)
                with _state_lock:
                    run["iterations"].append(entry)
                _persist(run)
                _log(run, f"Round {i}: CRASH - {e}")
                if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                    raise RuntimeError(f"{MAX_CONSECUTIVE_CRASHES} rounds in a row failed. Last error: {e}")
                continue
            consecutive_crashes = 0

            update(phase=f"round {i}: evaluating on {len(dev)} dev queries")
            metrics = evaluate(candidate, dev)
            delta = round(metrics["accuracy"] - best_metrics["accuracy"], 4)
            kept = metrics["accuracy"] > best_metrics["accuracy"]
            status = "keep" if kept else "discard"

            if kept:
                best_config, best_metrics = candidate, metrics
                _write_json(run_dir / "best_config.json", {"iteration": i, "accuracy": metrics["accuracy"], "config": candidate})

            entry.update(status=status, error=None, hypothesis=hypothesis, config=candidate,
                         accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                         delta_vs_best=delta, best_accuracy=best_metrics["accuracy"],
                         per_class=metrics["per_class"], warnings=warnings)
            _write_json(run_dir / f"iter_{i:03d}.json", {**entry, "metrics": metrics, "llm_reply": raw_reply})
            _append_results_tsv(run_id, i, metrics, status, hypothesis)
            history.append(entry)
            with _state_lock:
                run["iterations"].append(entry)
                if kept:
                    run.update(best_config=candidate, best_accuracy=metrics["accuracy"], best_iteration=i)
            _persist(run)
            _log(run, f"Round {i}: {status.upper()} accuracy {metrics['accuracy']:.4f} ({delta:+.4f}) - {hypothesis}")

        update(phase="scoring the holdout set")
        holdout_baseline = evaluate(baseline_config, holdout)
        improved = run["best_iteration"] > 0
        holdout_best = evaluate(best_config, holdout) if improved else holdout_baseline
        _write_json(run_dir / "holdout_results.json", {"baseline": holdout_baseline, "best": holdout_best})
        _log(run, f"Holdout accuracy: baseline {holdout_baseline['accuracy']:.4f}, best {holdout_best['accuracy']:.4f}")

        update(
            holdout={"baseline": summarise(holdout_baseline), "best": summarise(holdout_best)},
            improved=improved,
            status="stopped" if stop.is_set() else "completed",
            phase="done",
            finished_at=_now(),
        )
        _log(run, f"Run finished: dev accuracy {run['baseline']['accuracy']:.4f} -> {run['best_accuracy']:.4f} "
                  f"(best round {run['best_iteration']})")
        _record_experiment(run)
    except Exception as e:  # noqa: BLE001 - a failed run must be recorded, not lost with the thread
        _log(run, f"Run failed: {e}")
        update(status="failed", phase="done", error=str(e), finished_at=_now())
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
