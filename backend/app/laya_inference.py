"""Laya Router inference for the examples in app.tasks.registry.

A "prompt config" is everything about a Laya call that can vary between model
versions of one task:

- state_template: how the input text is rendered into the state. Either a string, or a
  dict of field -> string which becomes a JSON state (the form Laya's own presets
  use, with the instructions naming the field in backticks). The text goes where the
  task's placeholder is, e.g. {query} for search intent or {message} for support routing.
- serp_results: how many top SERP results the {serp_*} placeholders draw on (search intent only).
- instructions / criteria: the question text and one description per label.
- model: which Laya checkpoint answers ("auto" lets the Router decide per input).
- label_bias: per-label offsets added to the log-probabilities before the argmax.
  Fitted on the dev set by the Karpathy loop, never written by the LLM.

The labels themselves are fixed by the task.
"""

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app import serp
from app.config import CHECKPOINTS_DIR, LAYA_DEVICE
from app.tasks.registry import DEFAULT_TASK_ID, Task

QUESTION_ID = "answer"
BASELINE_VERSION = "v1_baseline"

LAYA_MODELS = ["auto", "english", "multilingual", "typed-decisions"]
SERP_PLACEHOLDERS = ["{serp_sites}", "{serp_titles}", "{serp_snippets}"]
DEFAULT_SERP_RESULTS = 4
MAX_STATE_FIELDS = 6

_router = None
_router_lock = threading.Lock()


def get_router():
    """Return the shared Laya Router, building it on first use."""
    global _router
    with _router_lock:
        if _router is None:
            from laya import Router
            device = None if LAYA_DEVICE in ("", "auto") else LAYA_DEVICE
            _router = Router(device=device)
        return _router


def _template_strings(template: Union[str, Dict[str, str]]) -> List[str]:
    return [template] if isinstance(template, str) else list(template.values())


def uses_serp(config: Dict[str, Any]) -> bool:
    return any(p in text for text in _template_strings(config["state_template"]) for p in SERP_PLACEHOLDERS)


def validate_config(config: Dict[str, Any], task: Task, allow_serp: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    """Check a prompt config against a task and return (normalised config, warnings).

    Raises ValueError when the config cannot be used at all.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    labels = task.labels
    allow_serp = allow_serp and task.supports_serp

    template = config.get("state_template", task.default_config["state_template"])
    if isinstance(template, dict):
        if not 1 <= len(template) <= MAX_STATE_FIELDS:
            raise ValueError(f"state_template must have between 1 and {MAX_STATE_FIELDS} fields")
        for field, value in template.items():
            if not isinstance(field, str) or not re.fullmatch(r"\w+", field):
                raise ValueError("state_template field names must be plain words (letters, digits, _)")
            if not isinstance(value, str):
                raise ValueError(f"state_template['{field}'] must be a string")
    elif not isinstance(template, str):
        raise ValueError("state_template must be a string or an object of field -> string")
    strings = _template_strings(template)
    if sum(text.count(task.placeholder) for text in strings) != 1:
        raise ValueError(f"state_template must contain {task.placeholder} exactly once")
    if not allow_serp and any(p in text for text in strings for p in SERP_PLACEHOLDERS):
        raise ValueError("SERP placeholders are not enabled for this run")

    instructions = config.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")

    criteria = config.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != set(labels):
        raise ValueError(f"criteria must have exactly these keys: {labels}")
    for label in labels:
        if not isinstance(criteria[label], str) or not criteria[label].strip():
            raise ValueError(f"criteria['{label}'] must be a non-empty string")

    warnings = []
    if len(instructions.split()) > task.max_instruction_words:
        warnings.append(f"instructions are over {task.max_instruction_words} words and may be truncated")
    for label in labels:
        if len(criteria[label].split()) > task.max_criterion_words:
            warnings.append(f"criteria['{label}'] is over {task.max_criterion_words} words; Laya ignores the tail")

    normalised: Dict[str, Any] = {
        "state_template": template,
        "instructions": instructions.strip(),
        # Fixed label order: option order is part of what the model sees.
        "criteria": {label: criteria[label].strip() for label in labels},
    }
    if any(p in text for text in strings for p in SERP_PLACEHOLDERS):
        results = config.get("serp_results", DEFAULT_SERP_RESULTS)
        if not isinstance(results, int) or isinstance(results, bool) or not 1 <= results <= serp.MAX_RESULTS:
            raise ValueError(f"serp_results must be an integer between 1 and {serp.MAX_RESULTS}")
        normalised["serp_results"] = results
    if config.get("model") is not None:
        if config["model"] not in LAYA_MODELS:
            raise ValueError(f"model must be one of {LAYA_MODELS}")
        normalised["model"] = config["model"]
    return normalised, warnings


def build_questions(config: Dict[str, Any], task: Task) -> Dict[str, Any]:
    """Turn a prompt config into the `questions` dict Router.predict expects."""
    criteria = dict(config["criteria"])
    if task.question_type == "noul":
        # Laya's yes/no question takes its two descriptions under "false" and "true".
        no, yes = task.labels
        criteria = {"false": criteria[no], "true": criteria[yes]}
    return {
        QUESTION_ID: {
            "type": task.question_type,
            "instructions": config["instructions"],
            "criteria": criteria,
        }
    }


def render_state(config: Dict[str, Any], task: Task, text: str) -> Union[str, Dict[str, str]]:
    """Fill the state template for one input."""
    values = {task.placeholder: text[: task.max_input_chars]}
    if uses_serp(config):
        results = serp.lookup(text)[: config.get("serp_results", DEFAULT_SERP_RESULTS)]
        values["{serp_sites}"] = ", ".join(r["site"] for r in results)
        values["{serp_titles}"] = " | ".join(r["title"] for r in results)
        values["{serp_snippets}"] = " | ".join(r["snippet"] for r in results)

    def fill(template_text: str) -> str:
        for placeholder, value in values.items():
            template_text = template_text.replace(placeholder, value)
        return template_text

    template = config.get("state_template", task.default_config["state_template"])
    if isinstance(template, dict):
        return {field: fill(value) for field, value in template.items()}
    return fill(template)


def _answer_probabilities(answer: Dict[str, Any], task: Task) -> List[float]:
    """One probability per task label from a Laya answer."""
    if task.question_type == "noul":
        yes = float(answer["noul"])
        return [1.0 - yes, yes]
    return [answer["probabilities"][label] for label in task.labels]


def predict_proba(texts: List[str], config: Dict[str, Any], task: Task, batch_size: int = 32) -> np.ndarray:
    """Laya's probability for each label (columns in task.labels order), one row per input."""
    questions = build_questions(config, task)
    model = config.get("model", "auto")
    requests = []
    for text in texts:
        request = {"state": render_state(config, task, text), "questions": questions}
        if model != "auto":
            request["model"] = model
        requests.append(request)
    results = get_router().predict_batch(requests, batch_size=batch_size, sort_by_length=True)
    return np.array(
        [_answer_probabilities(result["answers"][QUESTION_ID], task) for result in results], dtype=float
    ).reshape(len(texts), len(task.labels))


def apply_label_bias(proba: np.ndarray, label_bias: Optional[Dict[str, float]], labels: List[str]) -> np.ndarray:
    """Shift log-probabilities by the per-label bias and renormalise."""
    if not label_bias:
        return proba
    logits = np.log(proba + 1e-9) + np.array([label_bias.get(label, 0.0) for label in labels])
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def predict_batch(
    texts: List[str],
    config: Dict[str, Any],
    task: Task,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Classify many inputs with one prompt config.

    Returns one {"label", "confidence", "probabilities"} dict per input, in order.
    """
    labels = task.labels
    proba = apply_label_bias(predict_proba(texts, config, task, batch_size), config.get("label_bias"), labels)

    predictions = []
    for row in proba:
        choice = int(row.argmax())
        predictions.append({
            "label": labels[choice],
            "confidence": round(float(row[choice]), 4),
            "probabilities": {label: round(float(p), 4) for label, p in zip(labels, row)},
        })
    return predictions


def guess_language(query: str) -> str:
    """Best-effort language of a query that has no reviewed language.

    Uses Laya's detector, which spots other scripts and accented text but cannot tell
    unaccented Latin-script languages from English, so "en" here means "nothing says otherwise".
    """
    from laya import detect_language
    detection = detect_language(query)
    if detection["is_english"]:
        return "en"
    return detection.get("language") or "other"


# ---------------------------------------------------------------- saved versions

VERSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def checkpoint_path(version: str):
    if not VERSION_NAME_RE.match(version):
        raise ValueError("version names may only contain letters, digits, '_', '-' and '.'")
    return CHECKPOINTS_DIR / f"{version}.json"


def save_checkpoint(version: str, payload: Dict[str, Any]) -> None:
    with open(checkpoint_path(version), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_version_config(version: Optional[str], task: Task) -> Dict[str, Any]:
    """Return the prompt config saved under a version name for this task.

    Raises ValueError for a version that is neither the baseline nor a saved checkpoint of the task.
    """
    if not version or version == BASELINE_VERSION:
        return task.default_config
    path = checkpoint_path(version)
    if not path.exists():
        raise ValueError(f"No saved model version named '{version}'")
    with open(path, encoding="utf-8") as f:
        checkpoint = json.load(f)
    # Checkpoints saved before there were several examples have no task and belong to the default one.
    saved_for = checkpoint.get("task") or DEFAULT_TASK_ID
    if saved_for != task.id:
        raise ValueError(f"Model version '{version}' was saved for the example '{saved_for}', not '{task.id}'")
    return checkpoint["config"]
