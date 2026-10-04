"""Laya Router inference for intent classification.

A "prompt config" is everything about a Laya call that can vary between model
versions:

- state_template: how the query is rendered into the state. Either a string, or a
  dict of field -> string which becomes a JSON state (the form Laya's own presets
  use, with the instructions naming the field in backticks).
- serp_results: how many top SERP results the {serp_*} placeholders draw on.
- instructions / criteria: the question text and one description per intent label.
- model: which Laya checkpoint answers ("auto" lets the Router decide per query).
- label_bias: per-label offsets added to the log-probabilities before the argmax.
  Fitted on the dev set by the Karpathy loop, never written by the LLM.

The labels themselves are fixed.
"""

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app import serp
from app.config import CHECKPOINTS_DIR, LAYA_DEVICE
from app.tasks.intent.labels import SearchIntent

QUESTION_ID = "intent"
BASELINE_VERSION = "v1_baseline"
LABELS = SearchIntent.all_labels()

LAYA_MODELS = ["auto", "english", "multilingual", "typed-decisions"]
SERP_PLACEHOLDERS = ["{serp_sites}", "{serp_titles}", "{serp_snippets}"]
DEFAULT_SERP_RESULTS = 4
MAX_STATE_FIELDS = 6

# Laya silently keeps only the first 48 tokens of each option description, and the
# whole question (instructions + options) must fit its 192-token head budget.
MAX_CRITERION_WORDS = 35
MAX_INSTRUCTION_WORDS = 60

DEFAULT_CONFIG: Dict[str, Any] = {
    "state_template": "{query}",
    "instructions": "Classify the search query intent based on what the user is trying to accomplish.",
    "criteria": {
        SearchIntent.INFORMATIONAL.value: "User seeks information, education, or knowledge (how to, what is, why, explain, tutorial, guide)",
        SearchIntent.NAVIGATIONAL.value: "User seeks to reach a specific website or resource (login, sign in, account, official site, brand name)",
        SearchIntent.COMMERCIAL.value: "User researches products/services before purchasing (best, top, review, comparison, vs, alternative)",
        SearchIntent.TRANSACTIONAL.value: "User intends to complete a transaction (buy, purchase, order, download, book, subscribe, price, deal)",
    },
}

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


def validate_config(config: Dict[str, Any], allow_serp: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    """Check a prompt config and return (normalised config, warnings).

    Raises ValueError when the config cannot be used at all.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    template = config.get("state_template", DEFAULT_CONFIG["state_template"])
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
    if sum(text.count("{query}") for text in strings) != 1:
        raise ValueError("state_template must contain {query} exactly once")
    if not allow_serp and any(p in text for text in strings for p in SERP_PLACEHOLDERS):
        raise ValueError("SERP placeholders are not enabled for this run")

    instructions = config.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")

    criteria = config.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != set(LABELS):
        raise ValueError(f"criteria must have exactly these keys: {LABELS}")
    for label in LABELS:
        if not isinstance(criteria[label], str) or not criteria[label].strip():
            raise ValueError(f"criteria['{label}'] must be a non-empty string")

    warnings = []
    if len(instructions.split()) > MAX_INSTRUCTION_WORDS:
        warnings.append(f"instructions are over {MAX_INSTRUCTION_WORDS} words and may be truncated")
    for label in LABELS:
        if len(criteria[label].split()) > MAX_CRITERION_WORDS:
            warnings.append(f"criteria['{label}'] is over {MAX_CRITERION_WORDS} words; Laya ignores the tail")

    normalised: Dict[str, Any] = {
        "state_template": template,
        "instructions": instructions.strip(),
        # Fixed label order: option order is part of what the model sees.
        "criteria": {label: criteria[label].strip() for label in LABELS},
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


def build_questions(config: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a prompt config into the `questions` dict Router.predict expects."""
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": config["instructions"],
            "criteria": dict(config["criteria"]),
        }
    }


def render_state(config: Dict[str, Any], query: str) -> Union[str, Dict[str, str]]:
    """Fill the state template for one query."""
    values = {"{query}": query}
    if uses_serp(config):
        results = serp.lookup(query)[: config.get("serp_results", DEFAULT_SERP_RESULTS)]
        values["{serp_sites}"] = ", ".join(r["site"] for r in results)
        values["{serp_titles}"] = " | ".join(r["title"] for r in results)
        values["{serp_snippets}"] = " | ".join(r["snippet"] for r in results)

    def fill(text: str) -> str:
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        return text

    template = config.get("state_template", DEFAULT_CONFIG["state_template"])
    if isinstance(template, dict):
        return {field: fill(text) for field, text in template.items()}
    return fill(template)


def predict_proba(queries: List[str], config: Dict[str, Any], batch_size: int = 32) -> np.ndarray:
    """Laya's probability for each label (columns in LABELS order), one row per query."""
    questions = build_questions(config)
    model = config.get("model", "auto")
    requests = []
    for query in queries:
        request = {"state": render_state(config, query), "questions": questions}
        if model != "auto":
            request["model"] = model
        requests.append(request)
    results = get_router().predict_batch(requests, batch_size=batch_size, sort_by_length=True)
    return np.array([
        [result["answers"][QUESTION_ID]["probabilities"][label] for label in LABELS]
        for result in results
    ], dtype=float)


def apply_label_bias(proba: np.ndarray, label_bias: Optional[Dict[str, float]]) -> np.ndarray:
    """Shift log-probabilities by the per-label bias and renormalise."""
    if not label_bias:
        return proba
    logits = np.log(proba + 1e-9) + np.array([label_bias.get(label, 0.0) for label in LABELS])
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def predict_batch(
    queries: List[str],
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Classify many queries with one prompt config.

    Returns one {"intent", "confidence", "probabilities"} dict per query, in order.
    """
    config = config or DEFAULT_CONFIG
    proba = apply_label_bias(predict_proba(queries, config, batch_size), config.get("label_bias"))

    predictions = []
    for row in proba:
        choice = int(row.argmax())
        predictions.append({
            "intent": LABELS[choice],
            "confidence": round(float(row[choice]), 4),
            "probabilities": {label: round(float(p), 4) for label, p in zip(LABELS, row)},
        })
    return predictions


def classify_with_laya(
    query: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[SearchIntent, float]:
    """Classify one query. Returns (predicted_intent, confidence)."""
    prediction = predict_batch([query], config)[0]
    return SearchIntent(prediction["intent"]), prediction["confidence"]


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


def load_version_config(version: Optional[str]) -> Dict[str, Any]:
    """Return the prompt config saved under a version name.

    Raises ValueError for a version that is neither the baseline nor a saved checkpoint.
    """
    if not version or version == BASELINE_VERSION:
        return DEFAULT_CONFIG
    path = checkpoint_path(version)
    if not path.exists():
        raise ValueError(f"No saved model version named '{version}'")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["config"]
