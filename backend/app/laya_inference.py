"""Laya Router inference for intent classification.

A "prompt config" is everything the Karpathy loop is allowed to change about a
Laya call: how the query is rendered into the state, the question instructions,
and the description of each intent label. The labels themselves are fixed.
"""

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.config import CHECKPOINTS_DIR, LAYA_DEVICE
from app.tasks.intent.labels import SearchIntent

QUESTION_ID = "intent"
BASELINE_VERSION = "v1_baseline"

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
            _router = Router(device=LAYA_DEVICE)
        return _router


def validate_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Check a prompt config and return (normalised config, warnings).

    Raises ValueError when the config cannot be used at all.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    template = config.get("state_template", DEFAULT_CONFIG["state_template"])
    if not isinstance(template, str) or "{query}" not in template:
        raise ValueError("state_template must be a string containing {query}")

    instructions = config.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")

    criteria = config.get("criteria")
    labels = SearchIntent.all_labels()
    if not isinstance(criteria, dict) or set(criteria) != set(labels):
        raise ValueError(f"criteria must have exactly these keys: {labels}")
    for label in labels:
        if not isinstance(criteria[label], str) or not criteria[label].strip():
            raise ValueError(f"criteria['{label}'] must be a non-empty string")

    warnings = []
    if len(instructions.split()) > MAX_INSTRUCTION_WORDS:
        warnings.append(f"instructions are over {MAX_INSTRUCTION_WORDS} words and may be truncated")
    for label in labels:
        if len(criteria[label].split()) > MAX_CRITERION_WORDS:
            warnings.append(f"criteria['{label}'] is over {MAX_CRITERION_WORDS} words; Laya ignores the tail")

    normalised = {
        "state_template": template,
        "instructions": instructions.strip(),
        # Fixed label order: option order is part of what the model sees.
        "criteria": {label: criteria[label].strip() for label in labels},
    }
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


def render_state(config: Dict[str, Any], query: str) -> str:
    return config["state_template"].replace("{query}", query)


def predict_batch(
    queries: List[str],
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Classify many queries with one prompt config.

    Returns one {"intent", "confidence", "probabilities"} dict per query, in order.
    """
    config = config or DEFAULT_CONFIG
    questions = build_questions(config)
    requests = [{"state": render_state(config, q), "questions": questions} for q in queries]
    results = get_router().predict_batch(requests, batch_size=batch_size)

    predictions = []
    for result in results:
        answer = result["answers"][QUESTION_ID]
        choice = answer["choice"]
        predictions.append({
            "intent": choice,
            "confidence": round(float(answer["probabilities"][choice]), 4),
            "probabilities": answer["probabilities"],
        })
    return predictions


def classify_with_laya(
    query: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[SearchIntent, float]:
    """Classify one query. Returns (predicted_intent, confidence)."""
    prediction = predict_batch([query], config)[0]
    return SearchIntent(prediction["intent"]), prediction["confidence"]


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
