"""Scoring a Laya prompt config on labelled examples.

Shared by the Karpathy loop and the evaluation endpoint, so both report the same numbers.
Labels are passed as indices into the task's label list (k labels).
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from app.laya_inference import apply_label_bias, predict_proba
from app.tasks.registry import Task

METRICS = {
    "balanced": "mean of accuracy and macro-F1",
    "accuracy": "accuracy",
    "macro_f1": "macro-F1",
}
KEEP_CONFIDENCE = 0.8
BOOTSTRAP_SAMPLES = 1000
BIAS_GRID = sorted(np.round(np.linspace(-2.0, 2.0, 41), 2), key=abs)


def label_scores(y: np.ndarray, pred: np.ndarray, k: int) -> Tuple[float, float, np.ndarray]:
    """(accuracy, macro-F1 over labels present in y, confusion[actual][predicted])."""
    confusion = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    hits = np.diag(confusion)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    f1 = 2 * hits / np.maximum(support + predicted, 1)
    return hits.sum() / len(y), float(f1[support > 0].mean()), confusion


def objective(y: np.ndarray, pred: np.ndarray, metric: str, k: int) -> float:
    accuracy, macro_f1, _ = label_scores(y, pred, k)
    if metric == "accuracy":
        return float(accuracy)
    if metric == "macro_f1":
        return macro_f1
    return float((accuracy + macro_f1) / 2)


def fit_label_bias(logp: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    """Per-label offsets on the log-probabilities that maximise the objective (coordinate ascent)."""
    k = logp.shape[1]
    bias = np.zeros(k)
    best = objective(y, logp.argmax(axis=1), metric, k)
    for _ in range(3):
        moved = False
        for label in range(k):
            # Smallest offsets first, and only strict gains move: ties keep the bias small.
            for value in BIAS_GRID:
                trial = bias.copy()
                trial[label] = value
                score = objective(y, (logp + trial).argmax(axis=1), metric, k)
                if score > best + 1e-9:
                    best, bias, moved = score, trial, True
        if not moved:
            break
    return bias


def prob_better(y: np.ndarray, pred_new: np.ndarray, pred_old: np.ndarray, metric: str, k: int, seed: int) -> float:
    """Paired bootstrap: probability that pred_new scores above pred_old on a resampled set."""
    rng = np.random.default_rng(seed)
    wins = 0.0
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        new, old = objective(y[idx], pred_new[idx], metric, k), objective(y[idx], pred_old[idx], metric, k)
        wins += 1.0 if new > old else 0.5 if new == old else 0.0
    return wins / BOOTSTRAP_SAMPLES


def evaluate(
    config: Dict[str, Any],
    examples: List[Dict[str, str]],
    task: Task,
    metric: str,
    fit_bias: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Laya with a prompt config over labelled examples ({"text", "label"}) and score it.

    With fit_bias, a label bias is fitted on these examples and scored cross-fitted (each
    half scored with the bias fitted on the other half). If that beats the plain argmax by
    more than noise, the returned config carries the bias fitted on all the examples;
    otherwise it carries none. Without fit_bias, the config's own label_bias (if any) is
    applied as is.

    Returns (config to keep, metrics).
    """
    labels = task.labels
    k = len(labels)
    y = np.array([labels.index(e["label"]) for e in examples])
    proba = predict_proba([e["text"] for e in examples], config, task)
    raw_pred = proba.argmax(axis=1)

    if fit_bias:
        logp = np.log(proba + 1e-9)
        pred = np.empty(len(y), dtype=int)
        fold = np.arange(len(y)) % 2
        for held in (0, 1):
            bias = fit_label_bias(logp[fold != held], y[fold != held], metric)
            pred[fold == held] = (logp[fold == held] + bias).argmax(axis=1)
        # A bias has to earn its place the same way a prompt does. When the model is not
        # skewed towards a label, fitting one only adds noise, so the plain argmax stands.
        helps = (objective(y, pred, metric, k) > objective(y, raw_pred, metric, k)
                 and prob_better(y, pred, raw_pred, metric, k, seed=0) >= KEEP_CONFIDENCE)
        config = {key: value for key, value in config.items() if key != "label_bias"}
        if helps:
            full_bias = fit_label_bias(logp, y, metric)
            config["label_bias"] = {label: float(b) for label, b in zip(labels, full_bias)}
            proba = apply_label_bias(proba, config["label_bias"], labels)
        else:
            pred = raw_pred
    else:
        proba = apply_label_bias(proba, config.get("label_bias"), labels)
        pred = proba.argmax(axis=1)

    accuracy, macro_f1, confusion = label_scores(y, pred, k)
    raw_accuracy, raw_macro_f1, _ = label_scores(y, raw_pred, k)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    per_class = {}
    for i, label in enumerate(labels):
        hit = int(confusion[i, i])
        precision = hit / predicted[i] if predicted[i] else 0.0
        recall = hit / support[i] if support[i] else 0.0
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(2 * hit / (support[i] + predicted[i]) if support[i] + predicted[i] else 0.0, 3),
            "support": int(support[i]),
        }
    cases = [
        {
            "text": example["text"],
            "actual": example["label"],
            "predicted": labels[p],
            "confidence": round(float(proba[i, p]), 4),
            "correct": bool(p == y[i]),
        }
        for i, (example, p) in enumerate(zip(examples, pred))
    ]
    metrics = {
        "score": round(objective(y, pred, metric, k), 4),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(macro_f1, 4),
        # Plain argmax with no label bias, to show what calibration contributes.
        "raw_accuracy": round(float(raw_accuracy), 4),
        "raw_macro_f1": round(raw_macro_f1, 4),
        "correct": int(np.diag(confusion).sum()),
        "total": len(cases),
        "predicted_counts": {label: int(n) for label, n in zip(labels, predicted)},
        "gold_counts": {label: int(n) for label, n in zip(labels, support)},
        "per_class": per_class,
        "confusion": {a: {p: int(confusion[i, j]) for j, p in enumerate(labels)} for i, a in enumerate(labels)},
        "cases": cases,
    }
    return config, metrics


def pred_array(metrics: Dict[str, Any], labels: List[str]) -> np.ndarray:
    return np.array([labels.index(c["predicted"]) for c in metrics["cases"]])


def gold_array(metrics: Dict[str, Any], labels: List[str]) -> np.ndarray:
    return np.array([labels.index(c["actual"]) for c in metrics["cases"]])


def summarise(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Metrics without the per-case list (for run state and API responses)."""
    return {key: value for key, value in metrics.items() if key != "cases"}
