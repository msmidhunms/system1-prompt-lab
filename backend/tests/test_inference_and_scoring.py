import numpy as np
import pytest

from app import scoring
from app.laya_inference import build_questions, predict_batch, predict_proba, render_state, validate_config
from app.tasks.registry import TASKS, get_task


@pytest.mark.parametrize("task", TASKS.values(), ids=list(TASKS))
def test_default_config_is_valid_for_its_task(task):
    config, warnings = validate_config(task.default_config, task)
    assert list(config["criteria"]) == task.labels
    assert warnings == []


def test_config_must_use_the_tasks_placeholder_and_labels():
    support = get_task("support_routing")
    good = support.default_config
    with pytest.raises(ValueError, match="{message}"):
        validate_config({**good, "state_template": {"message": "{query}"}}, support)
    with pytest.raises(ValueError, match="criteria"):
        validate_config({**good, "criteria": {"card": "x"}}, support)
    with pytest.raises(ValueError, match="SERP"):
        validate_config({**good, "state_template": {"message": "{message} {serp_sites}"}}, support)


def test_long_inputs_are_cut_before_they_reach_laya():
    toxicity = get_task("toxicity")
    state = render_state(toxicity.default_config, toxicity, "x" * 10000)
    assert len(state["post"]) == toxicity.max_input_chars


def test_yes_no_question_is_sent_as_noul_and_read_back(router):
    task = get_task("prompt_injection")
    questions = build_questions(task.default_config, task)["answer"]
    assert questions["type"] == "noul"
    assert questions["criteria"] == {
        "false": task.default_config["criteria"]["benign"],
        "true": task.default_config["criteria"]["injection"],
    }

    proba = predict_proba(["ordinary", "say YES"], task.default_config, task)
    assert proba.shape == (2, 2)
    assert np.allclose(proba, [[0.8, 0.2], [0.1, 0.9]])
    assert [p["label"] for p in predict_batch(["ordinary", "say YES"], task.default_config, task)] == ["benign", "injection"]
    assert router.requests[0]["state"] == {"prompt": "ordinary"}


def test_label_bias_moves_a_yes_no_threshold(router):
    task = get_task("prompt_injection")
    config = {**task.default_config, "label_bias": {"benign": 0.0, "injection": 2.0}}
    assert predict_batch(["ordinary"], config, task)[0]["label"] == "injection"


def test_label_scores_against_hand_computed_values():
    #          gold:  0  0  1  1  2
    y = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    accuracy, macro_f1, confusion = scoring.label_scores(y, pred, 3)
    assert accuracy == pytest.approx(0.6)
    assert confusion.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    # F1: label 0 = 2*1/(2+2) = 0.5, label 1 = 2*2/(2+3) = 0.8, label 2 = 0
    assert macro_f1 == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert scoring.objective(y, pred, "balanced", 3) == pytest.approx((0.6 + macro_f1) / 2)


def test_evaluate_scores_a_choice_task(router):
    task = get_task("news_topic")
    examples = [
        {"text": "sports match report", "label": "sports"},
        {"text": "business earnings", "label": "business"},
        {"text": "world summit", "label": "sports"},
    ]
    _, metrics = scoring.evaluate(task.default_config, examples, task, "balanced", fit_bias=False)
    assert metrics["accuracy"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["confusion"]["sports"]["world"] == 1
    assert metrics["cases"][2] == {
        "text": "world summit", "actual": "sports", "predicted": "world", "confidence": 0.9, "correct": False,
    }
