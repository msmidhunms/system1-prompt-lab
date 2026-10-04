"""The models an example can be run on.

Laya is the first; the others are open zero-shot classifiers that take the same thing
a prompt config holds (the input text, a question, one description per label) and give
back one probability per label. Each engine turns the config into its own model's
input format:

- gliclass, verdict: all labels in one forward pass (GLiClass architecture).
- nli_*: one pass per label; the description becomes an NLI hypothesis.
- modernbert_instruct: one pass; a multiple-choice prompt answered by the masked-LM head.

Only one of these models is kept in memory at a time, so the set fits a small machine.
"""

import json
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.config import ENGINE_DEVICE

LAYA = "laya"

ENGINES: Dict[str, Dict[str, Any]] = {
    LAYA: {
        "name": "Laya",
        "repo": "convaiinnovations/laya",
        "params": None,
        "family": "laya",
        "passes": "one pass",
        "notes": "The System 1 decision model this harness was built around.",
    },
    "gliclass": {
        "name": "GLiClass modern-base v2.0",
        "repo": "knowledgator/gliclass-modern-base-v2.0",
        "params": "151M",
        "family": "gliclass",
        "passes": "one pass",
        "notes": "Scores every label in a single forward pass.",
    },
    "verdict": {
        "name": "Verdict 1.4 (OpenJev)",
        "repo": "heman10x/rlcd-modernbert-151m",
        "params": "151M",
        "family": "verdict",
        "passes": "one pass",
        "notes": "GLiClass backbone post-trained for calibrated decisions; up to 24 options, 512 tokens.",
    },
    "nli_xsmall": {
        "name": "DeBERTa-v3 xsmall zero-shot v1.1",
        "repo": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "params": "70.8M",
        "family": "nli",
        "passes": "one pass per label",
        "notes": "The smallest NLI classifier, built for edge devices.",
    },
    "nli_small_long": {
        "name": "DeBERTa small long NLI",
        "repo": "tasksource/deberta-small-long-nli",
        "params": "142M",
        "family": "nli",
        "passes": "one pass per label",
        "notes": "Trained on 600 tasks; handles long inputs.",
    },
    "nli_large": {
        "name": "DeBERTa-v3 large zero-shot v2.0",
        "repo": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "params": "435M",
        "family": "nli",
        "passes": "one pass per label",
        "notes": "The accurate end of the NLI family; the slowest here.",
    },
    "modernbert_instruct": {
        "name": "ModernBERT-Large-Instruct",
        "repo": "answerdotai/ModernBERT-Large-Instruct",
        "params": "396M",
        "family": "mlm",
        "passes": "one pass",
        "notes": "Answers a multiple-choice prompt with its masked-language head.",
    },
}

# What the optimizer's LLM is told about writing prompts for each kind of model.
PROMPT_NOTES: Dict[str, str] = {
    "gliclass": (
        "- The model is GLiClass, a zero-shot classifier. It reads the label descriptions and the text together in one "
        "pass and scores each description against the text.\n"
        "- The instructions are placed before the text as a question. Keep them short.\n"
        "- Descriptions work best as short plain phrases that could describe the text, not keyword lists."
    ),
    "verdict": (
        "- The model is Verdict, a GLiClass-based decision model. Each description is turned into the sentence "
        '"It is <description>", so write descriptions that complete that sentence (for example "a question about a '
        'refund", not "refund").\n'
        '- The instructions are given to the model as "Question: <instructions>", followed by the text.\n'
        "- It was trained on short contexts; long inputs are cut to 512 tokens."
    ),
    "nli": (
        "- The model is an NLI (entailment) zero-shot classifier. For each label it is asked whether the text entails "
        "a hypothesis sentence, and the labels compete on that entailment score.\n"
        '- The hypothesis is built from the instructions and the description. If the instructions contain "{}", they '
        "are the hypothesis template and each description is put in its place (for example \"This message is about "
        '{}."). Otherwise the hypothesis is "This text is about <description>.".\n'
        "- So the instructions should be a short statement template with {} in it, and each description a short noun "
        "phrase that reads naturally inside it."
    ),
    "mlm": (
        "- The model is ModernBERT-Large-Instruct. It is shown a multiple-choice prompt: the instructions and the text "
        "as the question, then the options as \"- A: <description>\", \"- B: <description>\"..., and it predicts the "
        "letter of the answer.\n"
        "- The instructions should be a clear question about the text. Descriptions should be short, distinct answer "
        "options."
    ),
}

BATCH = 16
MAX_TOKENS = 512

_loaded: Dict[str, Any] = {"id": None}
_lock = threading.Lock()


def get_engine(engine_id: Optional[str]) -> Dict[str, Any]:
    engine_id = engine_id or LAYA
    if engine_id not in ENGINES:
        raise ValueError(f"Unknown model '{engine_id}'. Choose one of: {', '.join(ENGINES)}")
    return {"id": engine_id, **ENGINES[engine_id]}


def public() -> List[Dict[str, Any]]:
    return [get_engine(engine_id) for engine_id in ENGINES]


def _device() -> str:
    if ENGINE_DEVICE not in ("", "auto"):
        return ENGINE_DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(engine_id: str) -> Dict[str, Any]:
    """The engine's model, loading it (and dropping the previously loaded one) if needed. Call with _lock held."""
    if _loaded["id"] == engine_id:
        return _loaded
    import torch
    from transformers import AutoTokenizer

    _loaded.clear()
    _loaded["id"] = None
    spec = ENGINES[engine_id]
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
    if spec["family"] in ("gliclass", "verdict"):
        from gliclass import GLiClassModel
        model = GLiClassModel.from_pretrained(spec["repo"])
    elif spec["family"] == "nli":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(spec["repo"])
    else:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(spec["repo"])
    model.to(device).eval()
    _loaded.update(id=engine_id, model=model, tokenizer=tokenizer, device=device, torch=torch)
    if spec["family"] == "verdict":
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(spec["repo"], "calibrator.json"), encoding="utf-8") as f:
            _loaded["calibrator"] = json.load(f)
    return _loaded


def state_text(state: Any) -> str:
    """The text of a rendered state. A one-field JSON state is its value; more fields become "name: value" lines."""
    if isinstance(state, str):
        return state
    if len(state) == 1:
        return next(iter(state.values()))
    return "\n".join(f"{field}: {value}" for field, value in state.items())


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- GLiClass and Verdict

VERDICT_ABSTAIN = "There is insufficient evidence to decide"


def _gliclass_logits(loaded: Dict[str, Any], texts: List[str], labels: List[str]) -> np.ndarray:
    """One forward pass: <<LABEL>>label1<<LABEL>>label2<<SEP>>text -> one logit per label."""
    torch = loaded["torch"]
    prefix = "".join(f"<<LABEL>>{label}" for label in labels) + "<<SEP>>"
    inputs = loaded["tokenizer"](
        [prefix + text for text in texts], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
    ).to(loaded["device"])
    with torch.inference_mode():
        logits = loaded["model"](**inputs).logits
    return logits[:, : len(labels)].float().cpu().numpy()


def _predict_gliclass(loaded, texts, instructions, descriptions) -> np.ndarray:
    # GLiClass was trained with lower-cased labels.
    labels = [description.lower() for description in descriptions]
    question = instructions.strip()
    prompts = [f"{question}\n{text}" if question else text for text in texts]
    return _softmax(_gliclass_logits(loaded, prompts, labels))


def _predict_verdict(loaded, texts, instructions, descriptions) -> np.ndarray:
    """Verdict's own input contract (core/formatting.py in its repository) and per-option-count temperature."""
    labels = [f"It is {description}" for description in descriptions] + [VERDICT_ABSTAIN]
    prompts = [f"Question: {instructions}\n\nContext:\n{text}" for text in texts]
    calibrator = loaded["calibrator"]
    temperature = float(calibrator.get("per_k", {}).get(str(len(labels)), calibrator["temperature"]))
    proba = _softmax(_gliclass_logits(loaded, prompts, labels), temperature)
    # The abstention option is Verdict's "cannot tell". Here every input has a label, so the rest are renormalised.
    proba = proba[:, :-1]
    return proba / proba.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- NLI

DEFAULT_HYPOTHESIS = "This text is about {}."


def _predict_nli(loaded, texts, instructions, descriptions) -> np.ndarray:
    torch = loaded["torch"]
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    template = instructions if "{}" in instructions else DEFAULT_HYPOTHESIS
    hypotheses = [template.replace("{}", description) for description in descriptions]
    label2id = {name.lower(): index for name, index in model.config.label2id.items()}
    entail = next((index for name, index in label2id.items() if name.startswith("entail")), 0)
    premises = [text for text in texts for _ in hypotheses]
    pairs = [hypothesis for _ in texts for hypothesis in hypotheses]
    scores = []
    for start in range(0, len(premises), BATCH * 2):
        inputs = tokenizer(
            premises[start:start + BATCH * 2], pairs[start:start + BATCH * 2],
            padding=True, truncation="only_first", max_length=MAX_TOKENS, return_tensors="pt",
        ).to(loaded["device"])
        with torch.inference_mode():
            scores.append(model(**inputs).logits[:, entail].float().cpu().numpy())
    # The labels compete on their entailment logit, as in the single-label zero-shot pipeline.
    return _softmax(np.concatenate(scores).reshape(len(texts), len(descriptions)))


# ---------------------------------------------------------------- ModernBERT-Instruct

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _predict_mlm(loaded, texts, instructions, descriptions) -> np.ndarray:
    torch = loaded["torch"]
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    choices = "\n".join(f"- {LETTERS[i]}: {description}" for i, description in enumerate(descriptions))
    tail = f"\nCHOICES:\n{choices}\nANSWER: [unused0] {tokenizer.mask_token}"
    head = "You will be given a question and options. Select the right answer.\nQUESTION: "
    tail_tokens = len(tokenizer(tail, add_special_tokens=False)["input_ids"])
    head_tokens = len(tokenizer(f"{head}{instructions}\n", add_special_tokens=False)["input_ids"])
    room = max(32, MAX_TOKENS - tail_tokens - head_tokens - 4)
    prompts = []
    for text in texts:
        # The text is cut, never the options or the mask at the end of the prompt.
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:room]
        prompts.append(f"{head}{instructions}\n{tokenizer.decode(ids)}{tail}")
    letter_ids = [tokenizer.encode(f" {LETTERS[i]}", add_special_tokens=False)[-1] for i in range(len(descriptions))]
    rows = []
    for start in range(0, len(prompts), BATCH):
        inputs = tokenizer(prompts[start:start + BATCH], padding=True, return_tensors="pt").to(loaded["device"])
        with torch.inference_mode():
            logits = model(**inputs).logits
        mask = (inputs["input_ids"] == tokenizer.mask_token_id).int().argmax(dim=1)
        at_mask = logits[torch.arange(logits.shape[0]), mask]
        rows.append(at_mask[:, letter_ids].float().cpu().numpy())
    return _softmax(np.concatenate(rows))


_PREDICT = {"gliclass": _predict_gliclass, "verdict": _predict_verdict, "nli": _predict_nli, "mlm": _predict_mlm}


def predict_proba(
    engine_id: str,
    texts: List[str],
    instructions: str,
    descriptions: List[str],
    on_progress: Optional[Callable[[int, int], None]] = None,
    chunk: int = 32,
) -> np.ndarray:
    """One probability per description (in order) for each text, from a non-Laya engine."""
    family = ENGINES[engine_id]["family"]
    if len(descriptions) > len(LETTERS):
        raise ValueError(f"This model takes at most {len(LETTERS)} labels")
    rows = []
    for start in range(0, len(texts), chunk):
        # The lock is taken per chunk, so an evaluation on another model can interleave (at the cost of a reload).
        with _lock:
            loaded = _load(engine_id)
            if family in ("gliclass", "verdict"):
                part = np.concatenate([
                    _PREDICT[family](loaded, texts[i:i + BATCH], instructions, descriptions)
                    for i in range(start, min(start + chunk, len(texts)), BATCH)
                ])
            else:
                part = _PREDICT[family](loaded, texts[start:start + chunk], instructions, descriptions)
        rows.append(part)
        if on_progress:
            on_progress(min(start + chunk, len(texts)), len(texts))
    return np.concatenate(rows) if rows else np.zeros((0, len(descriptions)))
