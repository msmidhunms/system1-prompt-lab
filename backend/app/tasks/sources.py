"""Where each example's golden dataset comes from, and importing it.

The new examples use public datasets on Hugging Face. A loader downloads the files it
needs (the Parquet copies every dataset has under refs/convert/parquet), maps them to
the task's labels and returns a seeded sample. The sample is kept as a snapshot under
data/tasks/, so importing again needs no network and gives the same rows.

Search intent is imported from data/test_db.json, which is not downloadable.
"""

import json
import random
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import DATA_DIR
from app.database import GoldenRow
from app.golden import input_key
from app.tasks.registry import Task

DEFAULT_ROWS = 2000
DEFAULT_SEED = 13
SNAPSHOT_DIR = DATA_DIR / "tasks"
TEST_DB_FILE = DATA_DIR / "test_db.json"

Row = Dict[str, Any]  # {"text", "label", and optionally "meta", "secondary_labels", "language"}


def _table(repo: str, path: str, columns: List[str]) -> List[Dict[str, Any]]:
    """Rows of one Parquet file of a Hugging Face dataset."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    local = hf_hub_download(repo, path, repo_type="dataset", revision="refs/convert/parquet")
    return pq.read_table(local, columns=columns).to_pylist()


def _clean(rows: List[Row]) -> List[Row]:
    """Drop empty inputs and inputs that repeat an earlier one (ignoring case and spacing)."""
    seen, kept = set(), []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        key = input_key(text)
        if key in seen:
            continue
        seen.add(key)
        kept.append({**row, "text": text})
    return kept


def _sample(rows: List[Row], size: int, seed: int) -> List[Row]:
    """A seeded sample, in random order. The same rows for the same seed, whatever order they were read in."""
    rows = sorted(_clean(rows), key=lambda row: row["text"])
    random.Random(seed).shuffle(rows)
    return rows[:size]


def _quotas(pools: Dict[str, List[Row]], size: int, seed: int) -> List[Row]:
    """An equal share of each pool, shuffled together."""
    share = size // len(pools)
    picked = []
    for i, name in enumerate(pools):
        picked += _sample(pools[name], share, seed + i)
    return _sample(picked, len(picked), seed)


# ---------------------------------------------------------------- support routing

# Banking77's 77 intents, grouped by what the message is about.
BANKING77_GROUPS = {
    "card": [
        "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "card_not_working", "change_pin", "compromised_card",
        "contactless_not_working", "disposable_card_limits", "get_disposable_virtual_card", "get_physical_card",
        "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "order_physical_card", "pin_blocked",
        "supported_cards_and_currencies", "virtual_card_not_working", "visa_or_mastercard",
    ],
    "card_payment": [
        "Refund_not_showing_up", "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "declined_card_payment", "direct_debit_payment_not_recognised",
        "extra_charge_on_statement", "pending_card_payment", "request_refund", "reverted_card_payment?",
        "transaction_charged_twice",
    ],
    "transfer": [
        "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer", "declined_transfer",
        "failed_transfer", "pending_transfer", "receiving_money", "transfer_fee_charged", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing",
    ],
    "top_up": [
        "automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
    ],
    "cash_withdrawal": [
        "atm_support", "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "account": [
        "age_limit", "country_support", "edit_personal_details", "exchange_charge", "exchange_rate",
        "exchange_via_app", "fiat_currency_support", "lost_or_stolen_phone", "passcode_forgotten",
        "terminate_account", "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "why_verify_identity",
    ],
}
BANKING77_GROUP_OF = {intent: group for group, intents in BANKING77_GROUPS.items() for intent in intents}
assert len(BANKING77_GROUP_OF) == 77 == sum(len(intents) for intents in BANKING77_GROUPS.values())


def _load_support_routing(size: int, seed: int) -> List[Row]:
    rows = _table("mteb/banking77", "default/train/0000.parquet", ["text", "label_text"])
    unmapped = {row["label_text"] for row in rows} - set(BANKING77_GROUP_OF)
    if unmapped:
        raise ValueError(f"Banking77 intents with no group: {sorted(unmapped)}")
    return _sample(
        [{"text": row["text"], "label": BANKING77_GROUP_OF[row["label_text"]], "meta": {"intent": row["label_text"]}}
         for row in rows],
        size, seed,
    )


# ---------------------------------------------------------------- single-source examples

def _load_prompt_injection(size: int, seed: int) -> List[Row]:
    rows = _table("xTRam1/safe-guard-prompt-injection", "default/train/0000.parquet", ["text", "label"])
    return _sample([{"text": row["text"], "label": "injection" if row["label"] == 1 else "benign"} for row in rows],
                   size, seed)


TREC_LABELS = {
    "ABBR": "abbreviation", "DESC": "description", "ENTY": "entity",
    "HUM": "human", "LOC": "location", "NUM": "number",
}


def _load_question_type(size: int, seed: int) -> List[Row]:
    rows = _table("SetFit/TREC-QC", "default/train/0000.parquet", ["text", "label_coarse_original", "label_original"])
    return _sample(
        [{"text": row["text"], "label": TREC_LABELS[row["label_coarse_original"]], "meta": {"fine": row["label_original"]}}
         for row in rows],
        size, seed,
    )


AG_NEWS_LABELS = ["world", "sports", "business", "sci_tech"]


def _load_news_topic(size: int, seed: int) -> List[Row]:
    rows = _table("fancyzhx/ag_news", "default/test/0000.parquet", ["text", "label"])
    return _sample([{"text": row["text"], "label": AG_NEWS_LABELS[row["label"]]} for row in rows], size, seed)


def _load_toxicity(size: int, seed: int) -> List[Row]:
    rows = _table("OxAISH-AL-LLM/wiki_toxic", "default/balanced_train/0000.parquet", ["comment_text", "label"])
    return _sample(
        [{"text": row["comment_text"], "label": "toxic" if row["label"] == 1 else "not_toxic"} for row in rows],
        size, seed,
    )


# ---------------------------------------------------------------- assembled examples

def _load_email_triage(size: int, seed: int) -> List[Row]:
    enron = _table("SetFit/enron_spam", "default/train/0000.parquet", ["text", "label_text"])
    phishing = _table("zefang-liu/phishing-email-dataset", "default/train/0000.parquet", ["Email Text", "Email Type"])

    def rows(source_rows, text_key, label_key, value, label, source) -> List[Row]:
        return [{"text": row[text_key], "label": label, "meta": {"source": source}}
                for row in source_rows if row[label_key] == value and row[text_key]]

    share = size // 3
    # Legitimate mail is drawn from both datasets, so the source alone does not give the label away.
    legitimate = (
        _sample(rows(enron, "text", "label_text", "ham", "legitimate", "enron_spam"), share - share // 2, seed)
        + _sample(rows(phishing, "Email Text", "Email Type", "Safe Email", "legitimate", "phishing-email-dataset"),
                  share // 2, seed)
    )
    return _quotas({
        "legitimate": legitimate,
        "spam": rows(enron, "text", "label_text", "spam", "spam", "enron_spam"),
        "phishing": rows(phishing, "Email Text", "Email Type", "Phishing Email", "phishing", "phishing-email-dataset"),
    }, size, seed)


_SPEAKER = re.compile(r"^User [12]:\s*")


def _load_request_domain(size: int, seed: int) -> List[Row]:
    def rows(texts, label, source) -> List[Row]:
        return [{"text": text, "label": label, "meta": {"source": source}} for text in texts if text]

    mbpp = [row["text"] for split in ("train", "test", "validation")
            for row in _table("google-research-datasets/mbpp", f"full/{split}/0000.parquet", ["text"])]
    gsm8k = [row["question"] for row in _table("openai/gsm8k", "main/train/0000.parquet", ["question"])]
    dolly = _table("databricks/databricks-dolly-15k", "default/train/0000.parquet", ["instruction", "category"])
    sql = [f"{row['question'].strip()} The table is: {row['context'].strip()}"
           for row in _table("b-mc2/sql-create-context", "default/train/0000.parquet", ["question", "context"])]
    chat = []
    for row in _table("google/Synthetic-Persona-Chat", "default/train/0000.parquet", ["Best Generated Conversation"]):
        for line in (row["Best Generated Conversation"] or "").splitlines():
            line = _SPEAKER.sub("", line.strip())
            # Lines with a "[user 1's name]" placeholder are template residue, not conversation.
            if 12 <= len(line) <= 200 and "[" not in line:
                chat.append(line)

    return _quotas({
        "code": rows(mbpp, "code", "mbpp"),
        "math_or_logic": rows(gsm8k, "math_or_logic", "gsm8k"),
        "writing": rows([r["instruction"] for r in dolly if r["category"] == "creative_writing"],
                        "writing", "dolly-15k creative_writing"),
        "factual_lookup": rows([r["instruction"] for r in dolly if r["category"] == "open_qa"],
                               "factual_lookup", "dolly-15k open_qa"),
        "data_analysis": rows(sql, "data_analysis", "sql-create-context"),
        "chitchat": rows(chat, "chitchat", "synthetic-persona-chat"),
    }, size, seed)


# ---------------------------------------------------------------- search intent

def _load_search_intent(size: int, seed: int) -> List[Row]:
    """Every row of data/test_db.json. The file is the whole dataset, so it is not sampled."""
    if not TEST_DB_FILE.exists():
        raise ValueError(f"{TEST_DB_FILE} not found")
    rows = []
    with open(TEST_DB_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            intent = data.get("search_intent_info") or {}
            if not data.get("keyword") or not intent.get("main_intent"):
                continue
            rows.append({
                "text": data["keyword"],
                "label": intent["main_intent"],
                "secondary_labels": intent.get("foreign_intent") or None,
                "language": (data.get("extra") or {}).get("detected_language"),
            })
    return _clean(rows)


LOADERS: Dict[str, Callable[[int, int], List[Row]]] = {
    "support_routing": _load_support_routing,
    "prompt_injection": _load_prompt_injection,
    "question_type": _load_question_type,
    "news_topic": _load_news_topic,
    "email_triage": _load_email_triage,
    "toxicity": _load_toxicity,
    "request_domain": _load_request_domain,
}


def importable(task: Task) -> bool:
    if task.id == "search_intent":
        return TEST_DB_FILE.exists()
    return task.id in LOADERS


def load_rows(task: Task, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED, refresh: bool = False) -> List[Row]:
    """The task's dataset sample: from its snapshot when that holds enough rows, otherwise downloaded."""
    if task.id == "search_intent":
        return _load_search_intent(rows, seed)
    if task.id not in LOADERS:
        raise ValueError(f"The example '{task.name}' has no dataset to import")

    snapshot = SNAPSHOT_DIR / f"{task.id}.jsonl"
    if snapshot.exists() and not refresh:
        with open(snapshot, encoding="utf-8") as f:
            saved = [json.loads(line) for line in f if line.strip()]
        # The snapshot is in sample order, so its first rows are themselves a random sample.
        if len(saved) >= rows:
            return saved[:rows]

    loaded = LOADERS[task.id](rows, seed)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = snapshot.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in loaded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(snapshot)
    return loaded


def import_task(
    db: Session, task: Task, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED, refresh: bool = False
) -> Dict[str, Any]:
    """Add the task's dataset to its golden rows.

    Inputs that are already golden rows are left as they are, so importing again never
    overwrites a label that was corrected by hand.
    """
    loaded = load_rows(task, rows, seed, refresh)
    unknown = sorted({row["label"] for row in loaded} - set(task.labels))
    if unknown:
        raise ValueError(f"The dataset has labels the example does not: {unknown}")

    existing = {key for (key,) in db.query(GoldenRow.input_key).filter(GoldenRow.task_id == task.id)}
    created = 0
    for row in loaded:
        key = input_key(row["text"])
        if key in existing:
            continue
        existing.add(key)
        db.add(GoldenRow(
            id=str(uuid.uuid4()),
            task_id=task.id,
            input_text=row["text"],
            input_key=key,
            label=row["label"],
            secondary_labels=row.get("secondary_labels") or None,
            language=row.get("language"),
            source="imported",
            meta=row.get("meta"),
        ))
        created += 1
    db.commit()
    return {
        "task": task.id,
        "rows_in_dataset": len(loaded),
        "created": created,
        "already_present": len(loaded) - created,
        "total_golden_rows": db.query(GoldenRow).filter(GoldenRow.task_id == task.id).count(),
    }
