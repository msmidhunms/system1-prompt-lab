import pytest

from app import golden
from app.database import GoldenRow
from app.tasks.registry import get_task

INTENT = get_task("search_intent")
SUPPORT = get_task("support_routing")


def test_input_key_ignores_case_and_spacing():
    assert golden.input_key("Best  Laptop ") == golden.input_key("best laptop")
    assert golden.input_key("best\tlaptop\n") == golden.input_key("best laptop")
    assert golden.input_key("best laptop") != golden.input_key("best laptops")


def test_upsert_creates_then_updates_the_same_row(db):
    row, created, previous = golden.upsert_row(db, SUPPORT, "Where is my card?", "card")
    db.commit()
    assert created and previous is None

    same, created, previous = golden.upsert_row(db, SUPPORT, "  where IS   my card? ", "account")
    db.commit()
    assert not created and previous == "card"
    assert same.id == row.id and same.label == "account"
    # The text is kept as it was first entered.
    assert same.input_text == "Where is my card?"
    assert db.query(GoldenRow).count() == 1


def test_same_input_in_another_example_is_a_separate_row(db):
    golden.upsert_row(db, SUPPORT, "hello", "card")
    _, created, _ = golden.upsert_row(db, INTENT, "hello", "informational", language="en")
    db.commit()
    assert created
    assert db.query(GoldenRow).count() == 2


def test_upsert_rejects_unknown_label_and_empty_text(db):
    with pytest.raises(ValueError):
        golden.upsert_row(db, SUPPORT, "hello", "informational")
    with pytest.raises(ValueError):
        golden.upsert_row(db, SUPPORT, "   ", "card")


def test_new_search_intent_row_gets_a_language(db):
    row, _, _ = golden.upsert_row(db, INTENT, "best laptop 2024", "commercial")
    db.commit()
    assert row.language == "en"
    assert golden.examples_for(db, INTENT) == [{"text": "best laptop 2024", "label": "commercial"}]


def test_examples_leave_out_other_languages(db):
    golden.upsert_row(db, INTENT, "mejor portátil", "commercial", language="es")
    golden.upsert_row(db, INTENT, "best laptop", "commercial", language="en")
    db.commit()
    assert [e["text"] for e in golden.examples_for(db, INTENT)] == ["best laptop"]
    assert golden.stats(db, INTENT)["excluded_other_language"] == 1


def test_update_row_refuses_to_collide_with_another_row(db):
    golden.upsert_row(db, SUPPORT, "first", "card")
    second, _, _ = golden.upsert_row(db, SUPPORT, "second", "card")
    db.commit()
    with pytest.raises(LookupError):
        golden.update_row(db, SUPPORT, second, text="FIRST")
    golden.update_row(db, SUPPORT, second, text="Second ", label="transfer")
    assert second.input_text == "Second" and second.label == "transfer"


def test_list_rows_search_and_filters(db):
    golden.upsert_row(db, SUPPORT, "my card is lost", "card")
    golden.upsert_row(db, SUPPORT, "transfer 100% failed", "transfer")
    db.commit()
    assert golden.list_rows(db, SUPPORT, q="CARD")["pagination"]["total_items"] == 1
    assert golden.list_rows(db, SUPPORT, q="100%")["pagination"]["total_items"] == 1
    assert golden.list_rows(db, SUPPORT, label="transfer")["items"][0]["text"] == "transfer 100% failed"
    assert golden.stats(db, SUPPORT)["by_label"]["card"] == 1
