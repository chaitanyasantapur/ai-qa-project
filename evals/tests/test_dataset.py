import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evals" / "datasets" / "golden.json"


@pytest.fixture(scope="module")
def items():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_dataset_has_enough_questions(items):
    assert len(items) >= 15


def test_every_item_has_a_question(items):
    for item in items:
        assert isinstance(item.get("question"), str)
        assert item["question"].strip()


def test_no_duplicate_questions(items):
    questions = [i["question"].strip().lower() for i in items]
    assert len(questions) == len(set(questions))


def test_answerable_items_have_keywords(items):
    for item in items:
        if not item.get("should_refuse"):
            assert item.get("keywords"), f"No keywords: {item['question']}"


def test_refusal_items_have_no_keywords(items):
    for item in items:
        if item.get("should_refuse"):
            assert item.get("keywords") == [], f"Remove keywords: {item['question']}"


def test_dataset_covers_both_kinds(items):
    refusals = [i for i in items if i.get("should_refuse")]
    answerable = [i for i in items if not i.get("should_refuse")]
    assert len(refusals) >= 3
    assert len(answerable) >= 10


def test_dataset_includes_prompt_injection_case(items):
    assert any("ignore" in i["question"].lower() for i in items)