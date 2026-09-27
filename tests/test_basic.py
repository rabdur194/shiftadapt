"""Basic tests for ShiftAdapt."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_data_files_exist():
    assert (ROOT / "data/old_domain/examples.json").exists()
    assert (ROOT / "data/new_domain/examples.json").exists()
    assert (ROOT / "data/adaptation_examples/examples.json").exists()


def test_load_data():
    with open(ROOT / "data/old_domain/examples.json") as f:
        old = json.load(f)
    assert len(old) >= 5
    assert "text" in old[0]
    assert "label" in old[0]


def test_mock_llm_and_kb():
    from src.knowledge_base import KnowledgeBase
    from src.rag_llm import generate_answer

    kb = KnowledgeBase()
    kb.add_documents([
        {"text": "Pay fee to claim prize now", "label": "scam"},
        {"text": "Your OTP is 123456. Do not share.", "label": "safe"},
    ])
    retrieved = kb.retrieve("You won a prize, pay shipping fee")
    assert len(retrieved) >= 1
    ans = generate_answer("You won a prize, pay shipping fee", retrieved)
    assert "label" in ans
    assert "confidence" in ans


def test_shift_detection_runs():
    from src.shift_detection import detect_shift

    old = ["Normal bank debit notification for your account"]
    new = ["Send crypto to this wallet immediately or lose access forever"]
    report = detect_shift(old, new, threshold=0.01)
    assert "shifted" in report
    assert "centroid_distance" in report
