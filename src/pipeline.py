"""End-to-end ShiftAdapt pipeline."""
import json
from pathlib import Path
from typing import Dict, Any, List

from src.config import OLD_DOMAIN_PATH, NEW_DOMAIN_PATH, ADAPT_PATH
from src.knowledge_base import KnowledgeBase
from src.shift_detection import detect_shift
from src.adapt import adapt_with_examples
from src.evaluate import evaluate_on_examples, compare_before_after
from src.rag_llm import generate_answer


def load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def run_full_pipeline() -> Dict[str, Any]:
    """
    Full Level-A pipeline:
    1. Build KB from old domain
    2. Detect shift on new domain
    3. Evaluate BEFORE adaptation
    4. Adapt with few new examples
    5. Evaluate AFTER adaptation
    6. Compare reliability
    """
    old_data = load_json(OLD_DOMAIN_PATH)
    new_data = load_json(NEW_DOMAIN_PATH)
    adapt_data = load_json(ADAPT_PATH)

    # 1. Base knowledge base (old domain only)
    kb = KnowledgeBase()
    kb.add_documents(old_data)

    # 2. Shift detection
    old_texts = [d["text"] for d in old_data]
    new_texts = [d["text"] for d in new_data]
    shift_report = detect_shift(old_texts, new_texts)

    # 3. Reliability BEFORE adaptation (on new domain)
    before = evaluate_on_examples(kb, new_data)

    # 4. Data-efficient adaptation
    adapt_report = adapt_with_examples(kb, adapt_data)

    # 5. Reliability AFTER adaptation
    after = evaluate_on_examples(kb, new_data)

    # 6. Comparison
    comparison = compare_before_after(before, after)

    # Persist updated KB
    kb.save()

    return {
        "shift_detection": shift_report,
        "adaptation": adapt_report,
        "metrics_before": {
            "accuracy": before["accuracy"],
            "avg_confidence": before["avg_confidence"],
            "high_confidence_accuracy": before["high_confidence_accuracy"],
            "n_examples": before["n_examples"],
        },
        "metrics_after": {
            "accuracy": after["accuracy"],
            "avg_confidence": after["avg_confidence"],
            "high_confidence_accuracy": after["high_confidence_accuracy"],
            "n_examples": after["n_examples"],
        },
        "comparison": comparison,
        "kb_size_final": len(kb),
    }


def predict_single(text: str, use_adapted_kb: bool = True) -> Dict[str, Any]:
    """Run a single prediction using the knowledge base."""
    kb = KnowledgeBase()
    if use_adapted_kb:
        try:
            kb.load()
        except Exception:
            pass
    if len(kb) == 0:
        # fallback to old domain
        old_data = load_json(OLD_DOMAIN_PATH)
        kb.add_documents(old_data)

    retrieved = kb.retrieve(text)
    answer = generate_answer(text, retrieved)
    return {
        "query": text,
        "retrieved": retrieved,
        "answer": answer,
    }
