"""Reliability evaluation before/after adaptation."""
from typing import List, Dict, Any
from src.knowledge_base import KnowledgeBase
from src.rag_llm import generate_answer


def evaluate_on_examples(
    kb: KnowledgeBase,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate RAG+LLM system on a list of labeled examples.
    Returns accuracy, average confidence, and per-example results.
    """
    correct = 0
    confidences = []
    results = []

    for ex in examples:
        retrieved = kb.retrieve(ex["text"])
        prediction = generate_answer(ex["text"], retrieved)
        pred_label = prediction.get("label", "unknown")
        conf = float(prediction.get("confidence", 0.0))
        confidences.append(conf)

        is_correct = pred_label == ex.get("label")
        if is_correct:
            correct += 1

        results.append({
            "id": ex.get("id"),
            "text": ex["text"][:100],
            "true_label": ex.get("label"),
            "pred_label": pred_label,
            "confidence": conf,
            "correct": is_correct,
        })

    n = len(examples) or 1
    accuracy = correct / n
    avg_confidence = sum(confidences) / n if confidences else 0.0

    # Simple reliability proxy: accuracy among high-confidence predictions
    high_conf = [r for r in results if r["confidence"] >= 0.7]
    high_conf_acc = (
        sum(1 for r in high_conf if r["correct"]) / len(high_conf)
        if high_conf else None
    )

    return {
        "n_examples": len(examples),
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_confidence, 4),
        "high_confidence_accuracy": round(high_conf_acc, 4) if high_conf_acc is not None else None,
        "n_high_confidence": len(high_conf),
        "results": results,
    }


def compare_before_after(
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare reliability metrics before and after adaptation."""
    return {
        "accuracy_before": metrics_before["accuracy"],
        "accuracy_after": metrics_after["accuracy"],
        "accuracy_delta": round(metrics_after["accuracy"] - metrics_before["accuracy"], 4),
        "confidence_before": metrics_before["avg_confidence"],
        "confidence_after": metrics_after["avg_confidence"],
        "confidence_delta": round(
            metrics_after["avg_confidence"] - metrics_before["avg_confidence"], 4
        ),
        "improved": metrics_after["accuracy"] >= metrics_before["accuracy"],
    }
