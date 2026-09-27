"""Data-efficient adaptation by updating the knowledge base."""
from typing import List, Dict, Any
from src.knowledge_base import KnowledgeBase


def adapt_with_examples(
    kb: KnowledgeBase,
    new_examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Data-efficient adaptation: add a small number of new labeled examples
    to the knowledge base and rebuild embeddings.

    This is the Level-A adaptation strategy:
    - No full model retraining
    - No LoRA
    - Only a few new examples
    """
    n_before = len(kb)
    kb.add_documents(new_examples, rebuild=True)
    n_after = len(kb)

    return {
        "method": "knowledge_base_update",
        "examples_added": len(new_examples),
        "kb_size_before": n_before,
        "kb_size_after": n_after,
        "message": (
            f"Adapted by adding {len(new_examples)} examples. "
            f"Knowledge base grew from {n_before} → {n_after} documents."
        ),
    }
