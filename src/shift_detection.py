"""Distribution shift detection using embedding space statistics."""
from typing import List, Dict, Any
import numpy as np
from src.embeddings import embed_texts, centroid, cosine_distance
from src.config import SHIFT_THRESHOLD


def compute_domain_centroid(texts: List[str]) -> np.ndarray:
    embs = embed_texts(texts)
    return centroid(embs)


def detect_shift(
    reference_texts: List[str],
    new_texts: List[str],
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Detect distribution shift between reference (old) and new data.

    Method: cosine distance between domain centroids in embedding space.
    Simple, interpretable, and data-efficient.
    """
    threshold = threshold if threshold is not None else SHIFT_THRESHOLD

    ref_centroid = compute_domain_centroid(reference_texts)
    new_centroid = compute_domain_centroid(new_texts)
    distance = cosine_distance(ref_centroid, new_centroid)

    shifted = distance >= threshold

    # Per-example distances to reference centroid (for inspection)
    new_embs = embed_texts(new_texts)
    per_example = [cosine_distance(e, ref_centroid) for e in new_embs]

    return {
        "shifted": shifted,
        "centroid_distance": round(distance, 4),
        "threshold": threshold,
        "mean_example_distance": round(float(np.mean(per_example)), 4),
        "max_example_distance": round(float(np.max(per_example)), 4),
        "n_reference": len(reference_texts),
        "n_new": len(new_texts),
        "interpretation": (
            f"Shift DETECTED (distance {distance:.3f} >= {threshold})"
            if shifted
            else f"No significant shift (distance {distance:.3f} < {threshold})"
        ),
    }
