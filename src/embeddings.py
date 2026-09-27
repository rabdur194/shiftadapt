"""Embedding utilities using a pretrained foundation model."""
from typing import List
import numpy as np

_model = None


def get_model(model_name: str = None):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL
        name = model_name or EMBEDDING_MODEL
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns (n, dim) array."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def centroid(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)
