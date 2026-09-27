"""Simple in-memory + disk-persistable knowledge base for RAG."""
from typing import List, Dict, Any, Optional
import json
import numpy as np
from pathlib import Path
from src.embeddings import embed_texts, cosine_distance
from src.config import KB_PATH, TOP_K


class KnowledgeBase:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[Dict[str, Any]], rebuild: bool = True):
        """Add documents. Each doc needs at least 'text' and preferably 'label'."""
        self.documents.extend(docs)
        if rebuild:
            self._rebuild_embeddings()

    def _rebuild_embeddings(self):
        if not self.documents:
            self.embeddings = None
            return
        texts = [d["text"] for d in self.documents]
        self.embeddings = embed_texts(texts)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        top_k = top_k or TOP_K
        if not self.documents or self.embeddings is None:
            return []

        from src.embeddings import embed_text
        q = embed_text(query)
        distances = [cosine_distance(q, e) for e in self.embeddings]
        indices = np.argsort(distances)[:top_k]

        results = []
        for i in indices:
            doc = self.documents[int(i)].copy()
            doc["distance"] = round(float(distances[int(i)]), 4)
            doc["score"] = round(1.0 - float(distances[int(i)]), 4)
            results.append(doc)
        return results

    def save(self, path: Path = None):
        path = path or KB_PATH
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f, indent=2)
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)

    def load(self, path: Path = None):
        path = path or KB_PATH
        doc_file = path / "documents.json"
        emb_file = path / "embeddings.npy"
        if doc_file.exists():
            with open(doc_file) as f:
                self.documents = json.load(f)
        if emb_file.exists():
            self.embeddings = np.load(emb_file)
        elif self.documents:
            self._rebuild_embeddings()

    def __len__(self):
        return len(self.documents)
