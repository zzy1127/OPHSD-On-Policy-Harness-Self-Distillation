"""Embedding model and online memory bank with filtered retrieval."""

import logging
import os
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Embedder
# ────────────────────────────────────────────────────────────────────
class Embedder:
    """Thin wrapper over sentence-transformers (primary) / TF-IDF (fallback).

    TF-IDF is fit incrementally: the first call to encode_query triggers a
    fit on whatever texts have been added so far via add_text().  Subsequent
    calls reuse that vocabulary.  Call refit() explicitly if needed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self.method: Optional[str] = None
        self._st_model = None
        self._tfidf_vec = None
        self._tfidf_svd = None
        self._fit_texts: list[str] = []   # corpus for TF-IDF fitting
        self._fitted = False

    def _init(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            self.method = "sentence_transformers"
            logger.info("Embedder: sentence-transformers (%s)", self.model_name)
        except (ImportError, Exception) as exc:
            logger.warning("sentence-transformers unavailable (%s), using TF-IDF fallback", exc)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._tfidf_vec = TfidfVectorizer(
                analyzer="char", ngram_range=(2, 4), max_features=50000, sublinear_tf=True
            )
            self._tfidf_svd = TruncatedSVD(n_components=256, random_state=42)
            self.method = "tfidf"
            logger.info("Embedder: TF-IDF char-ngrams + SVD(256)")

    def add_text(self, text: str):
        """Register a text into the TF-IDF corpus (no-op for sentence-transformers)."""
        if self.method == "tfidf" or self.method is None:
            self._fit_texts.append(text)
            self._fitted = False  # invalidate fit on new data

    def _ensure_fitted(self):
        if self.method == "tfidf" and not self._fitted and self._fit_texts:
            mat = self._tfidf_vec.fit_transform(self._fit_texts)
            self._tfidf_svd.fit(mat)
            self._fitted = True

    def encode_query(self, text: str) -> np.ndarray:
        if self.method is None:
            self._init()
        if self.method == "sentence_transformers":
            return self._st_model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
        self._ensure_fitted()
        if not self._fitted:
            # Not enough corpus yet — return a zero vector; memory bank will be empty anyway
            return np.zeros(256, dtype=np.float32)
        vec = self._tfidf_vec.transform([text])
        emb = self._tfidf_svd.transform(vec)[0].astype(np.float32)
        # SVD n_components is capped by corpus size early on; always pad to 256
        target_dim = self._tfidf_svd.n_components
        if len(emb) < target_dim:
            emb = np.pad(emb, (0, target_dim - len(emb)))
        return emb

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Batch encode (ST only is fast-path; TF-IDF falls back to per-text)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.method is None:
            self._init()
        if self.method == "sentence_transformers":
            bs = min(32, max(1, len(texts)))
            return self._st_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=bs,
            ).astype(np.float32)
        return np.stack([self.encode_query(t) for t in texts])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_name": self.model_name,
            "method": self.method,
            "fit_texts": self._fit_texts,
            "fitted": self._fitted,
        }
        if self.method == "tfidf":
            state["vectorizer"] = self._tfidf_vec
            state["svd"] = self._tfidf_svd
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "Embedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(model_name=state["model_name"])
        obj.method = state["method"]
        obj._fit_texts = state.get("fit_texts", [])
        obj._fitted = state.get("fitted", False)
        if state["method"] == "tfidf":
            obj._tfidf_vec = state["vectorizer"]
            obj._tfidf_svd = state["svd"]
        else:
            obj._init()
        return obj


# ────────────────────────────────────────────────────────────────────
# Memory bank  (online — grows one sample at a time)
# ────────────────────────────────────────────────────────────────────
class MemoryBank:
    """Online labelled-example store with cosine-similarity retrieval.

    Grows incrementally: call add() after each prediction with the sample's
    true label.  Retrieval uses exact cosine similarity over all stored vectors.
    """

    def __init__(self):
        self.examples: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None   # (N, D) normalised
        self._label_idx: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def add(self, example: dict, embedding: np.ndarray):
        """Append one labelled example and its embedding."""
        norm = float(np.linalg.norm(embedding))
        emb_normed = (embedding / max(norm, 1e-8)).astype(np.float32).reshape(1, -1)

        idx = len(self.examples)
        self.examples.append(example)
        self.embeddings = emb_normed if self.embeddings is None \
            else np.concatenate([self.embeddings, emb_normed], axis=0)

        for lab in example.get("accusations", []):
            self._label_idx.setdefault(lab, []).append(idx)

    def retrieve(
        self,
        query_emb: np.ndarray,
        k: int = 5,
        filter_labels: Optional[list[str]] = None,
        same_label: bool = True,
    ) -> list[dict]:
        """Return top-*k* examples by cosine similarity.

        filter_labels + same_label=True  → only examples sharing a label
        filter_labels + same_label=False → only examples NOT sharing a label
        """
        if not self.examples:
            return []

        q = query_emb / max(float(np.linalg.norm(query_emb)), 1e-8)
        sims: np.ndarray = self.embeddings @ q

        if filter_labels is not None:
            matching: set[int] = set()
            for lab in filter_labels:
                matching.update(self._label_idx.get(lab, []))
            mask = np.zeros(len(self.examples), dtype=bool)
            for idx in matching:
                mask[idx] = True
            if not same_label:
                mask = ~mask
            sims = np.where(mask, sims, -np.inf)

        top_ids = np.argsort(sims)[::-1][:k]
        return [
            {**self.examples[i], "similarity": float(sims[i])}
            for i in top_ids if sims[i] > -np.inf
        ]

    def save(self, path: str):
        """Snapshot current state to disk (for checkpoint/resume)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "examples": self.examples,
                "embeddings": self.embeddings,
                "label_idx": self._label_idx,
            }, f)

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        with open(path, "rb") as f:
            data = pickle.load(f)
        bank = cls()
        bank.examples = data["examples"]
        bank.embeddings = data["embeddings"]
        bank._label_idx = data["label_idx"]
        return bank
