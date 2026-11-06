"""LEVEL 1/2 - Dense (FAISS) + sparse (BM25) indexes, persisted to disk."""
import json
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

import config


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class Index:
    def __init__(self):
        self.chunks, self.faiss, self.bm25, self._embedder = [], None, None, None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    def embed(self, texts):
        return self.embedder.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).astype("float32")

    def build(self, chunks):
        self.chunks = chunks
        vecs = self.embed([c["text"] for c in chunks])
        self.faiss = faiss.IndexFlatIP(vecs.shape[1])   # cosine (vectors are normalized)
        self.faiss.add(vecs)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def save(self):
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(config.INDEX_DIR / "dense.faiss"))
        (config.INDEX_DIR / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    def load(self):
        f, c = config.INDEX_DIR / "dense.faiss", config.INDEX_DIR / "chunks.json"
        if not (f.exists() and c.exists()):
            return False
        self.faiss = faiss.read_index(str(f))
        self.chunks = json.loads(c.read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(x["text"]) for x in self.chunks])
        return True

    def dense(self, query, k):
        scores, ids = self.faiss.search(self.embed([query]), min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def sparse(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
