"""Search over the chunks: by meaning (vectors), by words (BM25), and the two combined."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.embed import Embedder


def tokenize(text: str) -> list[str]:
    # Keeps codes like "of-450h" and "e-26" as single tokens.
    return re.findall(r"[\w\-]+", text.lower())


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks], "passage")
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def vector_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks closest in meaning to the query, best first."""
        q = self.embedder.embed([query], "query")[0]
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:k]]

    def bm25_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks with the best word match, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion.

    `rankings` is a list of ranked lists of chunk indexes, best first.
    A chunk gets 1 / (k + rank) from every list it appears in (rank starts at 1),
    and the scores are summed. Return all chunk indexes, sorted by score, best first.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def is_current(chunk: Chunk) -> bool:
    """False for chunks from documents that were replaced by a newer version.

    Look at the metadata of the documents in data/ and decide which field tells you that.
    """
    return chunk.meta.get("status") != "superseded"


def mentioned_models(index: Index, query: str) -> list[str]:
    """Doc ids of the model documents the query names. Title words that most models share (like the brand) are ignored."""
    titles = {c.doc_id: set(tokenize(c.meta["title"])) for c in index.chunks if c.meta.get("type") == "model"}
    common = {t for ts in titles.values() for t in ts if sum(t in o for o in titles.values()) > len(titles) / 2}
    words = set(tokenize(query))
    return [doc_id for doc_id, ts in titles.items() if (ts - common) & words]


def search(index: Index, query: str, mode: str = "vector", top_k: int = 5,
           current_only: bool = False, decompose: bool = True) -> list[Chunk]:
    if mode == "vector":
        ranked = index.vector_search(query)
    elif mode == "hybrid":
        ranked = rrf([index.vector_search(query), index.bm25_search(query)])
    else:
        raise ValueError(f"unknown mode: {mode}")
    models = mentioned_models(index, query) if decompose else []
    if len(models) > 1:  # a multi-model question: add one ranking per model, so each model gets its best chunk in
        ranked = rrf([ranked] + [[i for i in ranked if index.chunks[i].doc_id == d] for d in models])
    if current_only:
        ranked = [i for i in ranked if is_current(index.chunks[i])]
    return [index.chunks[i] for i in ranked[:top_k]]
