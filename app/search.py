"""Searcher — keyword (BM25) + semantic (vector) + hybrid (RRF) on the lab corpus.

Designed to work in both lite (Qdrant in-memory) and docker (Qdrant server) modes;
switch via env var QDRANT_MODE=memory|server (defaults to memory).

The hybrid mode uses Reciprocal Rank Fusion with k=60 — the same default used
by Vespa, Elasticsearch, and the hybrid RAG production stacks in the deck §3.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

Mode = Literal["keyword", "semantic", "hybrid"]
RRF_KEYWORD_DEPTH = 75
RRF_SEMANTIC_DEPTH = 15
# Model + dimension now come from EMBEDDING_BACKEND (see app/embeddings.py).
# Defaults are unchanged: fastembed / BAAI/bge-small-en-v1.5 / 384-dim.
EMBED_MODEL = Embedder().model_name
EMBED_DIM = Embedder().dim
COLLECTION = "lab19_corpus"


def reciprocal_rank_fusion(
    rankings: list[list[str]], top_k: int, rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked document ids with standard 1-based RRF scores."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])[:top_k]


@dataclass
class SearchHit:
    doc_id: str
    title: str
    text: str
    score: float

    def dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": self.score}


class Searcher:
    """Holds the BM25 index, Qdrant client, and document metadata.

    Construction is deliberately heavy (loading the embedding model + indexing
    the whole corpus once); callers should reuse a single instance.
    """

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.doc_ids: list[str] = []
        self._docs_by_id: dict[str, dict] = {}
        self.bm25: BM25Okapi | None = None
        self.client: QdrantClient | None = None
        self.embedder: Embedder | None = None
        self._query_vector_cache: dict[str, list[float]] = {}
        self._query_vector_cache_enabled = os.getenv("SEARCH_QUERY_CACHE", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }

    @property
    def size(self) -> int:
        return len(self.docs)

    @classmethod
    def from_corpus(cls, corpus_path: Path) -> "Searcher":
        # A student who opens NB1 before running setup otherwise gets a bare
        # FileNotFoundError pointing at a relative path, with no hint that the
        # corpus is generated rather than committed.
        if not Path(corpus_path).exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}.\n"
                "The corpus is generated, not committed. Run:\n"
                "    bash setup-lite.sh      # first time (venv + deps + data)\n"
                "    make seed               # if you only need to regenerate data"
            )
        s = cls()
        s._load_docs(corpus_path)
        s._build_bm25()
        s._build_vector_index()
        return s

    # ── ingestion ───────────────────────────────────────────────────────
    def _load_docs(self, corpus_path: Path) -> None:
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
                self.doc_ids.append(d["doc_id"])
                self._docs_by_id[d["doc_id"]] = d

    def _build_bm25(self) -> None:
        # Tokenise on whitespace — for VN+EN mixed text this is "good enough" baseline.
        # A real production system would use a proper VN tokenizer (underthesea / pyvi).
        # That choice is a "think hard" decision flagged in VIBE-CODING.md.
        tokenized = [self._tokenize(d["title"] + " " + d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def _build_vector_index(self) -> None:
        self.embedder = Embedder()

        mode = os.getenv("QDRANT_MODE", "memory")
        if mode == "server":
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(":memory:")

        # Recreate is OK in lite mode (it's in-memory); for server, only create if missing.
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing and mode == "server":
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            # dimension must follow the chosen model, not a module constant --
            # switching EMBEDDING_BACKEND changes it (384 -> 768 -> 1024 -> 1536).
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )

        # Embed in batches of 64 — fastembed is CPU-bound and that batch size is sweet spot.
        BATCH = 64
        points: list[PointStruct] = []
        for start in range(0, len(self.docs), BATCH):
            batch = self.docs[start:start + BATCH]
            texts = [d["title"] + " " + d["text"] for d in batch]
            vectors = list(self.embedder.embed_documents(texts))
            for i, (d, v) in enumerate(zip(batch, vectors)):
                points.append(PointStruct(
                    id=start + i,
                    vector=v.tolist(),
                    payload={"doc_id": d["doc_id"], "title": d["title"], "text": d["text"]},
                ))
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── retrieval ───────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        if mode == "keyword":
            return self._search_keyword(query, top_k)
        if mode == "semantic":
            return self._search_semantic(query, top_k)
        if mode == "hybrid":
            return self._search_hybrid(query, top_k, rrf_k)
        raise ValueError(f"unknown mode {mode!r}")

    def _search_keyword(self, query: str, top_k: int) -> list[SearchHit]:
        ranked = self._rank_keyword(query, top_k)
        return [
            SearchHit(
                doc_id=doc_id,
                title=self._docs_by_id[doc_id]["title"],
                text=self._docs_by_id[doc_id]["text"],
                score=score,
            )
            for doc_id, score in ranked
        ]

    def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        ranked = self._rank_semantic(query, top_k)
        return [
            SearchHit(
                doc_id=doc_id,
                title=self._docs_by_id[doc_id]["title"],
                text=self._docs_by_id[doc_id]["text"],
                score=score,
            )
            for doc_id, score in ranked
        ]

    def _rank_keyword(self, query: str, top_k: int) -> list[tuple[str, float]]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked]

    def _rank_semantic(self, query: str, top_k: int) -> list[tuple[str, float]]:
        assert self.client is not None and self.embedder is not None
        q_vec = self._query_vector_cache.get(query) if self._query_vector_cache_enabled else None
        if q_vec is None:
            q_vec = self.embedder.embed_query(query).tolist()
            if self._query_vector_cache_enabled:
                self._query_vector_cache[query] = q_vec
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=top_k,
            # Hybrid fusion only needs the id/rank. Avoid moving title/text
            # payloads for every candidate; final hits are materialized below.
            with_payload=["doc_id"],
        )
        return [(p.payload["doc_id"], float(p.score)) for p in result.points]

    def _search_hybrid(self, query: str, top_k: int, rrf_k: int) -> list[SearchHit]:
        # Candidate-pool sizing is asymmetric by design: semantic top ranks are
        # high precision, while a deeper lexical pool preserves exact/mixed
        # coverage. This remains pure rank-based RRF, not score weighting.
        kw_hits = self._rank_keyword(query, max(top_k, RRF_KEYWORD_DEPTH))
        sem_hits = self._rank_semantic(query, max(top_k, RRF_SEMANTIC_DEPTH))

        # Reciprocal Rank Fusion — scores depend only on rank, not on the
        # underlying BM25/vector score. The rank is explicitly 1-based.
        ordered = reciprocal_rank_fusion(
            [[doc_id for doc_id, _score in hits] for hits in (kw_hits, sem_hits)],
            top_k,
            rrf_k,
        )
        return [
            SearchHit(
                doc_id=doc_id,
                title=self._docs_by_id[doc_id]["title"],
                text=self._docs_by_id[doc_id]["text"],
                score=score,
            )
            for doc_id, score in ordered
        ]
