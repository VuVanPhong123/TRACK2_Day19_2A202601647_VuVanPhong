from types import SimpleNamespace

import numpy as np

from app.embeddings import Embedder
from app.search import Searcher, reciprocal_rank_fusion


def test_e5_query_and_document_conventions_are_explicit(monkeypatch):
    embedder = Embedder("multilingual")
    seen: list[str] = []

    def fake_embed(texts):
        items = list(texts)
        seen.extend(items)
        return iter([np.zeros(embedder.dim, dtype=np.float32) for _ in items])

    monkeypatch.setattr(embedder, "embed", fake_embed)
    list(embedder.embed_documents(["document"]))
    list(embedder.embed_queries(["query"]))

    assert seen == ["passage: document", "query: query"]


def test_query_cache_disable_reembeds_each_query(monkeypatch):
    monkeypatch.setenv("SEARCH_QUERY_CACHE", "0")
    searcher = Searcher()
    searcher.client = SimpleNamespace(
        query_points=lambda **kwargs: SimpleNamespace(
            points=[SimpleNamespace(payload={"doc_id": "d1"}, score=1.0)]
        )
    )
    calls = []
    searcher.embedder = SimpleNamespace(
        embed_query=lambda query: calls.append(query) or np.zeros(3, dtype=np.float32)
    )

    searcher._rank_semantic("same", 1)
    searcher._rank_semantic("same", 1)

    assert calls == ["same", "same"]
    assert searcher._query_vector_cache == {}


def test_rrf_uses_one_based_rank_and_literal_formula():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], top_k=3, rrf_k=60)

    assert fused[0][0] == "a"
    assert fused[0][1] == 1 / (60 + 1) + 1 / (60 + 1)
    assert fused[1] == ("b", 1 / (60 + 2))
    assert fused[2] == ("c", 1 / (60 + 2))


def test_rank_only_hybrid_materializes_same_public_hit_shape():
    searcher = Searcher()
    searcher.docs = [
        {"doc_id": "d1", "title": "Title 1", "text": "Text 1"},
        {"doc_id": "d2", "title": "Title 2", "text": "Text 2"},
    ]
    searcher.doc_ids = ["d1", "d2"]
    searcher._docs_by_id = {d["doc_id"]: d for d in searcher.docs}
    searcher._rank_keyword = lambda query, top_k: [("d1", 10.0), ("d2", 1.0)]
    searcher._rank_semantic = lambda query, top_k: [("d2", 0.9), ("d1", 0.8)]

    hits = searcher._search_hybrid("q", top_k=2, rrf_k=60)

    assert [h.doc_id for h in hits] == ["d1", "d2"]
    assert hits[0].title == "Title 1"
    assert hits[0].text == "Text 1"
    assert hits[0].score == 1 / 61 + 1 / 62
