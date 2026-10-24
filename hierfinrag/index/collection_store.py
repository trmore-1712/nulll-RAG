"""
Named Collection Store with FAISS vector index
================================================
Manages multiple named document collections (one per /index call).
Each collection contains:
  - Parsed Document objects (in-memory cache)
  - A FAISS flat-IP index over all chunk embeddings
  - Per-chunk metadata (doc name, page number, section, text)
  - DocumentMetadata registry (for the MetadataRouter)

Falls back to numpy dot-product search if FAISS is not installed.
"""

import os
import json
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from ..parsing.base import Document
from ..routing.metadata_router import MetadataRouter, DocumentMetadata


class ChunkRecord:
    """One retrievable chunk stored in the index."""
    __slots__ = ("chunk_id", "doc_name", "text", "metadata", "embedding")

    def __init__(
        self,
        chunk_id: str,
        doc_name: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
    ):
        self.chunk_id  = chunk_id
        self.doc_name  = doc_name
        self.text      = text
        self.metadata  = metadata
        self.embedding = embedding


class CollectionStore:
    """
    One store per named collection.  Keeps an in-memory FAISS/numpy index
    for fast ANN retrieval plus the full parsed Document objects for
    section-level aggregation.
    """

    def __init__(self, collection_name: str, embedding_dim: int = 384):
        self.name           = collection_name
        self.embedding_dim  = embedding_dim
        self.chunks: List[ChunkRecord] = []
        self.docs: Dict[str, Document] = {}           # doc_name -> Document
        self.router         = MetadataRouter()
        self._index         = None                    # FAISS index (lazy)
        self._bm25          = None                    # BM25Okapi (lazy)
        self._index_dirty   = True
        self.created_at     = time.time()
        self.indexed_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_document(self, doc: Document, sample_text: str = "") -> None:
        """Register a parsed Document and extract its metadata."""
        self.docs[doc.title] = doc
        self.router.register(doc.title, sample_text)

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Bulk-add pre-embedded chunk records."""
        self.chunks.extend(chunks)
        self._index_dirty = True

    # ------------------------------------------------------------------
    # FAISS / numpy index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        if not self.chunks:
            return
        embeddings = np.stack([c.embedding for c in self.chunks if c.embedding is not None])
        # L2-normalise so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        try:
            import faiss  # type: ignore
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(embeddings.astype("float32"))
        except ImportError:
            # Fallback: store the normalised matrix for numpy dot-product search
            self._index = embeddings  # shape (N, D)

        # Also build BM25 at index time (fast, no embeddings needed)
        self._build_bm25()

        self._index_dirty = False
        self.indexed_at   = time.time()

    def _build_bm25(self) -> None:
        """Build a global BM25 lexical index over all chunk texts.
        Built once at index time; captures exact financial terms (NPA, SARFAESI,
        NCLT, specific rupee figures) that semantic embeddings miss.
        """
        try:
            from rank_bm25 import BM25Okapi
            corpus = [
                re.findall(r"\w+", c.text.lower()) for c in self.chunks
            ]
            corpus = [t if t else ["empty"] for t in corpus]
            self._bm25 = BM25Okapi(corpus)
            print(f"  [CollectionStore] BM25 index built over {len(self.chunks)} chunks.")
        except Exception as e:
            print(f"  [CollectionStore] BM25 build failed (continuing without): {e}")
            self._bm25 = None

    def bm25_search(
        self,
        query: str,
        top_k: int = 50,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Tuple[float, ChunkRecord]]:
        """Lexical BM25 search — catches exact financial terms vectors miss."""
        if self._bm25 is None:
            return []
        tokens = re.findall(r"\w+", query.lower()) or ["empty"]
        raw_scores = self._bm25.get_scores(tokens)
        # Sort by score descending
        ranked = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)
        doc_set = set(doc_filter) if doc_filter else None
        results: List[Tuple[float, ChunkRecord]] = []
        for i in ranked:
            if raw_scores[i] <= 0:
                break
            cr = self.chunks[i]
            if doc_set and cr.doc_name not in doc_set:
                continue
            results.append((float(raw_scores[i]), cr))
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_str: str,
        top_k: int = 20,
        doc_filter: Optional[List[str]] = None,
        rrf_k: int = 60,
    ) -> List[Tuple[float, ChunkRecord]]:
        """
        Hybrid retrieval: Reciprocal Rank Fusion of FAISS (semantic) + BM25 (lexical).

        RRF score = 1/(rrf_k + dense_rank) + 1/(rrf_k + bm25_rank)

        This captures both:
          - Semantically similar chunks (FAISS)
          - Exact financial terms: NPA, SARFAESI, Rs 4,053.90 cr, DRT (BM25)
        """
        # Lazy BM25 build: handles collections indexed before BM25 was added.
        # One-time cost (~2-5s for 35 docs), then cached in-memory for all future queries.
        if self._bm25 is None and self.chunks:
            print(f"  [CollectionStore] Building BM25 index lazily for '{self.name}'...")
            self._build_bm25()

        fetch_k = top_k * 4  # over-fetch, then fuse

        # Dense FAISS
        dense_hits = self.search(query_embedding, top_k=fetch_k, doc_filter=doc_filter)
        dense_rank_map: Dict[str, int] = {cr.chunk_id: rank for rank, (_, cr) in enumerate(dense_hits)}

        # Sparse BM25
        bm25_hits = self.bm25_search(query_str, top_k=fetch_k, doc_filter=doc_filter)
        bm25_rank_map: Dict[str, int] = {cr.chunk_id: rank for rank, (_, cr) in enumerate(bm25_hits)}

        # Build unified chunk map
        cr_map: Dict[str, ChunkRecord] = {cr.chunk_id: cr for _, cr in dense_hits}
        cr_map.update({cr.chunk_id: cr for _, cr in bm25_hits})

        n_dense = max(len(dense_hits), 1)
        n_bm25  = max(len(bm25_hits), 1)

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        for cid in cr_map:
            r_dense = dense_rank_map.get(cid, n_dense + rrf_k)
            r_bm25  = bm25_rank_map.get(cid, n_bm25  + rrf_k)
            rrf_scores[cid] = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_bm25))

        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [(rrf_scores[cid], cr_map[cid]) for cid in top_ids]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Tuple[float, ChunkRecord]]:
        """
        Returns (score, ChunkRecord) pairs, highest score first.
        doc_filter: if provided, restrict results to chunks from these docs.
        """
        if self._index_dirty:
            self._build_index()
        if self._index is None:
            return []

        # Normalise query
        q = query_embedding.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-9)

        # Determine search depth (wider when filtering by document)
        search_depth = min(max(top_k * 50, 500), len(self.chunks)) if doc_filter else min(top_k * 5, len(self.chunks))

        try:
            import faiss  # type: ignore
            q_2d = q.reshape(1, -1)
            scores, indices = self._index.search(q_2d, search_depth)
            candidates = [(float(scores[0][i]), self.chunks[indices[0][i]])
                          for i in range(len(indices[0])) if indices[0][i] >= 0]
        except (ImportError, Exception):
            # numpy fallback
            mat = self._index  # (N, D)
            sims = mat @ q
            top_indices = np.argsort(sims)[::-1][:search_depth]
            candidates = [(float(sims[i]), self.chunks[i]) for i in top_indices]

        # Apply document filter
        if doc_filter:
            doc_set = set(doc_filter)
            filtered_candidates = [(s, c) for s, c in candidates if c.doc_name in doc_set]
            if filtered_candidates:
                candidates = filtered_candidates

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.name,
            "num_documents":   len(self.docs),
            "num_chunks":      len(self.chunks),
            "created_at":      self.created_at,
            "indexed_at":      self.indexed_at,
        }


class CollectionRegistry:
    """
    Global registry of named CollectionStores.
    Thread-safe enough for single-worker FastAPI / uvicorn use.
    """

    def __init__(self):
        self._collections: Dict[str, CollectionStore] = {}

    def create(self, name: str, embedding_dim: int = 384) -> CollectionStore:
        store = CollectionStore(name, embedding_dim=embedding_dim)
        self._collections[name] = store
        return store

    def get(self, name: str) -> Optional[CollectionStore]:
        return self._collections.get(name)

    def get_or_create(self, name: str, embedding_dim: int = 384) -> CollectionStore:
        if name not in self._collections:
            return self.create(name, embedding_dim)
        return self._collections[name]

    def list_collections(self) -> List[str]:
        return list(self._collections.keys())

    def delete(self, name: str) -> bool:
        if name in self._collections:
            del self._collections[name]
            return True
        return False


# Singleton registry used by server.py
registry = CollectionRegistry()
