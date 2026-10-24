"""
Indexing Engine
================
Handles the full ingestion pipeline for POST /index:
  1. Unzip the uploaded archive
  2. Parse each PDF (PyMuPDF)
  3. Extract dynamic metadata (MetadataRouter)
  4. Batch-embed all chunks (SentenceTransformer)
  5. Store in named CollectionStore (FAISS or numpy)

Also provides the query pipeline used by POST /query.
"""

import os
import re
import time
import zipfile
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer

from ..parsing.pdf_parser import PDFParser
from ..parsing.base import Document, Paragraph, Table
from ..index.collection_store import CollectionStore, ChunkRecord, registry
from ..routing.metadata_router import MetadataRouter
from ..retrieval.synthesis_aggregator import AdaptiveRetriever, classify_query
from ..reasoning.fusion import SymbolicNeuralFusion
from ..graph.builder import GraphBuilder
from ..graph.ttgnn import TTGNN
from ..evidence.structural_graph import StructuralGraph
from ..evidence.coverage_selector import select_evidence


EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384
BATCH_SIZE  = 64

# ---------------------------------------------------------------------------
# In-process query-answer cache (LRU, max 256 entries per server lifetime)
# ---------------------------------------------------------------------------
from functools import lru_cache
import hashlib

_QUERY_CACHE: dict = {}   # (collection_name, query_hash) -> result dict
_CACHE_MAX   = 256

def _cache_key(collection_name: str, query: str) -> str:
    return hashlib.md5(f"{collection_name}::{query.strip().lower()}".encode()).hexdigest()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_to_text(table: Table, max_rows: int = 40) -> str:
    from ..retrieval.hierarchical import HierarchicalRetriever
    r = HierarchicalRetriever.__new__(HierarchicalRetriever)
    return r._format_table_markdown(table)[:3000]


def _iter_doc_chunks(doc: Document) -> List[Dict[str, Any]]:
    """Yields flat chunk dicts for every paragraph and table in the doc."""
    chunks = []
    for para in doc.paragraphs:
        if len(para.text.strip()) < 20:
            continue
        chunks.append({
            "id":       para.id,
            "type":     "paragraph",
            "text":     para.text,
            "doc_name": doc.title,
            "metadata": para.metadata,
        })
    for tbl in doc.tables:
        txt = _table_to_text(tbl)
        if not txt.strip():
            continue
        chunks.append({
            "id":       tbl.id,
            "type":     "table",
            "text":     txt,
            "doc_name": doc.title,
            "metadata": tbl.metadata,
        })
    return chunks


# ---------------------------------------------------------------------------
# Indexing Engine
# ---------------------------------------------------------------------------

class IndexingEngine:
    """
    Singleton-ish engine shared across all collections.
    The encoder is loaded once and reused.
    """

    def __init__(self):
        print("[IndexingEngine] Loading sentence encoder…")
        self.encoder  = SentenceTransformer(EMBED_MODEL)
        self.parser   = PDFParser()
        self.builder  = GraphBuilder(encoder=self.encoder)   # reuses same encoder
        print("[IndexingEngine] Ready.")

    # ------------------------------------------------------------------
    # Public: ingest a zip file → named collection
    # ------------------------------------------------------------------

    def index_zip(
        self,
        zip_path: str,
        collection_name: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Extracts, parses, embeds, and stores all PDFs in zip_path into
        a new (or replaced) collection with the given name.

        Returns a stats dict.
        """
        t0 = time.time()
        # Per Hackathon contract: replace previous contents if re-indexed
        store = registry.create(collection_name, embedding_dim=EMBED_DIM)

        # Invalidate any stale cached query answers for this collection name
        stale_keys = [k for k in _QUERY_CACHE if k.startswith(f"{collection_name}::")]
        for k in stale_keys:
            del _QUERY_CACHE[k]
        if stale_keys:
            print(f"[Cache] Invalidated {len(stale_keys)} cached entries for '{collection_name}'.")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Unzip
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            pdf_paths = []
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    if fname.lower().endswith(".pdf"):
                        pdf_paths.append(os.path.join(root, fname))

            if not pdf_paths:
                return {"status": "error", "message": "No PDF files found in archive"}

            total_chunks = 0
            struct_graph = StructuralGraph()  # lightweight in-memory graph
            for idx, pdf_path in enumerate(pdf_paths, 1):
                doc_name = os.path.basename(pdf_path)
                if progress_callback:
                    progress_callback(idx, len(pdf_paths), doc_name)
                else:
                    print(f"  [{idx}/{len(pdf_paths)}] Parsing {doc_name}…")

                try:
                    doc = self.parser.parse(pdf_path)
                except Exception as e:
                    print(f"  [Warning] Failed to parse {doc_name}: {e}")
                    continue

                # Sample text for metadata extraction (first 5000 chars)
                sample = " ".join(p.text for p in doc.paragraphs[:30])[:5000]
                store.add_document(doc, sample_text=sample)

                # Register in structural evidence graph
                try:
                    struct_graph.add_document(doc)
                except Exception as eg:
                    print(f"  [Warning] StructuralGraph.add_document failed for {doc_name}: {eg}")

                # Collect chunks for batch embedding
                raw_chunks = _iter_doc_chunks(doc)
                if not raw_chunks:
                    continue

                texts = [c["text"] for c in raw_chunks]
                embeddings = self._batch_embed(texts)

                records = []
                for i, raw in enumerate(raw_chunks):
                    records.append(ChunkRecord(
                        chunk_id  = raw["id"],
                        doc_name  = raw["doc_name"],
                        text      = raw["text"],
                        metadata  = raw["metadata"],
                        embedding = embeddings[i],
                    ))
                store.add_chunks(records)
                total_chunks += len(records)

        # Force index build now (so first query is fast)
        store._build_index()
        # Attach structural graph to the collection store for query-time recovery
        store._struct_graph = struct_graph
        print(f"  [StructuralGraph] {struct_graph.node_count()} nodes built for '{collection_name}'.")
        elapsed = time.time() - t0

        return {
            "status":          "success",
            "collection_name": collection_name,
            "num_documents":   len(store.docs),
            "num_chunks":      total_chunks,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed with sentence-transformers, returns list of 1-D np arrays."""
        all_embs = self.encoder.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [all_embs[i] for i in range(len(all_embs))]

    def _synthesis_expand_sections(
        self,
        faiss_results: List[Tuple[float, Any]],  # (score, ChunkRecord)
        store: Any,
        max_sections: int = 30,
        max_chunks_per_section: int = 12,
        output_cap: int = 120,
    ) -> List[Dict[str, Any]]:
        """
        Fast no-re-encode synthesis expansion.

        Algorithm:
          1. Build a section-score map: for each FAISS hit, credit its score to
             the section it belongs to (via metadata['section']).
          2. Pick the top-N sections by total credited score.
          3. For each top section, pull ALL ChunkRecords from the store that
             share the same (doc_name, section) — up to max_chunks_per_section.
          4. Return as chunk dicts with the FAISS-inherited score.

        Zero extra encoder calls. Total complexity: O(num_chunks_in_store).
        """
        # Step 1: score each (doc_name, section) pair from FAISS hits
        section_scores: Dict[Tuple[str, str], float] = {}
        for score, cr in faiss_results:
            section_key = (cr.doc_name, cr.metadata.get("section", "__root__"))
            section_scores[section_key] = section_scores.get(section_key, 0.0) + float(score)

        # Step 2: pick top-N sections
        top_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[:max_sections]
        top_section_set = {k for k, _ in top_sections}

        # Step 3: scan all chunks in the store, collect those in top sections
        # Group by section key, capped at max_chunks_per_section each
        from collections import defaultdict
        section_chunks: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for cr in store.chunks:
            key = (cr.doc_name, cr.metadata.get("section", "__root__"))
            if key in top_section_set:
                section_chunks[key].append(cr)

        # Step 4: build output list
        results: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for key, base_score in top_sections:
            cr_list = section_chunks.get(key, [])
            # Sort by chunk order if possible (page_number), else use insertion order
            cr_list.sort(key=lambda c: c.metadata.get("page_number", 0))
            for cr in cr_list[:max_chunks_per_section]:
                if cr.chunk_id in seen_ids:
                    continue
                seen_ids.add(cr.chunk_id)
                results.append({
                    "id":       cr.chunk_id,
                    "type":     "paragraph",
                    "text":     cr.text,
                    "score":    float(base_score),
                    "document": cr.doc_name,
                    "metadata": cr.metadata,
                })

        # Also include direct FAISS hits that may have been missed (e.g. __root__ section)
        for score, cr in faiss_results:
            if cr.chunk_id not in seen_ids:
                seen_ids.add(cr.chunk_id)
                results.append({
                    "id":       cr.chunk_id,
                    "type":     "paragraph",
                    "text":     cr.text,
                    "score":    float(score),
                    "document": cr.doc_name,
                    "metadata": cr.metadata,
                })

        return results[:output_cap]

    # ------------------------------------------------------------------
    # Public: query pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        collection_name: str,
        top_chunks: int = 8,
        provider: str = "opencode",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: metadata filter → FAISS retrieval → adaptive retrieval
        → LLM generation.  Returns answer + chunks + pipeline trace.
        """
        from ..observability.langfuse_tracer import ObservabilityTracer

        tracer = ObservabilityTracer(query=query, collection=collection_name)
        tracer.start()
        trace_steps: List[Dict[str, Any]] = []

        store = registry.get(collection_name)
        if store is None:
            return {
                "error":  "unknown collection",
                "answer": "",
                "chunks": [],
                "trace":  [],
            }

        # ── Query cache check ────────────────────────────────────────────
        ck = _cache_key(collection_name, query)
        if ck in _QUERY_CACHE:
            cached = dict(_QUERY_CACHE[ck])
            cached["_cached"] = True
            print(f"[Cache] HIT for query: {query[:60]}")
            return cached

        # ── Step 1: Query parsing & metadata filter ──────────────────
        t1 = time.time()
        with tracer.span("metadata_filter") as sp:
            query_constraints = store.router.query_parser.parse(query)
            all_docs = list(store.docs.keys())
            filtered_docs = store.router.filter_docs(query, all_docs)
            sp.update(
                input={"query": query, "total_docs": len(all_docs)},
                output={"filtered_docs": filtered_docs, "constraints": query_constraints},
            )

        trace_steps.append({
            "step_name":   "metadata_filter",
            "description": "Extracted entity/year constraints from query and pre-filtered document pool",
            "input":  {"query": query, "total_docs": len(all_docs)},
            "output": {
                "selected_docs":  filtered_docs,
                "constraints":    query_constraints,
                "docs_eliminated": len(all_docs) - len(filtered_docs),
            },
        })

        # ── Step 2: Classify query type ───────────────────────────────
        query_type = classify_query(query)
        trace_steps.append({
            "step_name":   "query_classifier",
            "description": "Determined whether query requires targeted or synthesis retrieval",
            "input":  {"query": query},
            "output": {"query_type": query_type},
        })

        # ── Step 3: Hybrid BM25 + FAISS retrieval (RRF fusion) ───────────
        t3 = time.time()
        with tracer.span("hybrid_retrieval") as sp:
            q_emb = self.encoder.encode(
                [query], show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )[0]
            # Fetch candidates: keep ratio 3× llm_cap to have enough for coverage selector
            fetch_k = 15 if query_type == "synthesis" else 12
            hybrid_results = store.hybrid_search(
                query_embedding=q_emb,
                query_str=query,
                top_k=fetch_k,
                doc_filter=filtered_docs,
            )
            sp.update(
                input={"fetch_k": fetch_k, "query_type": query_type},
                output={"hybrid_results": len(hybrid_results)},
            )

        trace_steps.append({
            "step_name":   "hybrid_retrieval",
            "description": "BM25 (lexical) + FAISS (semantic) fused via RRF — catches NPA, SARFAESI, exact ₹ figures",
            "input":  {"fetch_k": fetch_k, "doc_pool": len(filtered_docs)},
            "output": {"chunks_fused": len(hybrid_results)},
        })

        # ── Step 4: Context selection ─────────────────────────────────────
        # PERF: keep LLM context small — 5 chunks synthesis, 4 targeted.
        # Smaller prompt = faster LLM generation (biggest latency lever).
        llm_cap = 5 if query_type == "synthesis" else 4
        final_chunks_raw: List[Dict[str, Any]] = [
            {
                "id":       cr.chunk_id,
                "type":     cr.metadata.get("type", "paragraph"),
                "text":     cr.text,
                "score":    float(score),
                "document": cr.doc_name,
                "metadata": cr.metadata,
            }
            for score, cr in hybrid_results[:llm_cap]
        ]

        # ── Step 4b: Coverage-based evidence selection ──────────────────
        # Replaces pure RRF ranking with obligation-aware greedy selection.
        # Uses StructuralGraph for one recovery pass on missing requirements.
        with tracer.span("evidence_selection") as sp_ev:
            # Get structural graph for this collection (built lazily per doc)
            struct_graph: Optional[StructuralGraph] = getattr(store, "_struct_graph", None)
            entity_constraints = {
                "doc_name": filtered_docs
            } if filtered_docs else None

            final_chunks, ev_trace = select_evidence(
                query=query,
                candidates=final_chunks_raw,
                struct_graph=struct_graph,
                entity_constraints=entity_constraints,
                budget_tokens=6000,
                max_recovery_chunks=4,
            )
            sp_ev.update(
                input={"candidates": len(final_chunks_raw)},
                output={
                    "selected": len(final_chunks),
                    "recovery_used": ev_trace.get("recovery_used", False),
                    "ledger": ev_trace.get("ledger_summary", {}),
                },
            )

        trace_steps.append({
            "step_name":   "coverage_evidence_selection",
            "description": "Obligation-aware greedy selection: maximises requirement coverage, penalises redundancy, budget-constrained",
            "input":  {"query_type": query_type, "candidates": len(final_chunks_raw)},
            "output": ev_trace,
        })

        # ── Step 5: LLM generation ────────────────────────────────────
        t5 = time.time()
        fusion = SymbolicNeuralFusion(provider=provider, model=model)

        with tracer.span("llm_generation") as sp:
            result = fusion(query, final_chunks)
            gen_latency = time.time() - t5
            sp.update(
                input={"chunks": len(final_chunks), "provider": fusion.provider},
                output={"answer_length": len(result.get("answer", ""))},
            )

        # Estimate tokens / cost (rough approximation)
        answer_text  = result.get("answer", "")
        prompt_toks  = sum(len(c["text"].split()) for c in final_chunks) + len(query.split())
        comp_toks    = len(answer_text.split())
        # DeepSeek flash cost: ~$0.07/1M input, $0.28/1M output (as of Oct 2024)
        cost_usd     = (prompt_toks * 0.07 + comp_toks * 0.28) / 1_000_000
        tracer.record_generation(
            model=fusion.model,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            cost_usd=cost_usd,
        )

        trace_steps.append({
            "step_name":   "llm_answer_generation",
            "description": f"LLM ({fusion.provider}: {fusion.model}) synthesises grounded answer from retrieved chunks",
            "input":  {
                "provider":        fusion.provider,
                "model":           fusion.model,
                "num_chunks":      len(final_chunks),
                "prompt_tokens":   prompt_toks,
            },
            "output": {
                "completion_tokens": comp_toks,
                "cost_usd":          round(cost_usd, 6),
                "latency_s":         round(gen_latency, 2),
            },
        })

        # ── Finalise trace ────────────────────────────────────────────
        tracer.finish(answer=answer_text)

        # Build chunk response list (strictly conforming to Hackathon API Contract)
        chunk_response = []
        for c in final_chunks:
            meta = c.get("metadata", {})
            doc_name = str(c.get("document") or meta.get("document", ""))
            
            # Extract clean 1-based page numbers as list of integers
            raw_pages = meta.get("page_range") or meta.get("page_numbers") or [meta.get("page_number", 1)]
            if not isinstance(raw_pages, (list, tuple, set)):
                raw_pages = [raw_pages]
            clean_pages = []
            for p in raw_pages:
                try:
                    clean_pages.append(int(p))
                except (ValueError, TypeError):
                    pass
            if not clean_pages:
                clean_pages = [1]
            
            content_text = str(c.get("text", c.get("content", "")))
            
            chunk_response.append({
                "content": content_text,
                "metadata": {
                    "file_name": doc_name,
                    "page_numbers": clean_pages,
                    "section": meta.get("section", ""),
                    "score": round(c.get("score", 0.0), 4),
                },
                # Backward-compatibility flat aliases
                "document": doc_name,
                "page_numbers": clean_pages,
                "section": meta.get("section", ""),
                "score": round(c.get("score", 0.0), 4),
            })

        result_dict = {
            "answer":            answer_text,
            "chunks":            chunk_response,
            "trace":             trace_steps,
            "query_type":        query_type,
            "provider":          fusion.provider,
            "model":             fusion.model,
            "observability":     tracer.summary(),
        }

        # Store in query cache (evict oldest when at cap)
        if len(_QUERY_CACHE) >= _CACHE_MAX:
            oldest_key = next(iter(_QUERY_CACHE))
            del _QUERY_CACHE[oldest_key]
        _QUERY_CACHE[ck] = result_dict

        return result_dict


# Module-level singleton (instantiated once on server startup)
_engine: Optional[IndexingEngine] = None


def get_engine() -> IndexingEngine:
    global _engine
    if _engine is None:
        _engine = IndexingEngine()
    return _engine
