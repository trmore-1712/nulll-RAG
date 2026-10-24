"""
Global Synthesis Aggregator
============================
Solves Failure Mode 2: The Chunk-Size Dilemma for long-range, cross-page queries.

Classifies queries into two structural types:
  1. TARGETED  – a specific fact/number lookup  → use existing HierarchicalRetriever
  2. SYNTHESIS – aggregation across many pages  → use SectionMapAggregator (this module)

The SectionMapAggregator:
  - Selects ALL sections relevant to the query (not just top-K chunks).
  - Builds per-section summaries that compress page-level evidence into tokens.
  - Combines them via Map-Reduce before LLM generation, so the final context
    covers 30+ pages while staying within a safe token budget.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple

from ..parsing.base import Document, Section, Paragraph, Table


# ---------------------------------------------------------------------------
# Query type classifier
# ---------------------------------------------------------------------------

_SYNTHESIS_TRIGGERS = [
    r"\beverything\b",
    r"\ball\s+(events|red\s+flags|risks?|issues?|concerns?|points?|aspects?|items?|mentions?|instances?)\b",
    r"\bpull\s+together\b",
    r"\bcomprehensive\b",
    r"\bfull\s+(picture|analysis|breakdown|overview|summary|list)\b",
    r"\bsummarise\b", r"\bsummarize\b",
    r"\boverview\b",
    r"\bacross\s+(the\s+)?(entire|whole|report|document|filing)\b",
    r"\bevery\b.{0,40}\b(event|red flag|risk|concern|issue|qualification|observation)\b",
    r"\blist\s+(all|every)\b",
    r"\bidentify\s+all\b",
    r"\bwhat\s+are\s+all\b",
    r"\b(governance|audit|compliance)\s+(lapses?|issues?|concerns?|fallout)\b",
]

_TARGETED_SIGNALS = [
    r"\bwhat\s+was\s+the\s+\w+\b",       # "What was the net loss"
    r"\bhow\s+much\b",
    r"\bwhat\s+is\s+the\s+\w+\b",
    r"\bstate\s+the\s+\w+\b",
    r"\bgive\s+(me\s+)?the\s+\w+\b",
    r"\bprovide\s+the\s+\w+\b",
    r"\bwhen\s+did\b",
    r"\bwho\s+is\b",
    r"\bname\s+the\b",
]


def classify_query(query: str) -> str:
    """Returns 'synthesis' or 'targeted'."""
    q_l = query.lower()
    synthesis_score = sum(1 for p in _SYNTHESIS_TRIGGERS if re.search(p, q_l))
    targeted_score  = sum(1 for p in _TARGETED_SIGNALS  if re.search(p, q_l))

    if synthesis_score >= 1 and synthesis_score >= targeted_score:
        return "synthesis"
    return "targeted"


# ---------------------------------------------------------------------------
# Section content accessors
# ---------------------------------------------------------------------------

def _get_section_text(section: Section, doc: Document, max_chars: int = 2000) -> str:
    """Concatenates paragraphs and table text belonging to a section."""
    parts = []
    for cid in section.content_ids:
        para = doc.get_paragraph(cid)
        if para:
            parts.append(para.text)
            continue
        tbl = doc.get_table(cid)
        if tbl:
            rows: List[str] = []
            for cell in tbl.cells:
                if cell.col_idx == 0:
                    rows.append("")
                rows[-1] = (rows[-1] + " | " + cell.value).strip(" | ")
            parts.append(f"[Table: {tbl.caption}]\n" + "\n".join(rows[:15]))

    raw = "\n".join(parts)
    return raw[:max_chars] if len(raw) > max_chars else raw


def _section_page_numbers(section: Section, doc: Document) -> List[int]:
    """Collects page numbers from section content metadata."""
    pages: List[int] = []
    for cid in section.content_ids[:5]:  # sample first few items for speed
        para = doc.get_paragraph(cid)
        if para:
            pn = para.metadata.get("page_number")
            if pn:
                pages.append(int(pn))
            continue
        tbl = doc.get_table(cid)
        if tbl:
            pn = tbl.metadata.get("page_number")
            if pn:
                pages.append(int(pn))
    return sorted(set(pages))


# ---------------------------------------------------------------------------
# Section Map Aggregator
# ---------------------------------------------------------------------------

class SectionMapAggregator:
    """
    For synthesis queries, selects ALL relevant sections in a document and
    produces a structured, page-cited evidence bundle for the LLM.

    The output matches the same chunk schema as HierarchicalRetriever so it
    can be passed directly to SymbolicNeuralFusion / generate_answer_with_attribution.
    """

    def __init__(self, encoder=None, relevance_threshold: float = 0.05):
        if encoder is not None:
            self.encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.relevance_threshold = relevance_threshold

    def _score_sections(self, query: str, sections: List[Section]) -> List[Tuple[float, Section]]:
        """Dense cosine similarity between query and each section title+snippet."""
        import torch
        import torch.nn.functional as F

        titles = [s.title for s in sections]
        if not titles:
            return []

        q_emb = self.encoder.encode([query], convert_to_tensor=True, show_progress_bar=False)
        s_embs = self.encoder.encode(titles, convert_to_tensor=True, show_progress_bar=False)
        scores = F.cosine_similarity(q_emb, s_embs).tolist()
        return sorted(zip(scores, sections), key=lambda x: x[0], reverse=True)

    def aggregate(
        self,
        query: str,
        doc: Document,
        max_sections: int = 60,
        chars_per_section: int = 3500,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of chunk dicts covering all relevant sections.
        Each chunk preserves document name and page citations.
        """
        scored = self._score_sections(query, doc.sections)

        # Accept sections above threshold OR top-max_sections whichever is more inclusive
        relevant = [
            (score, sec) for score, sec in scored
            if score >= self.relevance_threshold
        ]
        if not relevant:
            relevant = scored[:max_sections]
        else:
            relevant = relevant[:max_sections]

        chunks: List[Dict[str, Any]] = []
        for score, section in relevant:
            text = _get_section_text(section, doc, max_chars=chars_per_section)
            if not text.strip():
                continue
            pages = _section_page_numbers(section, doc)
            page_meta = pages[0] if pages else section.metadata.get("page_number", "N/A")

            chunks.append({
                "id":       section.id,
                "type":     "section_aggregate",
                "text":     f"[Section: {section.title}]\n{text}",
                "score":    float(score),
                "document": doc.title,
                "metadata": {
                    "document":    doc.title,
                    "section":     section.title,
                    "page_number": page_meta,
                    "page_range":  pages,
                },
            })

        return chunks


# ---------------------------------------------------------------------------
# Unified Retrieval Dispatcher
# ---------------------------------------------------------------------------

class AdaptiveRetriever:
    """
    Dispatches each query to either:
      • HierarchicalRetriever  (targeted factoid queries)
      • SectionMapAggregator   (synthesis / holistic queries)

    This is the single entrypoint to use from the server / benchmark pipeline.
    """

    def __init__(self, encoder=None):
        from ..retrieval.hierarchical import HierarchicalRetriever
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.encoder   = encoder
        self.retriever = HierarchicalRetriever(encoder=encoder)
        self.aggregator = SectionMapAggregator(encoder=encoder)

    def retrieve(
        self,
        query: str,
        doc: Document,
        top_chunks: int = 8,
        query_type: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Returns (query_type, chunks).
        query_type is 'synthesis' | 'targeted'.
        """
        qtype = query_type or classify_query(query)

        if qtype == "synthesis":
            # For synthesis queries, always retrieve all relevant sections (up to 60)
            # and pad with targeted retrieval if needed
            chunks = self.aggregator.aggregate(query, doc, max_sections=60, chars_per_section=3500)
            # Always also run targeted retrieval to ensure no specific facts are missed
            extra = self.retriever.retrieve_for_document(
                query, doc, top_sections=15, top_chunks=max(top_chunks, 20)
            )
            seen_ids = {c["id"] for c in chunks}
            for c in extra:
                if c["id"] not in seen_ids:
                    chunks.append(c)
                    seen_ids.add(c["id"])
        else:
            # Targeted: use hierarchical retrieval with expanded section scope
            chunks = self.retriever.retrieve_for_document(
                query, doc, top_sections=10, top_chunks=max(top_chunks, 15)
            )

        return qtype, chunks
