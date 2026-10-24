"""
Dynamic Metadata Extractor & Entity-Temporal Router
====================================================
Solves Failure Mode 1: Near-duplicate / cross-year document confusion.

At INGESTION time:
  - Extracts entity name, fiscal period, scope (consolidated/standalone),
    document type, and key topic tags from each document's filename and content.

At QUERY time:
  - Parses the question to detect temporal (FY year) and entity constraints.
  - Filters the candidate document pool BEFORE semantic search runs.
  - Result: semantic search compares only against the correctly-scoped documents.
"""

import re
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Fiscal year / period patterns
# ---------------------------------------------------------------------------
_FY_PATTERNS = [
    r"\bFY\s*(\d{4})[–\-](\d{2,4})\b",          # FY2025-26, FY 2025-26
    r"\bFY\s*(\d{4})\b",                          # FY2025
    r"\bfinancial year\s+(\d{4})[–\-](\d{2,4})\b",
    r"\byear ended\s+(?:March|March 31|31 March)[,\s]+(\d{4})\b",
    r"\b(\d{4})[–\-](\d{2,4})\s+annual report\b",
    r"\bannual report\s+(\d{4})\b",
    r"\b(\d{4})\s+annual report\b",
    r"\b(20\d{2})\b",                             # bare 4-digit year fallback
]

_SCOPE_PATTERNS = {
    "consolidated": [r"\bconsolidated\b", r"\bgroup\b", r"\bconsolidated basis\b"],
    "standalone":   [r"\bstandalone\b",   r"\bcompany\b(?! limited| inc)", r"\bsolo\b"],
}

_DOCTYPE_PATTERNS = {
    "annual_report":   [r"\bannual report\b", r"\bintegrated report\b", r"\biar\b"],
    "10k":             [r"\b10[–\-]?k\b"],
    "10q":             [r"\b10[–\-]?q\b"],
    "quarterly":       [r"\bquarterly\b", r"\bq[1-4]\s+\d{4}\b"],
    "audit":           [r"\baudit\b", r"\bauditors?\b"],
    "risk":            [r"\brisk\b", r"\bcredit risk\b", r"\bmarket risk\b"],
    "financial_notes": [r"\bnotes?\s+to\s+(financial|accounts)\b"],
}

# Company name normalisation – extend as needed
_ENTITY_ALIASES = {
    "gammon india":    ["gammon india", "gammon", "gil"],
    "alphabet":        ["alphabet inc", "alphabet", "google"],
    "microsoft":       ["microsoft", "msft"],
    "apple":           ["apple inc", "apple"],
    "amazon":          ["amazon", "amzn", "aws"],
    "meta":            ["meta platforms", "meta", "facebook"],
    "tata":            ["tata group", "tata"],
    "reliance":        ["reliance industries", "ril"],
    "infosys":         ["infosys", "infy"],
    "wipro":           ["wipro"],
    "hdfc":            ["hdfc bank", "hdfc"],
    "icici":           ["icici bank", "icici"],
}


class DocumentMetadata:
    """Structured metadata record extracted for a single document."""

    def __init__(
        self,
        doc_name: str,
        entity: Optional[str] = None,
        fiscal_years: Optional[List[str]] = None,
        scope: Optional[str] = None,
        doc_type: Optional[str] = None,
        topic_tags: Optional[List[str]] = None,
    ):
        self.doc_name = doc_name
        self.entity = entity
        self.fiscal_years: List[str] = fiscal_years or []
        self.scope = scope                # "consolidated" | "standalone" | None
        self.doc_type = doc_type          # "annual_report" | "10k" | …
        self.topic_tags: List[str] = topic_tags or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "entity": self.entity,
            "fiscal_years": self.fiscal_years,
            "scope": self.scope,
            "doc_type": self.doc_type,
            "topic_tags": self.topic_tags,
        }

    def __repr__(self) -> str:
        return (
            f"DocumentMetadata(doc={self.doc_name!r}, entity={self.entity!r}, "
            f"fy={self.fiscal_years}, scope={self.scope!r}, type={self.doc_type!r})"
        )


class MetadataExtractor:
    """Extracts structured metadata from a document's filename and sample text."""

    @staticmethod
    def _extract_years(text: str) -> List[str]:
        """Returns all fiscal year strings found in the text."""
        text_l = text.lower()
        years: List[str] = []
        for pat in _FY_PATTERNS:
            for m in re.finditer(pat, text_l):
                groups = [g for g in m.groups() if g]
                if len(groups) == 2:
                    y1, y2 = groups
                    # Normalise short year suffix (e.g. "26" -> "2026")
                    y2 = y2 if len(y2) == 4 else f"20{y2}" if int(y2) < 50 else f"19{y2}"
                    years.append(f"{y1}-{y2}")
                elif len(groups) == 1:
                    years.append(groups[0])
        # Deduplicate while preserving order
        seen, out = set(), []
        for y in years:
            if y not in seen:
                seen.add(y)
                out.append(y)
        return out

    @staticmethod
    def _extract_entity(text: str) -> Optional[str]:
        text_l = text.lower()
        for canonical, aliases in _ENTITY_ALIASES.items():
            for alias in aliases:
                if alias in text_l:
                    return canonical
        return None

    @staticmethod
    def _extract_scope(text: str) -> Optional[str]:
        text_l = text.lower()
        for scope, patterns in _SCOPE_PATTERNS.items():
            for p in patterns:
                if re.search(p, text_l):
                    return scope
        return None

    @staticmethod
    def _extract_doc_type(text: str) -> Optional[str]:
        text_l = text.lower()
        for dtype, patterns in _DOCTYPE_PATTERNS.items():
            for p in patterns:
                if re.search(p, text_l):
                    return dtype
        return None

    @staticmethod
    def _extract_topics(text: str) -> List[str]:
        topics_kw = {
            "credit_risk":    [r"\bcredit\s+risk\b", r"\bnpa\b", r"\bdefault\b"],
            "governance":     [r"\bgovernance\b", r"\bdirector\b", r"\bboard\b"],
            "audit":          [r"\baudit\b", r"\bauditor\b"],
            "revenue":        [r"\brevenue\b", r"\bturnover\b", r"\bsales\b"],
            "debt":           [r"\bdebt\b", r"\bborrowings\b", r"\bloc\b", r"\bncd\b"],
            "esg":            [r"\besg\b", r"\bsustainab\b", r"\bclimate\b"],
            "capex":          [r"\bcapex\b", r"\bcapital expenditure\b"],
            "dividend":       [r"\bdividend\b"],
        }
        text_l = text.lower()
        found = []
        for tag, pats in topics_kw.items():
            for p in pats:
                if re.search(p, text_l):
                    found.append(tag)
                    break
        return found

    def extract(self, doc_name: str, sample_text: str = "") -> DocumentMetadata:
        """
        Extract metadata from a document's filename plus an optional text sample
        (e.g. first 5000 chars of the parsed document).
        """
        combined = f"{doc_name} {sample_text}"
        return DocumentMetadata(
            doc_name=doc_name,
            entity=self._extract_entity(combined),
            fiscal_years=self._extract_years(combined),
            scope=self._extract_scope(combined),
            doc_type=self._extract_doc_type(combined),
            topic_tags=self._extract_topics(combined),
        )


class QueryParser:
    """Parses a natural-language query to identify entity and temporal constraints."""

    extractor = MetadataExtractor()

    def parse(self, query: str) -> Dict[str, Any]:
        """Returns a dict of detected constraints: entity, fiscal_years, scope, topics."""
        return {
            "entity":       self.extractor._extract_entity(query),
            "fiscal_years": self.extractor._extract_years(query),
            "scope":        self.extractor._extract_scope(query),
            "topics":       self.extractor._extract_topics(query),
        }


class MetadataRouter:
    """
    Pre-filters the document pool using extracted metadata constraints so that
    semantic search operates only on genuinely relevant documents.

    Usage:
        router = MetadataRouter()

        # At ingestion: register every document
        router.register("Gammon_IAR_2025-26.pdf", parsed_doc_sample_text)

        # At query time: narrow the pool
        relevant_docs = router.filter_docs(query, all_doc_names)
    """

    def __init__(self):
        self.extractor = MetadataExtractor()
        self.query_parser = QueryParser()
        self._registry: Dict[str, DocumentMetadata] = {}

    def register(self, doc_name: str, sample_text: str = "") -> DocumentMetadata:
        """Register a document and its extracted metadata."""
        meta = self.extractor.extract(doc_name, sample_text)
        self._registry[doc_name] = meta
        return meta

    def get_metadata(self, doc_name: str) -> Optional[DocumentMetadata]:
        return self._registry.get(doc_name)

    def all_metadata(self) -> Dict[str, DocumentMetadata]:
        return dict(self._registry)

    def filter_docs(
        self,
        query: str,
        candidate_docs: List[str],
        strict_entity: bool = True,
        strict_year: bool = True,
    ) -> List[str]:
        """
        Returns the subset of candidate_docs that match the query's
        entity / fiscal-year constraints.

        Falls back to all candidates if filters would return nothing
        (safety net so we never return an empty pool).
        """
        constraints = self.query_parser.parse(query)
        q_entity = constraints.get("entity")
        q_years  = constraints.get("fiscal_years", [])

        filtered = []
        for doc in candidate_docs:
            meta = self._registry.get(doc)
            if meta is None:
                # Unregistered doc: include conservatively
                filtered.append(doc)
                continue

            entity_match = True
            year_match   = True

            if strict_entity and q_entity and meta.entity:
                entity_match = (meta.entity == q_entity)

            if strict_year and q_years and meta.fiscal_years:
                # Accept doc if ANY query year appears in doc's fiscal years
                year_match = any(
                    qy in " ".join(meta.fiscal_years) or
                    any(qy in fy or fy in qy for fy in meta.fiscal_years)
                    for qy in q_years
                )

            if entity_match and year_match:
                filtered.append(doc)

        # Safety: if strict filtering wiped everything, relax and return all
        if not filtered:
            return candidate_docs

        return filtered

    def explain_filter(self, query: str, candidate_docs: List[str]) -> Dict[str, Any]:
        """Debug helper: returns constraints + filter decision per doc."""
        constraints = self.query_parser.parse(query)
        filtered    = self.filter_docs(query, candidate_docs)
        decisions   = {}
        for doc in candidate_docs:
            decisions[doc] = {
                "included": doc in filtered,
                "metadata": self._registry.get(doc, {}).to_dict() if self._registry.get(doc) else {},
            }
        return {
            "query_constraints": constraints,
            "decisions": decisions,
            "selected": filtered,
        }
