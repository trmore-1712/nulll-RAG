"""
Evidence Obligation Ledger
===========================
Parses a query into typed evidence REQUIREMENTS and tracks whether
retrieved candidates satisfy them.

Design principles:
  - No LLM call: requirements derived from regex + keyword patterns only
  - Requirements are generic (not hardcoded to finance)
  - Supports: NUMERICAL_FACT, ENTITY_FACT, CAUSAL_EXPLANATION,
              COMPARISON, ENUMERATION, TEMPORAL_FACT, PROVENANCE
  - Each candidate is mapped to requirements it satisfies (full/partial)
  - Tracks: supported, partial, missing, conflicted, unresolved

Usage:
    ledger = ObligationLedger()
    reqs = ledger.parse_requirements(query)
    ledger.evaluate_candidates(reqs, chunk_dicts)
    missing = [r for r in reqs if r.status == RequirementStatus.MISSING]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class RequirementType(str, Enum):
    NUMERICAL_FACT      = "numerical_fact"       # "how much", "what was the X"
    ENTITY_FACT         = "entity_fact"           # "who", "which company"
    CAUSAL_EXPLANATION  = "causal_explanation"    # "why", "because of"
    COMPARISON          = "comparison"            # "compare", "vs", "difference"
    ENUMERATION         = "enumeration"           # "list all", "pull together every"
    TEMPORAL_FACT       = "temporal_fact"         # "when", "which year", "since"
    PROVENANCE          = "provenance"            # "according to", "source"
    GENERAL             = "general"               # catch-all


class RequirementStatus(str, Enum):
    UNRESOLVED = "unresolved"   # not yet evaluated
    SUPPORTED  = "supported"    # at least one candidate fully covers this
    PARTIAL    = "partial"      # partially covered but not complete
    MISSING    = "missing"      # no candidate covers this
    CONFLICTED = "conflicted"   # candidates contradict each other


@dataclass
class Requirement:
    """A single typed information need derived from the query."""
    req_id: str
    req_type: RequirementType
    description: str             # human-readable what we need
    keywords: List[str]          # keywords that help identify supporting evidence
    status: RequirementStatus = RequirementStatus.UNRESOLVED
    supporting_chunk_ids: List[str] = field(default_factory=list)
    conflicting_chunk_ids: List[str] = field(default_factory=list)
    # Coverage score 0-1 (1 = fully satisfied)
    coverage: float = 0.0


# ---------------------------------------------------------------------------
# Pattern library (domain-agnostic with financial specialisation)
# ---------------------------------------------------------------------------

_NUMERICAL_PATTERNS = re.compile(
    r"\b(how much|what (?:is|was|were) the (?:amount|value|total|figure|sum|cost|revenue|loss|profit|balance|rate|ratio|margin|percentage|change)|"
    r"calculate|compute|what percent|how many|quantif|what (?:number|count)|"
    r"crore|million|billion|lakh|rs\.?|inr|\$|€|£)\b",
    re.IGNORECASE
)

_ENTITY_PATTERNS = re.compile(
    r"\b(who|which (?:company|entity|bank|lender|auditor|director|person|firm)|"
    r"name the|identify the|what (?:company|bank|lender|party)|"
    r"management|board|auditor|promoter|subsidiary|affiliate)\b",
    re.IGNORECASE
)

_CAUSAL_PATTERNS = re.compile(
    r"\b(why|what caused|reason for|due to|because of|attributed to|"
    r"resulted in|led to|impact of|effect of|consequence)\b",
    re.IGNORECASE
)

_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|vs\.?|versus|difference between|contrast|relative to|"
    r"year.?on.?year|yoy|qoq|compared (?:to|with)|change from|growth from|"
    r"improvement|deteriorat|better|worse|higher|lower than)\b",
    re.IGNORECASE
)

_ENUMERATION_PATTERNS = re.compile(
    r"\b(list (?:all|every)|pull together|enumerate|what (?:are all|were all)|"
    r"every (?:instance|event|case|action|red.?flag|concern|default|issue)|"
    r"all (?:the|instances of)|comprehensive|complete (?:list|picture))\b",
    re.IGNORECASE
)

_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|which year|what (?:year|quarter|period|date)|since when|"
    r"fy\s*\d{2,4}|20\d{2}[-/–]20?\d{2}|q[1-4]\s*\d{4}|"
    r"during|in the year|for the period)\b",
    re.IGNORECASE
)

_PROVENANCE_PATTERNS = re.compile(
    r"\b(according to|as per|source|cite|citation|reported by|per the|"
    r"auditor(?:'s)?|annual report|note \d+|page \d+|stated in)\b",
    re.IGNORECASE
)

# Financial domain — map signals to keywords for evidence matching
_FINANCIAL_DOMAIN_KEYWORDS = {
    "lender": ["lender", "bank", "creditor", "recall", "npa", "arc", "arca", "ics"],
    "default": ["default", "overdue", "npа", "npa", "restructur", "debenture", "ncd"],
    "auditor": ["auditor", "qualification", "going concern", "audit", "opinion"],
    "market": ["stock", "exchange", "delist", "suspend", "bse", "nse", "sat"],
    "governance": ["director", "secretary", "board", "disqualif", "mgт", "mgt", "sebi"],
    "overseas": ["subsidiary", "overseas", "foreign", "italy", "ecuador", "sofinter"],
    "numeric": ["crore", "cr", "million", "lakh", "rs", "inr", "₹", "%", "percent"],
}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class ObligationLedger:
    """
    Parses a query into typed requirements and evaluates candidate evidence
    against those requirements.

    No LLM calls — entirely rule-based for speed and reproducibility.
    """

    def parse_requirements(self, query: str) -> List[Requirement]:
        """
        Derive typed requirements from the query text.
        Returns an ordered list (most specific first).
        """
        reqs: List[Requirement] = []
        q = query.strip()
        rid = 0

        def _add(req_type, desc, keywords):
            nonlocal rid
            reqs.append(Requirement(
                req_id=f"req_{rid:02d}",
                req_type=req_type,
                description=desc,
                keywords=keywords,
            ))
            rid += 1

        # Enumeration (check first — synthesis queries have multiple sub-needs)
        if _ENUMERATION_PATTERNS.search(q):
            _add(RequirementType.ENUMERATION,
                 "Complete enumeration of all relevant items/events",
                 _extract_domain_keywords(q))

        # Numerical
        if _NUMERICAL_PATTERNS.search(q):
            nums = _extract_numbers_context(q)
            _add(RequirementType.NUMERICAL_FACT,
                 f"Numerical value(s) required: {', '.join(nums) or 'figure from context'}",
                 ["amount", "figure", "crore", "million", "percent", "ratio"] + nums)

        # Comparison
        if _COMPARISON_PATTERNS.search(q):
            _add(RequirementType.COMPARISON,
                 "Comparison of two or more values/entities",
                 ["previous", "current", "change", "difference", "year", "growth"])

        # Temporal
        if _TEMPORAL_PATTERNS.search(q):
            years = re.findall(r"\b(20\d{2}|fy\s*\d{2,4})\b", q, re.IGNORECASE)
            _add(RequirementType.TEMPORAL_FACT,
                 f"Temporal context required: {', '.join(years) or 'period'}",
                 years + ["year", "period", "quarter", "fy", "annual"])

        # Causal
        if _CAUSAL_PATTERNS.search(q):
            _add(RequirementType.CAUSAL_EXPLANATION,
                 "Causal explanation of an event or outcome",
                 ["because", "caused", "resulted", "due to", "reason"])

        # Entity
        if _ENTITY_PATTERNS.search(q):
            entities = _extract_entities(q)
            _add(RequirementType.ENTITY_FACT,
                 f"Entity identification: {', '.join(entities) or 'entity from context'}",
                 entities + ["company", "bank", "director", "lender"])

        # Provenance
        if _PROVENANCE_PATTERNS.search(q):
            _add(RequirementType.PROVENANCE,
                 "Source/citation provenance required",
                 ["page", "report", "note", "auditor", "section"])

        # Always add a GENERAL catch-all using query keywords
        q_keywords = [
            w for w in re.findall(r"\b[a-z]{4,}\b", q.lower())
            if w not in {"this", "that", "with", "from", "into", "have", "been",
                         "what", "which", "when", "where", "there", "their"}
        ][:12]
        _add(RequirementType.GENERAL,
             "General information from context",
             q_keywords)

        return reqs

    def evaluate_candidates(
        self,
        requirements: List[Requirement],
        candidates: List[Dict[str, Any]],
    ) -> List[Requirement]:
        """
        For each requirement, scan candidates and mark which ones
        satisfy it (full or partial). Updates requirement.status in place.

        Returns the updated requirement list.
        """
        for req in requirements:
            keyword_set = {kw.lower() for kw in req.keywords}
            if not keyword_set:
                req.status = RequirementStatus.UNRESOLVED
                continue

            scores: List[Tuple[str, float]] = []
            for chunk in candidates:
                chunk_id = chunk.get("id", "")
                text = (chunk.get("text", "") or "").lower()
                # Keyword overlap score
                matched = sum(1 for kw in keyword_set if kw in text)
                if matched > 0:
                    score = matched / len(keyword_set)
                    scores.append((chunk_id, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            if not scores:
                req.status = RequirementStatus.MISSING
                req.coverage = 0.0
            elif scores[0][1] >= 0.4:
                req.status = RequirementStatus.SUPPORTED
                req.coverage = min(1.0, scores[0][1] * 1.5)
                req.supporting_chunk_ids = [cid for cid, s in scores if s >= 0.2]
            elif scores[0][1] >= 0.15:
                req.status = RequirementStatus.PARTIAL
                req.coverage = scores[0][1]
                req.supporting_chunk_ids = [cid for cid, s in scores if s >= 0.1]
            else:
                req.status = RequirementStatus.MISSING
                req.coverage = 0.0

        # Check for conflicts: same requirement → candidates with contradictory numbers
        # (simple heuristic: look for different crore values for same concept)
        for req in requirements:
            if req.req_type == RequirementType.NUMERICAL_FACT:
                nums_found: List[float] = []
                for chunk in candidates:
                    if chunk.get("id", "") in req.supporting_chunk_ids:
                        text = chunk.get("text", "")
                        nums = re.findall(r"(?:rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:crore|cr|lakh|million)", text, re.IGNORECASE)
                        nums_found.extend([float(n.replace(",", "")) for n in nums if n])
                if len(set(nums_found)) > 3:
                    req.status = RequirementStatus.CONFLICTED

        return requirements

    def summary(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Return a concise ledger summary for tracing."""
        status_counts: Dict[str, int] = {}
        for req in requirements:
            status_counts[req.status.value] = status_counts.get(req.status.value, 0) + 1
        coverage = sum(r.coverage for r in requirements) / len(requirements) if requirements else 0.0
        return {
            "total_requirements": len(requirements),
            "status_counts": status_counts,
            "avg_coverage": round(coverage, 3),
            "missing": [r.description for r in requirements if r.status == RequirementStatus.MISSING],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain_keywords(query: str) -> List[str]:
    """Extract domain-relevant keywords from query for evidence matching."""
    q_lower = query.lower()
    keywords = []
    for domain, kws in _FINANCIAL_DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in q_lower:
                keywords.extend(kws[:3])
                break
    # Also include long words from query
    keywords += [w for w in re.findall(r"\b[a-z]{5,}\b", q_lower)
                 if w not in {"which", "where", "there", "their", "about", "every"}]
    return list(dict.fromkeys(keywords))[:15]


def _extract_numbers_context(query: str) -> List[str]:
    """Extract numeric references from query (years, amounts)."""
    matches = re.findall(r"\b(\d[\d,]*(?:\.\d+)?)\b", query)
    return matches[:5]


def _extract_entities(query: str) -> List[str]:
    """Extract likely named entities (capitalized multi-word phrases)."""
    # Simple heuristic: capitalized words/phrases
    entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    return [e for e in entities if len(e) > 3][:6]
