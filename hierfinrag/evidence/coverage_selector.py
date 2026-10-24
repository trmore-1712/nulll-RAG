"""
Coverage-Based Evidence Selector
==================================
Replaces pure RRF-ranking with obligation-aware greedy selection.

Algorithm:
  1. Start with obligation requirements from ObligationLedger.
  2. Greedy selection: score each candidate by
       marginal_coverage_gain × relevance - redundancy_penalty
     subject to a configurable token budget.
  3. One recovery pass: for still-missing requirements, use
     StructuralGraph to find sibling/page-adjacent evidence.
  4. Return selected chunks annotated with which requirements they cover.

Key design:
  - Does NOT replace retrieval (BM25+FAISS+RRF stays).
  - Runs AFTER hybrid_search(), BEFORE LLM generation.
  - O(N × R) where N = candidates, R = requirements — fast even for N=200.
  - Configurable: budget_tokens, max_recovery_chunks, redundancy_threshold.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple, Any

from .obligation_ledger import (
    ObligationLedger, Requirement, RequirementStatus, RequirementType
)
from .structural_graph import StructuralGraph


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_TOKENS = 6000   # ~4,000 words — safe for 8K context LLMs
DEFAULT_MAX_RECOVERY  = 4      # at most 4 chunks added via structural recovery
REDUNDANCY_THRESHOLD  = 0.70   # Jaccard similarity above this → redundant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Rough word-based token estimate (1 token ≈ 0.75 words)."""
    return max(1, int(len(text.split()) / 0.75))


def _jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two text strings."""
    words_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
    words_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _marginal_coverage(
    candidate: Dict[str, Any],
    requirements: List[Requirement],
    already_covered: Set[str],
) -> float:
    """
    How much new requirement coverage does this candidate add?
    Returns a [0, 1] score proportional to newly covered keywords.
    """
    text = (candidate.get("text", "") or "").lower()
    total_new = 0
    total_possible = 0
    for req in requirements:
        if req.req_id in already_covered:
            continue
        keyword_set = {kw.lower() for kw in req.keywords}
        if not keyword_set:
            continue
        matched = sum(1 for kw in keyword_set if kw in text)
        total_new += matched
        total_possible += len(keyword_set)
    if total_possible == 0:
        return 0.0
    return total_new / total_possible


def _relevance_score(candidate: Dict[str, Any]) -> float:
    """Use the retrieval score (RRF or cosine), normalised to [0,1]."""
    score = float(candidate.get("score", 0.0))
    # RRF scores are typically in [0, 0.05] range; cosine in [0, 1]
    # Normalise: if score < 0.1 it's likely RRF, scale up
    if score < 0.1:
        score = score * 20  # scale RRF to ~[0, 1]
    return min(1.0, max(0.0, score))


def _is_redundant(
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
    threshold: float = REDUNDANCY_THRESHOLD,
) -> bool:
    """Return True if candidate is too similar to an already-selected chunk."""
    ctext = candidate.get("text", "")
    for sel in selected:
        if _jaccard(ctext, sel.get("text", "")) >= threshold:
            return True
    return False


def _mark_covered_requirements(
    selected: List[Dict[str, Any]],
    requirements: List[Requirement],
) -> Set[str]:
    """Return req_ids that are at least partially covered by selected chunks."""
    covered = set()
    for req in requirements:
        if req.req_id in covered:
            continue
        keyword_set = {kw.lower() for kw in req.keywords}
        for chunk in selected:
            text = (chunk.get("text", "") or "").lower()
            if any(kw in text for kw in keyword_set):
                covered.add(req.req_id)
                break
    return covered


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

class CoverageSelector:
    """
    Greedy marginal-coverage evidence selector with structural recovery pass.

    Parameters
    ----------
    budget_tokens : int
        Maximum total tokens in the selected evidence context.
    max_recovery_chunks : int
        Maximum extra chunks to add via structural graph recovery.
    redundancy_threshold : float
        Jaccard similarity above which a chunk is considered redundant.
    """

    def __init__(
        self,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        max_recovery_chunks: int = DEFAULT_MAX_RECOVERY,
        redundancy_threshold: float = REDUNDANCY_THRESHOLD,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.max_recovery = max_recovery_chunks
        self.redundancy_threshold = redundancy_threshold
        self._ledger = ObligationLedger()

    def select(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        struct_graph: Optional[StructuralGraph] = None,
        entity_constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select evidence from candidates that best covers the query requirements.

        Parameters
        ----------
        query : str
        candidates : list of chunk dicts (from hybrid_search)
        struct_graph : StructuralGraph (optional) — used for recovery pass
        entity_constraints : dict — e.g. {"doc_name": "Gammon India..."} for
                             period/scope filtering (from MetadataRouter)

        Returns
        -------
        (selected_chunks, selection_trace)
        """
        if not candidates:
            return [], {"requirements": [], "selected": 0, "recovery_used": False}

        # 1. Parse requirements from query
        requirements = self._ledger.parse_requirements(query)

        # 2. Filter candidates that violate explicit entity/period constraints
        if entity_constraints:
            filtered = self._apply_constraints(candidates, entity_constraints)
            # If constraints eliminate too much (>80%), be lenient and keep all
            candidates = filtered if len(filtered) >= len(candidates) * 0.2 else candidates

        # 3. Evaluate candidates against requirements before selection
        self._ledger.evaluate_candidates(requirements, candidates)

        # 4. Greedy selection: maximise coverage within token budget
        selected: List[Dict[str, Any]] = []
        selected_ids: Set[str] = set()
        covered_reqs: Set[str] = set()
        tokens_used = 0

        # Sort initial candidates: hybrid score (relevance) as tiebreaker
        ranked_candidates = sorted(
            candidates,
            key=lambda c: _relevance_score(c),
            reverse=True,
        )

        for candidate in ranked_candidates:
            cid = candidate.get("id", "")
            if cid in selected_ids:
                continue

            # Check budget
            tok = _token_count(candidate.get("text", ""))
            if tokens_used + tok > self.budget_tokens:
                continue

            # Check redundancy
            if _is_redundant(candidate, selected, self.redundancy_threshold):
                continue

            # Compute selection score
            cov_gain = _marginal_coverage(candidate, requirements, covered_reqs)
            rel_score = _relevance_score(candidate)
            # Weighted: coverage dominates for synthesis, relevance for targeted
            selection_score = 0.6 * cov_gain + 0.4 * rel_score

            # Always include highly relevant chunks even with low marginal coverage
            if selection_score < 0.05 and rel_score < 0.2 and len(selected) > 5:
                continue

            selected.append({**candidate, "_coverage_score": round(selection_score, 4)})
            selected_ids.add(cid)
            tokens_used += tok
            covered_reqs = _mark_covered_requirements(selected, requirements)

        # 5. Re-evaluate after selection
        self._ledger.evaluate_candidates(requirements, selected)
        missing_reqs = [r for r in requirements if r.status == RequirementStatus.MISSING]

        # 6. One recovery pass using structural graph for missing requirements
        recovery_used = False
        recovery_chunks_added = 0
        if missing_reqs and struct_graph and struct_graph.node_count() > 0:
            recovery_used = True
            recovery_pool: List[Dict[str, Any]] = []

            for chunk in selected[:8]:  # look at top selected chunks
                cid = chunk.get("id", "")
                # Get structural neighbours
                siblings = struct_graph.get_section_siblings(cid, max_siblings=3)
                neighbours = struct_graph.get_page_neighbours(cid, max_neighbours=2)
                for node in siblings + neighbours:
                    if node.node_id not in selected_ids:
                        recovery_pool.append({
                            "id":       node.node_id,
                            "type":     node.node_type,
                            "text":     node.text,
                            "score":    0.1,  # low score — recovery only
                            "document": node.doc_name,
                            "metadata": node.metadata,
                            "_is_recovery": True,
                        })

            # Score recovery candidates against missing requirements
            for rchunk in recovery_pool:
                if recovery_chunks_added >= self.max_recovery:
                    break
                if rchunk["id"] in selected_ids:
                    continue
                tok = _token_count(rchunk.get("text", ""))
                if tokens_used + tok > self.budget_tokens:
                    continue
                cov_gain = _marginal_coverage(rchunk, missing_reqs, covered_reqs)
                if cov_gain > 0.1 and not _is_redundant(rchunk, selected, 0.8):
                    selected.append(rchunk)
                    selected_ids.add(rchunk["id"])
                    tokens_used += tok
                    covered_reqs = _mark_covered_requirements(selected, requirements)
                    recovery_chunks_added += 1

        # 7. Build trace
        ledger_summary = self._ledger.summary(requirements)
        trace = {
            "requirements": [
                {
                    "id": r.req_id,
                    "type": r.req_type.value,
                    "description": r.description,
                    "status": r.status.value,
                    "coverage": r.coverage,
                    "supporting_chunks": len(r.supporting_chunk_ids),
                }
                for r in requirements
            ],
            "selected_chunks": len(selected),
            "tokens_used": tokens_used,
            "recovery_used": recovery_used,
            "recovery_chunks_added": recovery_chunks_added,
            "ledger_summary": ledger_summary,
        }

        return selected, trace

    def _apply_constraints(
        self,
        candidates: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Reject evidence that violates explicit constraints.
        Currently supports: doc_name (exact or partial match).
        """
        doc_filter = constraints.get("doc_name")
        if not doc_filter:
            return candidates
        if isinstance(doc_filter, str):
            doc_filter = [doc_filter]
        doc_filter_lower = [d.lower() for d in doc_filter]
        return [
            c for c in candidates
            if any(
                df in (c.get("document", "") or "").lower()
                for df in doc_filter_lower
            )
        ] or candidates  # fallback to all if filter eliminates everything


# ---------------------------------------------------------------------------
# Convenience function (used in engine.py)
# ---------------------------------------------------------------------------

def select_evidence(
    query: str,
    candidates: List[Dict[str, Any]],
    struct_graph: Optional[StructuralGraph] = None,
    entity_constraints: Optional[Dict[str, Any]] = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    max_recovery_chunks: int = DEFAULT_MAX_RECOVERY,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience wrapper around CoverageSelector.select().
    Returns (selected_chunks, selection_trace).
    """
    selector = CoverageSelector(
        budget_tokens=budget_tokens,
        max_recovery_chunks=max_recovery_chunks,
    )
    return selector.select(
        query=query,
        candidates=candidates,
        struct_graph=struct_graph,
        entity_constraints=entity_constraints,
    )
