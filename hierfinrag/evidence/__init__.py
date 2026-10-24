"""
Evidence Intelligence Layer
============================
Adds structural evidence links, obligation-aware requirement tracking,
and coverage-based evidence selection on top of the existing BM25+FAISS
hybrid retrieval.

Modules:
  structural_graph   — lightweight links from GraphBuilder output
  obligation_ledger  — query → typed requirements, evidence mapping
  coverage_selector  — greedy marginal-coverage selection with budget
"""
from .structural_graph import StructuralGraph
from .obligation_ledger import ObligationLedger, Requirement, RequirementStatus
from .coverage_selector import CoverageSelector, select_evidence
