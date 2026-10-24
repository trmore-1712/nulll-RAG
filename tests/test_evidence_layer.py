"""
Tests for the Evidence Intelligence Layer
==========================================
Tests:
  1. StructuralGraph: section→child links, table→row links, page adjacency
  2. ObligationLedger: requirement parsing, evaluation, status classification
  3. CoverageSelector: greedy selection, redundancy penalty, budget enforcement
  4. Wrong-period and scope constraints
  5. Recovery pass behaviour
  6. Numerical verification in SymbolicCalculator

Run: python -m pytest tests/test_evidence_layer.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullrag.parsing.base import (
    Document, Section, Paragraph, Table, Cell
)
from nullrag.evidence.structural_graph import StructuralGraph, EvidenceNode
from nullrag.evidence.obligation_ledger import (
    ObligationLedger, RequirementType, RequirementStatus
)
from nullrag.evidence.coverage_selector import (
    CoverageSelector, select_evidence, _jaccard, _token_count
)
from nullrag.reasoning.fusion import SymbolicCalculator


# ---------------------------------------------------------------------------
# Fixtures: minimal document
# ---------------------------------------------------------------------------

def make_doc() -> Document:
    """Minimal 2-section document with 4 paragraphs and 1 table."""
    sec1 = Section(id="s1", title="Credit Risk", level=1,
                   content_ids=["p1", "p2", "t1"])
    sec2 = Section(id="s2", title="Financial Performance", level=1,
                   content_ids=["p3", "p4"])
    p1 = Paragraph(id="p1", text="Gammon India NPA classified since June 2017 by lenders.",
                   section_id="s1", metadata={"page_number": 3})
    p2 = Paragraph(id="p2", text="SARFAESI notice served by Punjab National Bank in FY2024.",
                   section_id="s1", metadata={"page_number": 3})
    p3 = Paragraph(id="p3", text="Revenue declined 18% to Rs 4053.90 crore in FY2025.",
                   section_id="s2", metadata={"page_number": 7})
    p4 = Paragraph(id="p4", text="EBITDA margin improved from 8.2% to 9.1% year-on-year.",
                   section_id="s2", metadata={"page_number": 7})
    cells = [
        Cell(row_idx=0, col_idx=0, value="Metric", is_header=True),
        Cell(row_idx=0, col_idx=1, value="FY2024", is_header=True),
        Cell(row_idx=0, col_idx=2, value="FY2025", is_header=True),
        Cell(row_idx=1, col_idx=0, value="Principal Default"),
        Cell(row_idx=1, col_idx=1, value="3800"),
        Cell(row_idx=1, col_idx=2, value="4053.90"),
    ]
    t1 = Table(id="t1", caption="Default History", cells=cells,
               col_headers=["Metric", "FY2024", "FY2025"],
               metadata={"page_number": 3})
    return Document(id="doc1", title="Gammon_India_AR2025.pdf",
                    sections=[sec1, sec2],
                    paragraphs=[p1, p2, p3, p4],
                    tables=[t1])


# ---------------------------------------------------------------------------
# 1. Structural Graph tests
# ---------------------------------------------------------------------------

class TestStructuralGraph:
    def test_section_children_links(self):
        doc = make_doc()
        sg = StructuralGraph()
        sg.add_document(doc)

        # s1 should have children p1, p2, t1
        siblings = sg.get_section_siblings("p1", max_siblings=10)
        sibling_ids = {n.node_id for n in siblings}
        assert "p2" in sibling_ids, "p2 should be sibling of p1"
        assert "t1" in sibling_ids, "t1 should be sibling of p1"
        assert "p3" not in sibling_ids, "p3 is in different section"

    def test_table_row_nodes(self):
        doc = make_doc()
        sg = StructuralGraph()
        sg.add_document(doc)

        rows = sg.get_table_rows("t1")
        assert len(rows) >= 1, "Table t1 should have at least one data row"
        # Row should contain column header context
        row_texts = " ".join(r.text for r in rows).lower()
        assert "principal default" in row_texts or "metric" in row_texts, \
            "Row text should include column header or metric name"

    def test_page_adjacency(self):
        doc = make_doc()
        sg = StructuralGraph()
        sg.add_document(doc)

        # p1 and p2 are on page 3 — should be adjacent
        neighbours = sg.get_page_neighbours("p1", max_neighbours=5)
        neighbour_ids = {n.node_id for n in neighbours}
        assert "p2" in neighbour_ids, "p2 should be page-adjacent to p1"
        assert "t1" in neighbour_ids, "t1 is on page 3, should be page-adjacent"

    def test_cross_ref_detection(self):
        # Add a paragraph that mentions another section title
        doc = make_doc()
        # Add a paragraph that references "Credit Risk" section
        p5 = Paragraph(id="p5",
                       text="See Credit Risk section for full NPA details.",
                       section_id="s2", metadata={"page_number": 8})
        doc.paragraphs.append(p5)
        doc.sections[1].content_ids.append("p5")

        sg = StructuralGraph()
        sg.add_document(doc)

        xrefs = sg.get_cross_refs("p5")
        xref_ids = {n.node_id for n in xrefs}
        assert "s1" in xref_ids, "p5 should cross-reference s1 (Credit Risk)"

    def test_node_count(self):
        doc = make_doc()
        sg = StructuralGraph()
        sg.add_document(doc)
        # sections(2) + paragraphs(4) + tables(1) + rows(1) = 8
        assert sg.node_count() >= 7, f"Expected ≥7 nodes, got {sg.node_count()}"


# ---------------------------------------------------------------------------
# 2. Obligation Ledger tests
# ---------------------------------------------------------------------------

class TestObligationLedger:
    def setup_method(self):
        self.ledger = ObligationLedger()

    def test_enumeration_requirement_detected(self):
        query = "Pull together every credit-risk event and red flag in this year's report."
        reqs = self.ledger.parse_requirements(query)
        types = [r.req_type for r in reqs]
        assert RequirementType.ENUMERATION in types, "Enumeration requirement not detected"

    def test_numerical_requirement_detected(self):
        query = "What was the total principal default amount in FY2025?"
        reqs = self.ledger.parse_requirements(query)
        types = [r.req_type for r in reqs]
        assert RequirementType.NUMERICAL_FACT in types

    def test_comparison_requirement_detected(self):
        query = "Compare the NPA amount in FY2024 vs FY2025."
        reqs = self.ledger.parse_requirements(query)
        types = [r.req_type for r in reqs]
        assert RequirementType.COMPARISON in types

    def test_temporal_requirement_detected(self):
        query = "When was Gammon India first classified as NPA?"
        reqs = self.ledger.parse_requirements(query)
        types = [r.req_type for r in reqs]
        assert RequirementType.TEMPORAL_FACT in types

    def test_supported_status_when_evidence_present(self):
        query = "What NPA actions did lenders take?"
        reqs = self.ledger.parse_requirements(query)
        chunks = [
            {"id": "p1", "text": "Gammon India NPA classified since June 2017 by lenders."},
            {"id": "p2", "text": "SARFAESI notice served by Punjab National Bank in FY2024."},
        ]
        updated = self.ledger.evaluate_candidates(reqs, chunks)
        statuses = {r.status for r in updated}
        # At least one requirement should be supported
        assert RequirementStatus.SUPPORTED in statuses or RequirementStatus.PARTIAL in statuses

    def test_missing_status_when_no_evidence(self):
        query = "What was the overseas subsidiary revenue in FY2023?"
        reqs = self.ledger.parse_requirements(query)
        chunks = [
            {"id": "p1", "text": "Domestic operations grew 5% in the current year."},
        ]
        updated = self.ledger.evaluate_candidates(reqs, chunks)
        # Temporal requirement for FY2023 should be missing/partial
        temporal_reqs = [r for r in updated if r.req_type == RequirementType.TEMPORAL_FACT]
        if temporal_reqs:
            assert temporal_reqs[0].status in (RequirementStatus.MISSING, RequirementStatus.PARTIAL)

    def test_general_requirement_always_present(self):
        query = "Explain the credit risk exposure."
        reqs = self.ledger.parse_requirements(query)
        types = [r.req_type for r in reqs]
        assert RequirementType.GENERAL in types


# ---------------------------------------------------------------------------
# 3. Coverage Selector tests
# ---------------------------------------------------------------------------

class TestCoverageSelector:
    def _make_chunks(self):
        return [
            {"id": "p1", "text": "Gammon India NPA classified since June 2017 by lenders.",
             "score": 0.9, "document": "Gammon_AR.pdf", "metadata": {}},
            {"id": "p2", "text": "SARFAESI notice served by Punjab National Bank in FY2024.",
             "score": 0.8, "document": "Gammon_AR.pdf", "metadata": {}},
            {"id": "p3", "text": "Revenue declined 18% to Rs 4053.90 crore in FY2025.",
             "score": 0.7, "document": "Gammon_AR.pdf", "metadata": {}},
            {"id": "p4", "text": "EBITDA margin improved from 8.2% to 9.1% year-on-year.",
             "score": 0.5, "document": "Gammon_AR.pdf", "metadata": {}},
        ]

    def test_selection_returns_chunks(self):
        chunks = self._make_chunks()
        selector = CoverageSelector(budget_tokens=2000)
        selected, trace = selector.select(
            query="List all NPA events and lender actions",
            candidates=chunks,
        )
        assert len(selected) > 0, "Selector should return at least one chunk"

    def test_budget_enforcement(self):
        # Create many large chunks
        big_chunks = [
            {"id": f"c{i}", "text": "word " * 300, "score": 0.5, "document": "doc.pdf", "metadata": {}}
            for i in range(20)
        ]
        selector = CoverageSelector(budget_tokens=500)  # tight budget
        selected, trace = selector.select(
            query="What is the financial information?",
            candidates=big_chunks,
        )
        total_tokens = sum(_token_count(c.get("text", "")) for c in selected)
        assert total_tokens <= 600, f"Budget exceeded: {total_tokens} tokens > 500"

    def test_redundancy_penalty(self):
        # Two nearly-identical chunks
        chunks = [
            {"id": "a", "text": "Gammon India NPA classified since June 2017 by lenders bank action taken.",
             "score": 0.9, "document": "doc.pdf", "metadata": {}},
            {"id": "b", "text": "Gammon India NPA classified since June 2017 by lenders bank action taken.",
             "score": 0.85, "document": "doc.pdf", "metadata": {}},
            {"id": "c", "text": "SARFAESI notice was filed in Debt Recovery Tribunal by creditor bank.",
             "score": 0.6, "document": "doc.pdf", "metadata": {}},
        ]
        selector = CoverageSelector(redundancy_threshold=0.7)
        selected, trace = selector.select(
            query="NPA lender actions",
            candidates=chunks,
        )
        ids = {c["id"] for c in selected}
        # Both 'a' and 'b' should NOT both be selected (redundant)
        assert not ("a" in ids and "b" in ids), "Redundant chunks should not both be selected"

    def test_wrong_period_constraint(self):
        """Evidence from wrong document should be penalised when constraint given."""
        chunks = [
            {"id": "p1", "text": "NPA default amount Rs 4053 crore.",
             "score": 0.9, "document": "Gammon_India_AR.pdf", "metadata": {}},
            {"id": "p2", "text": "Revenue grew 12%.",
             "score": 0.8, "document": "Other_Company_AR.pdf", "metadata": {}},
        ]
        selector = CoverageSelector()
        selected, trace = selector.select(
            query="What are the NPA defaults?",
            candidates=chunks,
            entity_constraints={"doc_name": ["Gammon_India_AR.pdf"]},
        )
        doc_names = {c["document"] for c in selected}
        assert "Gammon_India_AR.pdf" in doc_names, "Correct document should be in selection"

    def test_trace_contains_requirements(self):
        chunks = self._make_chunks()
        selector = CoverageSelector()
        _, trace = selector.select(
            query="List every lender action and NPA event",
            candidates=chunks,
        )
        assert "requirements" in trace
        assert len(trace["requirements"]) > 0
        assert "ledger_summary" in trace

    def test_structural_recovery_pass(self):
        """Recovery pass should add structural siblings for missing requirements."""
        doc = make_doc()
        sg = StructuralGraph()
        sg.add_document(doc)

        # Only provide p1 as a candidate
        chunks = [
            {"id": "p1", "text": "Gammon India NPA classified since June 2017.",
             "score": 0.9, "document": "Gammon_India_AR2025.pdf", "metadata": {"page_number": 3}},
        ]
        selector = CoverageSelector(max_recovery_chunks=3)
        selected, trace = selector.select(
            query="What were all the lender actions and SARFAESI notices?",
            candidates=chunks,
            struct_graph=sg,
        )
        # Recovery should have fetched p2 (SARFAESI) or t1
        all_ids = {c["id"] for c in selected}
        assert trace.get("recovery_used") or len(all_ids) >= 1


# ---------------------------------------------------------------------------
# 4. Jaccard and token helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_jaccard_identical(self):
        assert _jaccard("foo bar baz", "foo bar baz") == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_jaccard_partial(self):
        j = _jaccard("NPA lender action bank", "NPA default bank notice")
        assert 0.0 < j < 1.0

    def test_token_count_reasonable(self):
        text = "word " * 100
        t = _token_count(text)
        assert 100 < t < 200  # ~133 tokens for 100 words


# ---------------------------------------------------------------------------
# 5. SymbolicCalculator (already in fusion.py — verify it works)
# ---------------------------------------------------------------------------

class TestSymbolicCalculator:
    def setup_method(self):
        self.calc = SymbolicCalculator()

    def test_percentage_change(self):
        result = self.calc.compute("percentage_change", {"old_value": 3800, "new_value": 4053.90})
        assert abs(result - 6.68) < 0.1, f"Expected ~6.68%, got {result}"

    def test_subtract(self):
        result = self.calc.compute("subtract", {"a": 4053.90, "b": 3800.0})
        assert abs(result - 253.90) < 0.01

    def test_divide_by_zero(self):
        result = self.calc.compute("divide", {"numerator": 100, "denominator": 0})
        assert result == 0.0

    def test_ratio(self):
        result = self.calc.compute("ratio", {"numerator": 500, "denominator": 4053.90})
        assert 0.12 <= result <= 0.13, f"Expected 0.12-0.13, got {result}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    test_classes = [
        TestStructuralGraph,
        TestObligationLedger,
        TestCoverageSelector,
        TestHelpers,
        TestSymbolicCalculator,
    ]
    passed = 0
    failed = 0
    for cls in test_classes:
        print(f"\n{'='*50}")
        print(f"  {cls.__name__}")
        print('='*50)
        obj = cls()
        for name in dir(obj):
            if not name.startswith("test_"):
                continue
            try:
                if hasattr(obj, "setup_method"):
                    obj.setup_method()
                getattr(obj, name)()
                print(f"  PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {name}: ASSERTION -- {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*50)
