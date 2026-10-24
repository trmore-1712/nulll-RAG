"""
Structural Evidence Graph
==========================
A lightweight in-memory graph built from `GraphBuilder`-compatible structures
that are ALREADY in the parsed `Document` objects (Section.content_ids,
Table.cells with row/col headers, Paragraph.metadata['page_number']).

No GNN. No graph database. No new ML models.

Purpose: given a retrieved chunk (paragraph or table), quickly find:
  1. Its parent section and sibling paragraphs/tables (structural neighbours)
  2. Co-located evidence on the same page (page-adjacency)
  3. Table rows linked to their column headers (for numerical context)

These links power the "recovery pass" in CoverageSelector: if a requirement
is missing after ranking, we do one targeted lookup here before giving up.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from ..parsing.base import Document, Section, Paragraph, Table


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceLink:
    """A directed relationship between two evidence items."""
    source_id: str          # chunk / cell / section id
    target_id: str
    link_type: str          # "section_child", "table_cell", "page_adjacent",
                            # "header_row", "cross_ref"
    confidence: float = 1.0 # 1.0 = structurally certain, <1.0 = inferred
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceNode:
    """A single retrievable piece of evidence in the graph."""
    node_id: str
    node_type: str          # "paragraph", "table", "section", "cell"
    text: str
    doc_name: str
    page_number: Optional[int]
    section_id: Optional[str]
    section_title: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Populated after graph construction
    links_out: List[EvidenceLink] = field(default_factory=list)


class StructuralGraph:
    """
    Lightweight structural evidence graph over one or more Documents.

    Construction is O(N) in total evidence items.
    Queries are O(degree) lookups via adjacency dicts.

    Relationships established:
      - section_child:   Section → all Paragraphs and Tables it contains
                         (from Section.content_ids — structurally certain)
      - table_header:    Table → its column headers (from Table.col_headers)
      - table_cell:      Table → Cell rows (from Table.cells, header flagged)
      - page_adjacent:   Paragraphs on the same page, within 2 positions
                         (weak structural link, confidence=0.6)
      - cross_ref:       Paragraph text mentions another section/table by title
                         (heuristic, confidence=0.5)
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, EvidenceNode] = {}          # node_id → node
        self._section_children: Dict[str, List[str]] = defaultdict(list)
        self._page_index: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        self._table_cells: Dict[str, List[EvidenceNode]] = defaultdict(list)
        self._section_titles: Dict[str, str] = {}         # section_id → title
        self._built = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_document(self, doc: Document) -> None:
        """
        Ingest a parsed Document and populate all structural links.
        Call this for every document indexed in the collection.
        """
        doc_name = doc.title

        # 1. Register sections
        sec_by_id: Dict[str, Section] = {}
        for sec in doc.sections:
            self._section_titles[sec.id] = sec.title or ""
            sec_by_id[sec.id] = sec
            node = EvidenceNode(
                node_id=sec.id, node_type="section",
                text=sec.title or "", doc_name=doc_name,
                page_number=sec.metadata.get("page_number"),
                section_id=sec.id, section_title=sec.title,
                metadata=sec.metadata,
            )
            self.nodes[sec.id] = node

        # 2. Register paragraphs + section_child + page_adjacent links
        page_paragraphs: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        for para in doc.paragraphs:
            page_num = para.metadata.get("page_number")
            sec_id = para.section_id
            sec_title = self._section_titles.get(sec_id, "") if sec_id else ""
            node = EvidenceNode(
                node_id=para.id, node_type="paragraph",
                text=para.text, doc_name=doc_name,
                page_number=page_num,
                section_id=sec_id, section_title=sec_title,
                metadata=para.metadata,
            )
            self.nodes[para.id] = node
            if page_num:
                self._page_index[(doc_name, page_num)].append(para.id)

        # 3. Register tables + header/cell links
        for tbl in doc.tables:
            page_num = tbl.metadata.get("page_number")
            # Find parent section (search sections for this table in content_ids)
            parent_sec_id = None
            for sec in doc.sections:
                if tbl.id in sec.content_ids:
                    parent_sec_id = sec.id
                    break
            sec_title = self._section_titles.get(parent_sec_id, "") if parent_sec_id else ""

            tbl_text = f"{tbl.caption} | Cols: {', '.join(tbl.col_headers)}"
            tbl_node = EvidenceNode(
                node_id=tbl.id, node_type="table",
                text=tbl_text, doc_name=doc_name,
                page_number=page_num,
                section_id=parent_sec_id, section_title=sec_title,
                metadata=tbl.metadata,
            )
            self.nodes[tbl.id] = tbl_node
            if page_num:
                self._page_index[(doc_name, page_num)].append(tbl.id)

            # Row nodes (one per data row, combining row header + values)
            rows: Dict[int, List[str]] = defaultdict(list)
            header_row: Dict[int, str] = {}
            for cell in tbl.cells:
                if cell.row_idx == 0 or cell.is_header and cell.col_idx == 0:
                    header_row[cell.col_idx] = str(cell.value or "")
                rows[cell.row_idx].append(str(cell.value or ""))

            for row_idx, vals in rows.items():
                if row_idx == 0:
                    continue  # skip header row itself
                row_text = " | ".join(
                    f"{header_row.get(ci, str(ci))}: {v}"
                    for ci, v in enumerate(vals)
                )
                row_id = f"{tbl.id}_row{row_idx}"
                row_node = EvidenceNode(
                    node_id=row_id, node_type="table_row",
                    text=row_text, doc_name=doc_name,
                    page_number=page_num,
                    section_id=parent_sec_id, section_title=sec_title,
                    metadata={**tbl.metadata, "table_id": tbl.id, "row_idx": row_idx,
                               "col_headers": tbl.col_headers},
                )
                self.nodes[row_id] = row_node
                self._table_cells[tbl.id].append(row_node)
                # table_cell link: table → row
                tbl_node.links_out.append(EvidenceLink(
                    source_id=tbl.id, target_id=row_id,
                    link_type="table_row", confidence=1.0,
                    metadata={"col_headers": tbl.col_headers}
                ))

        # 4. Section → children links (from Section.content_ids)
        for sec in doc.sections:
            sec_node = self.nodes.get(sec.id)
            if not sec_node:
                continue
            for cid in sec.content_ids:
                if cid in self.nodes:
                    self._section_children[sec.id].append(cid)
                    sec_node.links_out.append(EvidenceLink(
                        source_id=sec.id, target_id=cid,
                        link_type="section_child", confidence=1.0,
                    ))
                    child_node = self.nodes[cid]
                    child_node.links_out.append(EvidenceLink(
                        source_id=cid, target_id=sec.id,
                        link_type="section_parent", confidence=1.0,
                    ))

        # 5. Page-adjacency links (same page, within 2 positions)
        for (dname, pnum), ids in self._page_index.items():
            for i, nid in enumerate(ids):
                node_i = self.nodes.get(nid)
                if not node_i:
                    continue
                for j in range(max(0, i - 2), min(len(ids), i + 3)):
                    if i == j:
                        continue
                    neighbour_id = ids[j]
                    node_i.links_out.append(EvidenceLink(
                        source_id=nid, target_id=neighbour_id,
                        link_type="page_adjacent", confidence=0.6,
                        metadata={"page": pnum, "distance": abs(i - j)},
                    ))

        # 6. Cross-reference links: paragraph mentions another section title
        section_title_patterns = {
            sec.id: re.compile(re.escape(sec.title.strip()), re.IGNORECASE)
            for sec in doc.sections if sec.title and len(sec.title) > 6
        }
        for para in doc.paragraphs:
            para_node = self.nodes.get(para.id)
            if not para_node:
                continue
            for sec_id, pattern in section_title_patterns.items():
                if sec_id == para.section_id:
                    continue
                if pattern.search(para.text):
                    para_node.links_out.append(EvidenceLink(
                        source_id=para.id, target_id=sec_id,
                        link_type="cross_ref", confidence=0.5,
                        metadata={"matched_section": self._section_titles.get(sec_id, "")},
                    ))

        self._built = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_section_siblings(
        self, chunk_id: str, max_siblings: int = 6
    ) -> List[EvidenceNode]:
        """
        Return up to `max_siblings` other paragraphs/tables in the same section.
        Used for recovery pass when a section is partially covered.
        """
        node = self.nodes.get(chunk_id)
        if not node or not node.section_id:
            return []
        siblings = self._section_children.get(node.section_id, [])
        result = []
        for sid in siblings:
            if sid == chunk_id:
                continue
            snode = self.nodes.get(sid)
            if snode and snode.node_type in ("paragraph", "table"):
                result.append(snode)
            if len(result) >= max_siblings:
                break
        return result

    def get_table_rows(self, table_id: str) -> List[EvidenceNode]:
        """Return all row nodes for a table (with column header context)."""
        return self._table_cells.get(table_id, [])

    def get_page_neighbours(
        self, chunk_id: str, max_neighbours: int = 4
    ) -> List[EvidenceNode]:
        """Return paragraphs/tables on the same page, ordered by proximity."""
        node = self.nodes.get(chunk_id)
        if not node or not node.page_number:
            return []
        page_ids = self._page_index.get((node.doc_name, node.page_number), [])
        result = []
        for pid in page_ids:
            if pid == chunk_id:
                continue
            pnode = self.nodes.get(pid)
            if pnode:
                result.append(pnode)
            if len(result) >= max_neighbours:
                break
        return result

    def get_cross_refs(self, chunk_id: str) -> List[EvidenceNode]:
        """Return nodes that `chunk_id` cross-references."""
        node = self.nodes.get(chunk_id)
        if not node:
            return []
        result = []
        for link in node.links_out:
            if link.link_type == "cross_ref":
                target = self.nodes.get(link.target_id)
                if target:
                    result.append(target)
        return result

    def node_count(self) -> int:
        return len(self.nodes)
