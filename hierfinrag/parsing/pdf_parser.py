import os
import re
from typing import List, Dict, Any, Optional
import fitz # PyMuPDF
from .base import DocumentParser, Document, Section, Paragraph, Table, Cell

class PDFParser(DocumentParser):
    """
    High-performance PDF financial report parser using PyMuPDF (fitz).
    Parses hundreds of pages in seconds, extracting Sections, Paragraphs,
    and Tables with row/column structures.
    """
    def __init__(self, max_pages: Optional[int] = None):
        self.max_pages = max_pages

    def parse(self, source: str) -> Document:
        doc_filename = os.path.basename(source)
        doc_id = doc_filename.replace(".pdf", "")
        
        sections: List[Section] = []
        paragraphs: List[Paragraph] = []
        tables: List[Table] = []
        
        current_section = Section(
            id=f"{doc_id}_sec_root",
            title="Overview",
            level=1,
            content_ids=[],
            metadata={"document": doc_filename}
        )
        sections.append(current_section)
        
        p_counter = 0
        t_counter = 0
        s_counter = 1
        
        doc = fitz.open(source)
        total_pages = len(doc)
        pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
        
        for page_idx in range(pages_to_process):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_meta = {"page_number": page_num, "document": doc_filename}
            
            # 1. Extract structured tables
            try:
                tab_finder = page.find_tables()
                for t in tab_finder.tables:
                    raw_rows = t.extract()
                    if not raw_rows or len(raw_rows) < 2:
                        continue
                        
                    t_counter += 1
                    table_id = f"t_{doc_id}_p{page_num}_{t_counter}"
                    cells: List[Cell] = []
                    col_headers: List[str] = [str(h or "") for h in (t.header.names or [])]
                    row_headers: List[str] = []
                    
                    for r_idx, row in enumerate(raw_rows):
                        row_vals = [str(c or "").strip() for c in row]
                        if not any(row_vals):
                            continue
                        if r_idx > 0 and row_vals[0]:
                            row_headers.append(row_vals[0])
                            
                        for c_idx, val in enumerate(row_vals):
                            cells.append(Cell(
                                row_idx=r_idx,
                                col_idx=c_idx,
                                value=val,
                                is_header=(r_idx == 0 or c_idx == 0)
                            ))
                            
                    table_obj = Table(
                        id=table_id,
                        caption=f"Table at Page {page_num}",
                        cells=cells,
                        row_headers=row_headers,
                        col_headers=col_headers,
                        metadata={**page_meta, "section": current_section.title, "section_id": current_section.id}
                    )
                    tables.append(table_obj)
                    current_section.content_ids.append(table_id)
            except Exception:
                pass

            # 2. Extract text blocks
            blocks = page.get_text("blocks") # list of (x0, y0, x1, y1, text, block_no, block_type)
            for block in blocks:
                # block[6] == 0 means text
                if len(block) >= 5 and isinstance(block[4], str):
                    text = block[4].strip()
                    if not text or len(text) < 15:
                        continue
                        
                    # Heading detection heuristic
                    lines = text.split("\n")
                    first_line = lines[0].strip()
                    is_heading = (
                        len(first_line) < 90 and
                        (first_line.isupper() or re.match(r"^(\d+(\.\d+)*|[A-Z]\.|\bNote\s+\d+|\bSection\b)", first_line))
                        and len(lines) <= 3
                    )
                    
                    if is_heading:
                        s_counter += 1
                        sec_id = f"{doc_id}_sec_{s_counter}"
                        current_section = Section(
                            id=sec_id,
                            title=first_line,
                            level=2,
                            content_ids=[],
                            metadata=page_meta
                        )
                        sections.append(current_section)
                        
                    p_counter += 1
                    p_id = f"p_{doc_id}_{p_counter}"
                    para_obj = Paragraph(
                        id=p_id,
                        text=text.replace("\n", " "),
                        section_id=current_section.id,
                        metadata={
                            **page_meta,
                            "section": current_section.title,
                            "section_id": current_section.id,
                        }
                    )
                    paragraphs.append(para_obj)
                    current_section.content_ids.append(p_id)
                    
        return Document(
            id=doc_id,
            title=doc_filename,
            sections=sections,
            paragraphs=paragraphs,
            tables=tables,
            metadata={"source_file": source, "total_pages": total_pages}
        )

    @staticmethod
    def from_benchmark_chunks(doc_name: str, chunks: List[Dict[str, Any]]) -> Document:
        doc_id = doc_name.replace(".pdf", "")
        sections: List[Section] = []
        paragraphs: List[Paragraph] = []
        tables: List[Table] = []
        
        sec_map: Dict[str, Section] = {}
        
        for idx, chunk in enumerate(chunks):
            sec_name = chunk.get("section_name", f"Section_{idx+1}")
            content = chunk.get("content", "")
            meta = chunk.get("metadata", {})
            
            if sec_name not in sec_map:
                sec_obj = Section(
                    id=f"{doc_id}_{sec_name}",
                    title=sec_name,
                    level=1,
                    content_ids=[],
                    metadata=meta
                )
                sec_map[sec_name] = sec_obj
                sections.append(sec_obj)
            else:
                sec_obj = sec_map[sec_name]
                
            if content.strip().startswith("|") and "|" in content:
                table_id = f"t_{doc_id}_{idx}"
                rows = [r.strip().split("|")[1:-1] for r in content.strip().split("\n") if "|" in r and "---" not in r]
                cells = []
                col_headers = []
                row_headers = []
                
                for r_idx, row in enumerate(rows):
                    row_vals = [c.strip() for c in row]
                    if r_idx == 0:
                        col_headers = row_vals
                    elif row_vals and row_vals[0]:
                        row_headers.append(row_vals[0])
                    for c_idx, val in enumerate(row_vals):
                        cells.append(Cell(
                            row_idx=r_idx,
                            col_idx=c_idx,
                            value=val,
                            is_header=(r_idx == 0 or c_idx == 0)
                        ))
                
                t_obj = Table(
                    id=table_id,
                    caption=sec_name,
                    cells=cells,
                    row_headers=row_headers,
                    col_headers=col_headers,
                    metadata=meta
                )
                tables.append(t_obj)
                sec_obj.content_ids.append(table_id)
            else:
                p_id = f"p_{doc_id}_{idx}"
                p_obj = Paragraph(
                    id=p_id,
                    text=content,
                    section_id=sec_obj.id,
                    metadata=meta
                )
                paragraphs.append(p_obj)
                sec_obj.content_ids.append(p_id)
                
        return Document(
            id=doc_id,
            title=doc_name,
            sections=sections,
            paragraphs=paragraphs,
            tables=tables,
            metadata={"document": doc_name}
        )
