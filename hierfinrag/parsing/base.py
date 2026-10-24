from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Cell:
    """Represents a single cell in a table."""
    row_idx: int
    col_idx: int
    value: str
    is_header: bool = False
    
@dataclass
class Table:
    """Represents a table structure."""
    id: str
    caption: str
    cells: List[Cell]
    row_headers: List[str] = field(default_factory=list)
    col_headers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_rows(self) -> int:
        return max([c.row_idx for c in self.cells]) + 1 if self.cells else 0
        
    @property
    def num_cols(self) -> int:
        return max([c.col_idx for c in self.cells]) + 1 if self.cells else 0

@dataclass
class Paragraph:
    """Represents a text block."""
    id: str
    text: str
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Section:
    """Represents a document section."""
    id: str
    title: str
    level: int
    content_ids: List[str] = field(default_factory=list) # IDs of paragraphs/tables in this section
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """Represents a fully parsed document."""
    id: str
    title: str
    sections: List[Section] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_paragraph(self, p_id: str) -> Optional[Paragraph]:
        for p in self.paragraphs:
            if p.id == p_id:
                return p
        return None
        
    def get_table(self, t_id: str) -> Optional[Table]:
        for t in self.tables:
            if t.id == t_id:
                return t
        return None

class DocumentParser:
    """Abstract base class for document parsers."""
    def parse(self, source: Any) -> Document:
        raise NotImplementedError
