import json
from typing import Any, Dict, List
from .base import DocumentParser, Document, Section, Paragraph, Table, Cell

class JSONParser(DocumentParser):
    """
    Parses a document from a JSON file.
    Expected JSON structure:
    {
        "id": "doc_id",
        "title": "Document Title",
        "sections": [
            {
                "id": "sec_1",
                "title": "Section Title",
                "level": 1,
                "content_ids": ["p_1", "t_1"]
            }
        ],
        "paragraphs": [
            {
                "id": "p_1",
                "text": "Paragraph text...",
                "section_id": "sec_1"
            }
        ],
        "tables": [
            {
                "id": "t_1",
                "caption": "Table Caption",
                "cells": [
                    {"row": 0, "col": 0, "value": "Header", "is_header": true},
                    {"row": 1, "col": 0, "value": "Value", "is_header": false}
                ]
            }
        ]
    }
    """
    def parse(self, source: str) -> Document:
        """
        Args:
            source: Path to the JSON file.
        """
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return self._parse_dict(data)
        
    def _parse_dict(self, data: Dict[str, Any]) -> Document:
        # Parse Paragraphs
        paragraphs = []
        for p_data in data.get("paragraphs", []):
            paragraphs.append(Paragraph(
                id=p_data["id"],
                text=p_data["text"],
                section_id=p_data.get("section_id")
            ))
            
        # Parse Tables
        tables = []
        for t_data in data.get("tables", []):
            cells = []
            for c_data in t_data.get("cells", []):
                cells.append(Cell(
                    row_idx=c_data["row"],
                    col_idx=c_data["col"],
                    value=str(c_data["value"]),
                    is_header=c_data.get("is_header", False)
                ))
            
            tables.append(Table(
                id=t_data["id"],
                caption=t_data.get("caption", ""),
                cells=cells,
                row_headers=t_data.get("row_headers", []),
                col_headers=t_data.get("col_headers", [])
            ))
            
        # Parse Sections
        sections = []
        for s_data in data.get("sections", []):
            sections.append(Section(
                id=s_data["id"],
                title=s_data["title"],
                level=s_data.get("level", 1),
                content_ids=s_data.get("content_ids", [])
            ))
            
        return Document(
            id=data["id"],
            title=data.get("title", ""),
            sections=sections,
            paragraphs=paragraphs,
            tables=tables
        )
