import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from ..parsing.base import Document, Section, Paragraph, Table, Cell

class HierarchicalRetriever:
    """
    Two-stage Hierarchical Retriever:
    - Level 1: Dense Section retrieval via cosine similarity.
    - Level 2: Hybrid Paragraph & Table retrieval using BM25 + Dense embeddings
               fused with Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, encoder=None, model_name="all-MiniLM-L6-v2"):
        if encoder is not None:
            self.encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        embs = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embs.detach().cpu()

    def level1_section_retrieval(
        self, 
        query: str, 
        sections: List[Section], 
        top_k: int = 15
    ) -> List[Section]:
        """
        Retrieves top-k most relevant sections for a query using section titles & descriptions.
        """
        if not sections:
            return []
        if len(sections) <= top_k:
            return sections

        section_texts = [f"{s.title}" for s in sections]
        query_emb = self.encode([query]) # [1, d]
        sec_embs = self.encode(section_texts) # [N, d]

        cos_sim = F.cosine_similarity(query_emb, sec_embs).numpy()
        top_indices = np.argsort(cos_sim)[-top_k:][::-1]
        return [sections[i] for i in top_indices]

    def level2_hybrid_retrieval(
        self,
        query: str,
        candidate_items: List[Dict[str, Any]],
        top_k: int = 8,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining BM25 (lexical) and Dense cosine similarity (semantic)
        via Reciprocal Rank Fusion (RRF).
        
        candidate_items: list of dicts with at least 'text' and 'id'.
        """
        if not candidate_items:
            return []
        if len(candidate_items) <= top_k:
            return candidate_items

        corpus_texts = [item.get("text", "") for item in candidate_items]
        
        # 1. Sparse BM25
        tokenized_corpus = [
            re.findall(r"\w+", text.lower()) for text in corpus_texts
        ]
        # Avoid empty tokens
        tokenized_corpus = [t if t else ["empty"] for t in tokenized_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = re.findall(r"\w+", query.lower()) or ["empty"]
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {idx: rank for rank, idx in enumerate(bm25_ranking)}

        # 2. Dense Similarity
        query_emb = self.encode([query])
        corpus_embs = self.encode(corpus_texts)
        dense_scores = F.cosine_similarity(query_emb, corpus_embs).numpy()
        dense_ranking = np.argsort(dense_scores)[::-1]
        dense_ranks = {idx: rank for rank, idx in enumerate(dense_ranking)}

        # 3. Reciprocal Rank Fusion
        rrf_scores = {}
        for idx in range(len(candidate_items)):
            r_bm25 = bm25_ranks[idx]
            r_dense = dense_ranks[idx]
            rrf_scores[idx] = (1.0 / (rrf_k + r_bm25)) + (1.0 / (rrf_k + r_dense))

        top_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            res_item = dict(candidate_items[idx])
            res_item["score"] = float(rrf_scores[idx])
            results.append(res_item)
        return results

    def retrieve_for_document(
        self, 
        query: str, 
        doc: Document, 
        top_sections: int = 15, 
        top_chunks: int = 20
    ) -> List[Dict[str, Any]]:
        """
        End-to-end two-stage retrieval on a parsed Document.
        """
        # Step 1: Filter to top sections if document has multiple sections
        if len(doc.sections) > top_sections:
            selected_sections = self.level1_section_retrieval(query, doc.sections, top_k=top_sections)
            allowed_content_ids = set()
            for s in selected_sections:
                allowed_content_ids.update(s.content_ids)
        else:
            allowed_content_ids = None

        # Step 2: Build candidate items from paragraphs and tables
        candidates = []
        for p in doc.paragraphs:
            if allowed_content_ids is None or p.id in allowed_content_ids:
                candidates.append({
                    "id": p.id,
                    "type": "paragraph",
                    "text": p.text,
                    "metadata": p.metadata,
                    "document": doc.title
                })
                
        for t in doc.tables:
            if allowed_content_ids is None or t.id in allowed_content_ids:
                # Format table content as readable markdown
                table_str = self._format_table_markdown(t)
                candidates.append({
                    "id": t.id,
                    "type": "table",
                    "text": table_str,
                    "metadata": t.metadata,
                    "document": doc.title
                })

        # Fallback if section filtering was too aggressive
        if not candidates:
            for p in doc.paragraphs:
                candidates.append({
                    "id": p.id, "type": "paragraph", "text": p.text, "metadata": p.metadata, "document": doc.title
                })
            for t in doc.tables:
                candidates.append({
                    "id": t.id, "type": "table", "text": self._format_table_markdown(t), "metadata": t.metadata, "document": doc.title
                })

        # Step 3: Run Level 2 Hybrid Retrieval
        return self.level2_hybrid_retrieval(query, candidates, top_k=top_chunks)

    def _format_table_markdown(self, table: Table) -> str:
        if not table.cells:
            return table.caption or "Empty Table"
        
        num_rows = table.num_rows
        num_cols = table.num_cols
        grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in table.cells:
            if cell.row_idx < num_rows and cell.col_idx < num_cols:
                grid[cell.row_idx][cell.col_idx] = cell.value

        md_lines = []
        if table.caption:
            md_lines.append(f"### {table.caption}")
        for r_idx, row in enumerate(grid):
            md_lines.append("| " + " | ".join(row) + " |")
            if r_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
        return "\n".join(md_lines)
