import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ..parsing.base import Document

class GraphBuilder:
    """
    Constructs a PyG graph from a parsed Document object using real sentence embeddings.
    
    Node Types:
    0: Paragraph (P)
    1: Section (S)
    2: Table (T)
    3: Cell (C)
    
    Edge Types:
    0: Semantic (sem) - calculated via cosine threshold
    1: Structural (struct) - explicit hierarchy (Section <-> Paragraph/Table <-> Cell)
    2: Cross-Reference (ref) - explicit mentions
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: Optional[int] = None, encoder: Optional[Any] = None):
        self.model_name = model_name
        self.encoder = encoder
        if self.encoder is None:
            self._load_encoder()
        self.embedding_dim = embedding_dim or (
            self.encoder.get_sentence_embedding_dimension() if self.encoder and hasattr(self.encoder, "get_sentence_embedding_dimension") else 384
        )

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"[GraphBuilder] Warning: could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic projection.")
            self.encoder = None

    def _encode(self, text: str) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(self.embedding_dim)
        if self.encoder is not None:
            emb = self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)
            return emb.detach().cpu()
        else:
            # Deterministic pseudo-embedding if model not available
            seed = sum(ord(c) for c in text[:100]) % (2**31)
            rng = np.random.default_rng(seed)
            return torch.tensor(rng.standard_normal(self.embedding_dim), dtype=torch.float32)

    def build_graph(self, doc: Document, semantic_similarity_threshold: float = 0.70) -> Data:
        node_features = []
        node_types = []
        node_ids_map = {}   # Map internal ID to node index
        node_text_map = {}  # Map node index to original text/representation
        current_idx = 0
        
        # 1. Create Nodes
        node_texts = []
        
        # Sections
        for sec in doc.sections:
            node_texts.append(sec.title or "")
            node_types.append(1) # Section
            node_ids_map[sec.id] = current_idx
            node_text_map[current_idx] = {
                "id": sec.id, "type": "Section", "text": sec.title, "metadata": sec.metadata
            }
            current_idx += 1
            
        # Paragraphs
        for p in doc.paragraphs:
            node_texts.append(p.text or "")
            node_types.append(0) # Paragraph
            node_ids_map[p.id] = current_idx
            node_text_map[current_idx] = {
                "id": p.id, "type": "Paragraph", "text": p.text, "metadata": p.metadata
            }
            current_idx += 1
            
        # Tables
        for table in doc.tables:
            table_summary = f"{table.caption} {' '.join(table.col_headers)}"
            node_texts.append(table_summary)
            node_types.append(2) # Table
            node_ids_map[table.id] = current_idx
            node_text_map[current_idx] = {
                "id": table.id, "type": "Table", "text": table_summary, "metadata": table.metadata
            }
            current_idx += 1
            
            # Cell nodes
            for cell in table.cells:
                cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                cell_text = str(cell.value or "")
                node_texts.append(cell_text)
                node_types.append(3) # Cell
                node_ids_map[cell_id] = current_idx
                node_text_map[current_idx] = {
                    "id": cell_id, "type": "Cell", "text": cell_text, 
                    "table_id": table.id, "row": cell.row_idx, "col": cell.col_idx
                }
                current_idx += 1
                
        # Batch encode features
        if node_texts:
            if self.encoder is not None:
                embs = self.encoder.encode(node_texts, batch_size=128, show_progress_bar=False, convert_to_tensor=True)
                x = embs.detach().cpu().to(torch.float32)
            else:
                x = torch.stack([self._encode(t) for t in node_texts])
            node_types_tensor = torch.tensor(node_types, dtype=torch.long)
        else:
            x = torch.empty((0, self.embedding_dim), dtype=torch.float32)
            node_types_tensor = torch.empty((0,), dtype=torch.long)
        
        # 2. Create Edges
        edge_indices = [[], []]
        edge_attrs = []
        
        # Structural Edges (Section <-> Content)
        for sec in doc.sections:
            if sec.id in node_ids_map:
                sec_idx = node_ids_map[sec.id]
                for content_id in sec.content_ids:
                    if content_id in node_ids_map:
                        content_idx = node_ids_map[content_id]
                        
                        # Parent -> Child
                        edge_indices[0].append(sec_idx)
                        edge_indices[1].append(content_idx)
                        edge_attrs.append(1) # struct
                        
                        # Child -> Parent
                        edge_indices[0].append(content_idx)
                        edge_indices[1].append(sec_idx)
                        edge_attrs.append(1)

        # Structural Edges (Table <-> Cell)
        for table in doc.tables:
            if table.id in node_ids_map:
                table_idx = node_ids_map[table.id]
                for cell in table.cells:
                    cell_id = f"{table.id}_r{cell.row_idx}_c{cell.col_idx}"
                    if cell_id in node_ids_map:
                        cell_idx = node_ids_map[cell_id]
                        
                        edge_indices[0].append(table_idx)
                        edge_indices[1].append(cell_idx)
                        edge_attrs.append(1) # struct
                        
                        edge_indices[0].append(cell_idx)
                        edge_indices[1].append(table_idx)
                        edge_attrs.append(1)

        # 3. Semantic Edges between Paragraphs and Tables (Cosine similarity threshold)
        if len(x) > 0 and semantic_similarity_threshold > 0:
            norm_x = F.normalize(x, p=2, dim=1)
            sim_matrix = torch.mm(norm_x, norm_x.t())
            
            # Select paragraph (0) and table (2) nodes
            para_or_table = (node_types_tensor == 0) | (node_types_tensor == 2)
            indices = torch.where(para_or_table)[0]
            
            for i in range(len(indices)):
                idx_i = indices[i].item()
                for j in range(i + 1, len(indices)):
                    idx_j = indices[j].item()
                    if sim_matrix[idx_i, idx_j] >= semantic_similarity_threshold:
                        edge_indices[0].append(idx_i)
                        edge_indices[1].append(idx_j)
                        edge_attrs.append(0) # sem
                        
                        edge_indices[0].append(idx_j)
                        edge_indices[1].append(idx_i)
                        edge_attrs.append(0)

        # Convert to tensor
        if edge_indices[0]:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
            
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_types=node_types_tensor)
        graph_data.node_ids_map = node_ids_map
        graph_data.node_text_map = node_text_map
        return graph_data
