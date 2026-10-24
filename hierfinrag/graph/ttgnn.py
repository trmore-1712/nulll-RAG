import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

class TTGNN(nn.Module):
    """
    Table-Text Graph Neural Network (TTGNN).
    
    Architecture:
    - Node Type Embeddings: Distinct embeddings for P, S, T, C
    - Edge Type Embeddings: Incorporating edge relations (sem, struct, ref)
    - Graph Attention Layers: Relational attention mechanism
    """
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=3, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Feature Transformation
        # Project input embeddings (e.g., from BERT/E5) to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Node Type Embeddings
        # 5 types: Paragraph, Section, Table, Cell, Global(optional)
        self.node_type_emb = nn.Embedding(5, hidden_dim)
        
        # 3. Edge Type Embeddings
        # 3 types: Semantic, Structural, Reference
        self.edge_type_emb = nn.Embedding(3, hidden_dim)
        
        # 4. GNN Layers (GATv2 with edge features)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim, # Critical: Use edge features in attention
                    add_self_loops=True,
                    concat=True
                )
            )
            
        # 5. Output Projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, edge_attr, node_types):
        """
        Args:
            x: Node features [N, input_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge type indices [E]
            node_types: Node type indices [N]
        """
        # A. Initial Embedding Fusion
        h = self.input_proj(x)
        h = h + self.node_type_emb(node_types)
        
        # B. Edge Embedding Lookup
        # Ensure edge_attr are indices for embedding lookup
        edge_embeddings = self.edge_type_emb(edge_attr)
        
        # C. Message Passing
        for layer in self.layers:
            # Residual connection
            h_in = h
            
            # GATv2 Layer
            h = layer(h, edge_index, edge_attr=edge_embeddings)
            
            # Activation & Dropout
            h = F.relu(h)
            h = self.dropout(h)
            
            # Residual add (if shapes match, GAT concat may change shape if not careful)
            # Here input is hidden_dim, output is heads * (hidden/heads) = hidden_dim
            h = h + h_in
            
        # D. Final Projection
        h = self.output_proj(h)
        
        return h
