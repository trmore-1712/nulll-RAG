import os
import sys
import json
import argparse
import torch

from nullrag.parsing.pdf_parser import PDFParser
from nullrag.graph.builder import GraphBuilder
from nullrag.graph.ttgnn import TTGNN
from nullrag.retrieval.hierarchical import HierarchicalRetriever
from nullrag.reasoning.fusion import SymbolicNeuralFusion
from nullrag.generation.attribution import generate_answer_with_attribution

def run_pdf_pipeline(pdf_path: str, query: str, max_pages: int = 15, provider: str = "auto", model: str = None):
    print("=" * 60)
    print(f"NullRAG Pipeline: Processing PDF '{os.path.basename(pdf_path)}'")
    print("=" * 60)
    
    # 1. Parsing Phase
    print(f"\n[Phase 1] Parsing PDF document (up to {max_pages} pages)...")
    parser = PDFParser(max_pages=max_pages)
    doc = parser.parse(pdf_path)
    print(f" -> Parsed: {doc.title}")
    print(f"    Sections:   {len(doc.sections)}")
    print(f"    Paragraphs: {len(doc.paragraphs)}")
    print(f"    Tables:     {len(doc.tables)}")

    # 2. Graph Construction Phase
    print("\n[Phase 2] Building Heterogeneous Table-Text Graph...")
    builder = GraphBuilder()
    graph = builder.build_graph(doc)
    print(f" -> Graph Nodes: {graph.x.shape[0]} (embedding dim: {graph.x.shape[1]})")
    print(f"    Graph Edges: {graph.edge_index.shape[1]}")

    # 3. TTGNN Relational Message Passing
    print("\n[Phase 3] Running TTGNN (Relational Graph Attention)...")
    ttgnn = TTGNN(
        input_dim=graph.x.shape[1], 
        hidden_dim=128, 
        num_layers=2, 
        num_heads=4
    )
    ttgnn.eval()
    with torch.no_grad():
        node_embeddings = ttgnn(graph.x, graph.edge_index, graph.edge_attr, graph.node_types)
    print(f" -> TTGNN contextual embeddings generated: {node_embeddings.shape}")

    # 4. Hierarchical Hybrid Retrieval
    print(f"\n[Phase 4] Hierarchical Retrieval for Query:\n    \"{query}\"")
    retriever = HierarchicalRetriever(encoder=builder.encoder)
    retrieved_contexts = retriever.retrieve_for_document(query, doc, top_sections=5, top_chunks=6)
    print(f" -> Retrieved {len(retrieved_contexts)} candidate contexts (Paragraphs / Tables)")
    for i, c in enumerate(retrieved_contexts[:3], 1):
        snippet = c.get("text", "")[:120].replace("\n", " ")
        print(f"    [{i}] (Type: {c.get('type')}, Score: {c.get('score', 0):.4f}) {snippet}...")

    # 5. Symbolic-Neural Fusion & Generation
    print("\n[Phase 5] Symbolic-Neural Fusion Reasoning & Answer Attribution...")
    fusion_engine = SymbolicNeuralFusion(provider=provider, model=model)
    result = generate_answer_with_attribution(query, retrieved_contexts, fusion_engine=fusion_engine)
    
    print("\n" + "=" * 60)
    print(f"FINAL ANSWER (Provider: {result.get('provider', provider).upper()}, Mode: {result['mode'].upper()}, Confidence: {result['confidence']:.2f}):")
    print("=" * 60)
    print(result["answer_text"])
    print("=" * 60)
    return result

def main():
    parser = argparse.ArgumentParser(description="NullRAG End-to-End Pipeline")
    parser.add_argument("--pdf", type=str, default="Train/Documents/fact-sheet-SUMMARY.pdf", help="Path to PDF file")
    parser.add_argument("--query", type=str, default="What are the key capital requirements and risk-based metrics summarized?", help="Question")
    parser.add_argument("--max_pages", type=int, default=10, help="Max pages to parse")
    parser.add_argument("--provider", type=str, default="opencode", choices=["opencode", "openai", "extractive", "auto"])
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. deepseek-v4.1-flash)")
    args = parser.parse_args()

    run_pdf_pipeline(args.pdf, args.query, max_pages=args.max_pages, provider=args.provider, model=args.model)

if __name__ == "__main__":
    main()
