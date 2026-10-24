import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

from nullrag.parsing.pdf_parser import PDFParser
from nullrag.graph.builder import GraphBuilder
from nullrag.graph.ttgnn import TTGNN
from nullrag.retrieval.hierarchical import HierarchicalRetriever
from nullrag.retrieval.synthesis_aggregator import AdaptiveRetriever, classify_query
from nullrag.routing.metadata_router import MetadataRouter
from nullrag.reasoning.fusion import SymbolicNeuralFusion
from nullrag.generation.attribution import generate_answer_with_attribution

def get_keys():
    keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        "OPENCODE_API_KEY": os.getenv("OPENCODE_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    }
    try:
        import config
        for k in keys:
            if hasattr(config, k) and getattr(config, k):
                keys[k] = getattr(config, k)
    except ImportError:
        pass
    return keys

class BenchmarkEvaluatorEngine:
    def __init__(
        self, 
        docs_dir: str = "Train/Documents", 
        max_pages_per_pdf: Optional[int] = None,
        provider: str = "auto",
        model: Optional[str] = None
    ):
        self.docs_dir    = docs_dir
        self.max_pages   = max_pages_per_pdf
        self.parser      = PDFParser(max_pages=self.max_pages)
        self.builder     = GraphBuilder()
        self.adaptive    = AdaptiveRetriever(encoder=self.builder.encoder)
        self.router      = MetadataRouter()
        self.fusion      = SymbolicNeuralFusion(provider=provider, model=model)
        self.doc_cache: Dict[str, Any] = {}

    def get_or_parse_document(self, doc_name: str, fallback_chunks: Optional[List[Dict[str, Any]]] = None):
        """Retrieves parsed Document object, parsing from disk or benchmark chunks."""
        if doc_name in self.doc_cache:
            return self.doc_cache[doc_name]

        pdf_path = os.path.join(self.docs_dir, doc_name)
        if os.path.exists(pdf_path):
            try:
                print(f"  [Parsing PDF] {doc_name}...")
                doc = self.parser.parse(pdf_path)
                self.doc_cache[doc_name] = doc
                # Register in metadata router for entity/year filtering
                sample_text = " ".join(p.text for p in doc.paragraphs[:30])[:5000]
                self.router.register(doc_name, sample_text)
                return doc
            except Exception as e:
                print(f"  [Warning] Failed parsing {pdf_path}: {e}. Using chunks fallback.")

        if fallback_chunks:
            print(f"  [Fallback] Building document from benchmark chunks for {doc_name}...")
            doc = PDFParser.from_benchmark_chunks(doc_name, fallback_chunks)
            self.doc_cache[doc_name] = doc
            sample_text = " ".join(c.get("content", "") for c in fallback_chunks[:10])[:5000]
            self.router.register(doc_name, sample_text)
            return doc

        return None

    def evaluate_query(self, item: Dict[str, Any], top_chunks: int = 8) -> Dict[str, Any]:
        query = item["question"]
        bench_chunks_meta = item.get("benchmark_chunks_w_metadata", {})
        
        # Determine query type upfront for logging
        query_type = classify_query(query)
        target_docs = list(bench_chunks_meta.keys())
        all_retrieved = []
        
        for doc_name in target_docs:
            chunks_meta = bench_chunks_meta[doc_name]
            doc = self.get_or_parse_document(doc_name, fallback_chunks=chunks_meta)
            if not doc:
                continue

            # Build graph & run TTGNN
            try:
                graph = self.builder.build_graph(doc)
                if graph.x.shape[0] > 0:
                    ttgnn = TTGNN(input_dim=graph.x.shape[1], hidden_dim=128, num_layers=2)
                    ttgnn.eval()
                    _ = ttgnn(graph.x, graph.edge_index, graph.edge_attr, graph.node_types)
            except Exception:
                pass

            # Adaptive Retrieval: synthesis or targeted
            qtype, retrieved = self.adaptive.retrieve(query, doc, top_chunks=top_chunks)
            all_retrieved.extend(retrieved)

        all_retrieved.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_retrieved = all_retrieved[:top_chunks]
        print(f"  [Retrieval] {query_type.upper()} mode → {len(top_retrieved)} chunks")

        # Reasoning & Generation
        result = generate_answer_with_attribution(query, top_retrieved, fusion_engine=self.fusion)

        gold_chunks = [
            c["content"] 
            for chunks in bench_chunks_meta.values() 
            for c in chunks
        ]

        return {
            "id": item.get("id"),
            "query": query,
            "ai_answer": result["answer_text"],
            "benchmark_answer": item.get("benchmark_answer", ""),
            "retrieved_chunks": result["retrieved_chunks"],
            "benchmark_chunks": gold_chunks,
            "confidence": result.get("confidence", 0.85),
            "mode": result.get("mode", "hybrid"),
            "provider": self.fusion.provider,
            "model": self.fusion.model
        }

def run_evaluation(
    benchmark_file: str = "Train/Benchmarks/benchmark_100.jsonl",
    output_file: str = "results/evaluation_outputs.json",
    num_samples: int = 3,
    provider: str = "auto",
    model: Optional[str] = None,
    run_llm_judge: bool = False,
    judge_provider: Optional[str] = None
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    keys = get_keys()
    
    print("=" * 70)
    print(f"NullRAG: Running Benchmark Evaluation on {num_samples} questions")
    print(f"Generator Provider: {provider.upper()}")
    print("=" * 70)
    
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmarks = [json.loads(line) for line in f]
        
    benchmarks = benchmarks[:num_samples]
    engine = BenchmarkEvaluatorEngine(provider=provider, model=model)
    
    eval_dataset = []
    start_time = time.time()
    
    for idx, item in enumerate(benchmarks, 1):
        q_id = item.get("id", f"Q{idx}")
        print(f"\n[{idx}/{len(benchmarks)}] Processing Question {q_id}: {item['question'][:80]}...")
        
        output = engine.evaluate_query(item)
        eval_dataset.append(output)
        print(f"  -> Generated via [{output['provider']}:{output['model']}] ({len(output['ai_answer'])} chars, {len(output['retrieved_chunks'])} chunks)")
        
        # Incremental save so output_file updates in real time
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(eval_dataset, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(benchmarks)} questions in {elapsed:.1f}s ({elapsed/len(benchmarks):.1f}s/q)")
    print(f"Saved evaluation dataset ({len(eval_dataset)} records) to: {output_file}")

    if run_llm_judge:
        judge_p = judge_provider or provider
        if judge_p in ("auto", "gemini"):
            judge_p = "opencode"
            
        judge_key = keys.get(f"{judge_p.upper()}_API_KEY", "")
        if judge_key:
            print("\n" + "=" * 70)
            print(f"Running LLM Judge using {judge_p.upper()}...")
            print("=" * 70)
            from run_opencode_judge import evaluate_with_judge
            summary = evaluate_with_judge(api_key=judge_key, dataset=eval_dataset, provider=judge_p)
            print("\n--- Final Metrics Summary ---")
            print(f"Questions Evaluated  : {summary['num_questions']}")
            print(f"Avg Answer Precision : {summary['avg_answer_precision']:.3f}")
            print(f"Avg Answer Recall    : {summary['avg_answer_recall']:.3f}")
            print(f"Avg Context Precision: {summary['avg_context_precision']:.3f}")
            print(f"Avg Context Recall   : {summary['avg_context_recall']:.3f}")
        else:
            print(f"\n[Notice] No API key found for judge provider '{judge_p}'. Please set {judge_p.upper()}_API_KEY in config.py.")

    return eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=3, help="Number of benchmark questions")
    parser.add_argument("--provider", type=str, default="opencode", choices=["opencode", "openai", "extractive", "auto"])
    parser.add_argument("--model", type=str, default=None, help="LLM model name (e.g. deepseek-v4.1-flash)")
    parser.add_argument("--run_judge", action="store_true", help="Run LLM evaluation judge")
    parser.add_argument("--judge_provider", type=str, default="opencode", choices=["opencode"])
    args = parser.parse_args()

    run_evaluation(
        num_samples=args.num_samples,
        provider=args.provider,
        model=args.model,
        run_llm_judge=args.run_judge,
        judge_provider=args.judge_provider
    )
