"""
run_api_benchmark_eval.py
=========================
Tests the NullRAG REST APIs against the Train dataset:
  1. Indexes documents via POST /index (from train_documents.zip)
  2. Queries the collection via POST /query for questions in Train/Benchmarks/benchmark_100.jsonl
  3. Formats outputs to match Train/Evaluation/Evaluation.ipynb schema:
       - query
       - ai_answer
       - benchmark_answer
       - retrieved_chunks
       - benchmark_chunks
  4. Saves outputs to results/evaluation_outputs.json
  5. Optionally computes Answer Precision/Recall and Context Precision/Recall
     using OpenCode LLM-as-a-judge (the exact evaluator in Evaluation.ipynb).

Usage:
  # Test with first 5 questions (fast run):
  python run_api_benchmark_eval.py --limit 5

  # Run full benchmark evaluation with judge metrics:
  python run_api_benchmark_eval.py --limit 10 --eval

  # Skip re-indexing if already indexed:
  python run_api_benchmark_eval.py --skip-index --limit 5
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_opencode_key() -> str:
    key = os.getenv("OPENCODE_API_KEY", "")
    if not key:
        try:
            import config
            key = getattr(config, "OPENCODE_API_KEY", "")
        except ImportError:
            pass
    if not key:
        key = "oc_sk_30348db96428_nsJ5ZQztyT3WVQ5ZHohPEjnpqhUt0KdJ"
    return key


class OpenCodeClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4.1-flash",
        base_url: str = "https://opencode.ai/inference/openai/v1/chat/completions"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 10000,
        temperature: float = 0,
        max_retries: int = 4
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError(f"Empty content, finish_reason={data['choices'][0].get('finish_reason')}")
                return self._safe_json_parse(content)
            except Exception as e:
                last_error = e
                wait = 3.0 * (attempt + 1)
                print(f"    [Retry {attempt+1}/{max_retries}] Judge call failed: {e}. Waiting {wait:.0f}s...")
                time.sleep(wait)
        raise RuntimeError(f"OpenCode call failed after {max_retries} attempts: {last_error}")

    @staticmethod
    def _safe_json_parse(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse model output as JSON: {text[:500]}") from e


class RAGMetricsEvaluator:
    def __init__(self, client: OpenCodeClient):
        self.client = client

    def _decompose_points(self, text: str) -> list:
        if not text or not text.strip():
            return []
        system = (
            "You are a strict financial evaluator. Decompose the following text into "
            "independent, atomic, factual points / claims. Each point should contain "
            "exactly one verifiable fact, metric, or assertion. Respond only with JSON."
        )
        user = (
            "Break down this answer into atomic factual points. If the answer states that information "
            "is not found, treat that as a point. If the answer is empty or contains no checkable claims, "
            "return an empty list.\n\n"
            f"Answer:\n{text}\n\n"
            'Respond as JSON exactly in this shape: {"points": ["point 1", "point 2", ...]}'
        )
        try:
            result = self.client.call_structured(system, user)
            return result.get("points", []) or []
        except Exception as e:
            print(f"    [Warning] Decomposition error: {e}")
            return []

    def answer_precision_recall(self, ai_answer: str, benchmark_answer: str) -> tuple:
        ai_points = self._decompose_points(ai_answer)
        bench_points = self._decompose_points(benchmark_answer)
        if not ai_points and not bench_points:
            return 1.0, 1.0, {"ai_points": [], "benchmark_points": [], "matched_ids": []}
        if not ai_points:
            return 0.0, 0.0, {"ai_points": [], "benchmark_points": bench_points, "matched_ids": []}
        if not bench_points:
            return 0.0, 0.0, {"ai_points": ai_points, "benchmark_points": [], "matched_ids": []}

        system = (
            "You are an expert evaluator matching generated answer points against gold benchmark points. "
            "Respond only with JSON."
        )
        user = (
            f"Benchmark points:\n{json.dumps(bench_points, indent=2)}\n\n"
            f"Generated points:\n{json.dumps(ai_points, indent=2)}\n\n"
            "For each generated point, indicate which benchmark point index (0-based) it matches, "
            "or null if it is an extra/unsupported claim.\n"
            'Respond as JSON: {"matches": [{"ai_id": 0, "matched_benchmark_id": 0}]}'
        )
        try:
            res = self.client.call_structured(system, user)
            matches = res.get("matches", [])
            matched_ai = sum(1 for m in matches if m.get("matched_benchmark_id") is not None)
            matched_bench = len({m["matched_benchmark_id"] for m in matches if m.get("matched_benchmark_id") is not None})
            precision = matched_ai / len(ai_points) if ai_points else 0.0
            recall = matched_bench / len(bench_points) if bench_points else 0.0
            return precision, recall, {"ai_points": ai_points, "benchmark_points": bench_points, "matches": matches}
        except Exception as e:
            print(f"    [Warning] Match error: {e}")
            return 0.0, 0.0, {}

    def context_precision_recall(self, query: str, retrieved_chunks: list, benchmark_chunk_count: int) -> tuple:
        if not retrieved_chunks:
            return 0.0, 0.0
        system = "You are a strict evaluator judging whether retrieved text chunks are relevant to a user's query. Respond only with JSON."
        numbered = [{"id": i, "text": c[:1500]} for i, c in enumerate(retrieved_chunks)]
        user = (
            f"User query:\n{query}\n\n"
            f"Retrieved chunks:\n{json.dumps(numbered, indent=2)}\n\n"
            "For each chunk, decide if it is genuinely relevant and useful for answering the query.\n"
            'Respond as JSON: {"verdicts": [{"id": 0, "relevant": true}, {"id": 1, "relevant": false}]}'
        )
        try:
            result = self.client.call_structured(system, user)
            verdicts = result.get("verdicts", []) or []
            relevant = sum(1 for v in verdicts if v.get("relevant") is True)
            precision = relevant / len(retrieved_chunks)
            recall = relevant / benchmark_chunk_count if benchmark_chunk_count else 0.0
            return precision, recall
        except Exception as e:
            print(f"    [Warning] Context eval error: {e}")
            return 0.0, 0.0


def check_api_health(base_url: str):
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            print(f"[API] Server healthy at {base_url}: {r.json()}")
            return True
    except Exception as e:
        print(f"[API] Server not reachable at {base_url}: {e}")
    return False


def ensure_train_zip(zip_path: str, docs_dir: str = "Train/Documents") -> str:
    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[ZIP] Using existing archive: {zip_path} ({size_mb:.2f} MB)")
        return zip_path

    import zipfile
    print(f"[ZIP] Creating archive {zip_path} from {docs_dir}...")
    files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(os.path.join(docs_dir, f), f)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[ZIP] Created {zip_path} with {len(files)} PDFs ({size_mb:.2f} MB)")
    return zip_path


def index_collection(base_url: str, collection_name: str, zip_path: str):
    print(f"\n[POST /index] Ingesting {zip_path} into collection '{collection_name}'...")
    t0 = time.time()
    with open(zip_path, "rb") as f:
        files = {"file": (os.path.basename(zip_path), f, "application/zip")}
        data = {"collection_name": collection_name}
        resp = requests.post(f"{base_url}/index", files=files, data=data, timeout=900)
    
    if resp.status_code != 200:
        raise RuntimeError(f"Indexing failed [{resp.status_code}]: {resp.text}")

    res = resp.json()
    elapsed = time.time() - t0
    print(f"[POST /index] Success in {elapsed:.1f}s!")
    print(f"  Docs: {res.get('num_documents')}, Chunks: {res.get('num_chunks')}")
    return res


def run_benchmark(
    base_url: str,
    collection_name: str,
    benchmark_file: str = "Train/Benchmarks/benchmark_100.jsonl",
    limit: int = 5,
    top_chunks: int = 30,
    provider: str = "opencode",
    output_path: str = "results/evaluation_outputs.json",
    run_eval: bool = False
):
    print(f"\n[Benchmark] Loading questions from {benchmark_file}...")
    items = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if limit > 0:
        items = items[:limit]

    print(f"[Benchmark] Evaluating {len(items)} queries against collection '{collection_name}'...")
    dataset_records = []

    evaluator = None
    if run_eval:
        key = get_opencode_key()
        evaluator = RAGMetricsEvaluator(OpenCodeClient(api_key=key))
        print("[Eval] LLM Judge initialized (deepseek-v4.1-flash).")

    eval_metrics_list = []

    for idx, item in enumerate(items, 1):
        q_id = item.get("id", f"q{idx}")
        query = item["question"]
        gold_answer = item.get("benchmark_answer", "")
        chunks_meta = item.get("benchmark_chunks_w_metadata", {})
        
        # Flatten gold benchmark chunks
        gold_chunks = []
        for doc_name, doc_chunks in chunks_meta.items():
            for c in doc_chunks:
                if isinstance(c, dict) and "content" in c:
                    gold_chunks.append(c["content"])
                elif isinstance(c, str):
                    gold_chunks.append(c)

        print(f"\n--- [{idx}/{len(items)}] ID: {q_id} ---")
        print(f"Query: {query[:90]}...")

        # Call POST /query
        payload = {
            "collection_name": collection_name,
            "query": query,
            "top_chunks": top_chunks,
            "provider": provider,
        }
        t0 = time.time()
        try:
            resp = requests.post(f"{base_url}/query", json=payload, timeout=90)
            resp.raise_for_status()
            q_res = resp.json()
        except Exception as e:
            print(f"  [Error] Query call failed: {e}")
            q_res = {"answer": f"Error: {e}", "chunks": [], "trace": []}

        q_elapsed = time.time() - t0
        ai_answer = q_res.get("answer", "")
        retrieved_raw = q_res.get("chunks", [])
        retrieved_texts = [c.get("content", "") for c in retrieved_raw if c.get("content")]

        print(f"  Latency: {q_elapsed:.2f}s | Chunks retrieved: {len(retrieved_texts)}")
        print(f"  Answer preview: {ai_answer[:120].strip()}...")

        # Record for Evaluation.ipynb format
        record = {
            "query": query,
            "ai_answer": ai_answer,
            "benchmark_answer": gold_answer,
            "retrieved_chunks": retrieved_texts,
            "benchmark_chunks": gold_chunks,
            "metadata": {
                "id": q_id,
                "latency_s": round(q_elapsed, 2),
                "query_type": q_res.get("query_type", "unknown"),
                "chunks_detail": retrieved_raw,
                "trace": q_res.get("trace", [])
            }
        }
        dataset_records.append(record)

        # Optional LLM-as-a-judge scoring
        if evaluator:
            ans_p, ans_r, _ = evaluator.answer_precision_recall(ai_answer, gold_answer)
            ctx_p, ctx_r = evaluator.context_precision_recall(query, retrieved_texts, len(gold_chunks))
            eval_metrics_list.append({
                "id": q_id,
                "ans_p": ans_p, "ans_r": ans_r,
                "ctx_p": ctx_p, "ctx_r": ctx_r
            })
            print(f"  Scores: AnsP={ans_p:.2f} AnsR={ans_r:.2f} CtxP={ctx_p:.2f} CtxR={ctx_r:.2f}")

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Dataset written to {output_path} ({len(dataset_records)} records).")

    # Also save copy to evaluation_outputs_api.json
    api_out = os.path.join(os.path.dirname(output_path), "evaluation_outputs_api.json")
    with open(api_out, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2, ensure_ascii=False)

    if eval_metrics_list:
        avg_ap = sum(m["ans_p"] for m in eval_metrics_list) / len(eval_metrics_list)
        avg_ar = sum(m["ans_r"] for m in eval_metrics_list) / len(eval_metrics_list)
        avg_cp = sum(m["ctx_p"] for m in eval_metrics_list) / len(eval_metrics_list)
        avg_cr = sum(m["ctx_r"] for m in eval_metrics_list) / len(eval_metrics_list)

        print("\n" + "=" * 50)
        print("         BENCHMARK EVALUATION SUMMARY")
        print("=" * 50)
        print(f"  Questions Evaluated : {len(eval_metrics_list)}")
        print(f"  Avg Answer Precision : {avg_ap:.3f}")
        print(f"  Avg Answer Recall    : {avg_ar:.3f}")
        print(f"  Avg Context Precision: {avg_cp:.3f}")
        print(f"  Avg Context Recall   : {avg_cr:.3f}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test NullRAG REST APIs against benchmark dataset.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of NullRAG server")
    parser.add_argument("--collection", default="sample_test", help="Collection name to index/query")
    parser.add_argument("--zip-path", default="train_docs_sample.zip", help="Path to zip archive of PDFs")
    parser.add_argument("--benchmark-file", default="Train/Benchmarks/benchmark_100.jsonl", help="Benchmark JSONL file")
    parser.add_argument("--skip-index", action="store_true", help="Skip POST /index step if collection already exists")
    parser.add_argument("--limit", type=int, default=5, help="Number of benchmark queries to run (0 for all)")
    parser.add_argument("--top-chunks", type=int, default=30, help="Number of chunks to retrieve per query")
    parser.add_argument("--provider", default="opencode", help="LLM provider: opencode, gemini, or auto")
    parser.add_argument("--eval", action="store_true", help="Run LLM judge (RAGMetricsEvaluator) on outputs")
    parser.add_argument("--output", default="results/evaluation_outputs.json", help="Path to save evaluation dataset")

    args = parser.parse_args()

    if not check_api_health(args.api_url):
        print("[Error] Cannot reach NullRAG API server. Make sure `python server.py` is running.")
        sys.exit(1)

    # 1. Indexing step
    if not args.skip_index:
        zip_file = ensure_train_zip(args.zip_path)
        index_collection(args.api_url, args.collection, zip_file)
    else:
        print(f"[Skip Index] Using existing collection '{args.collection}'.")

    # 2. Query step
    run_benchmark(
        base_url=args.api_url,
        collection_name=args.collection,
        benchmark_file=args.benchmark_file,
        limit=args.limit,
        top_chunks=args.top_chunks,
        provider=args.provider,
        output_path=args.output,
        run_eval=args.eval
    )

if __name__ == "__main__":
    main()
