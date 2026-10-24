import os
import json
import time
import requests
from typing import List, Dict, Any, Optional

OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1/chat/completions"
OPENCODE_MODEL = "deepseek-v4.1-flash"

class OpenCodeClient:
    def __init__(self, api_key: str, model: str = OPENCODE_MODEL, base_url: str = OPENCODE_BASE_URL):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def call_structured(self, system_prompt: str, user_prompt: str, max_tokens: int = 10000,
                         temperature: float = 0, max_retries: int = 3) -> dict:
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
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError(f"Empty content, finish_reason={data['choices'][0].get('finish_reason')}")
                return self._safe_json_parse(content)
            except (requests.RequestException, KeyError, ValueError) as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))
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

class GeminiJudgeClient:
    """Judge client using Google Gemini JSON mode for testing."""
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def call_structured(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict:
        full_prompt = f"System Instructions: {system_prompt}\n\nTask:\n{user_prompt}"
        
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config={"response_mime_type": "application/json", "temperature": 0.0}
                )
                return OpenCodeClient._safe_json_parse(resp.text)
            except Exception as e:
                last_error = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Gemini judge call failed: {last_error}")

class RAGMetricsEvaluator:
    def __init__(self, client: Any):
        self.client = client

    def _decompose_points(self, text: str) -> list:
        system = "You are a precise fact-extraction assistant. Respond only with JSON."
        user = (
            "Break the following answer into a list of discrete, atomic factual points. "
            "Each point should be a single, independently checkable claim. If the answer "
            "is empty or contains no checkable claims, return an empty list.\n\n"
            f"Answer:\n{text}\n\n"
            'Respond as JSON exactly in this shape: {"points": ["point 1", "point 2", ...]}'
        )
        result = self.client.call_structured(system, user)
        return result.get("points", []) or []

    def _compare_answer_points(self, ai_points: list, benchmark_points: list) -> dict:
        system = (
            "You are a strict evaluator comparing two labeled sets of factual points: "
            "a benchmark (reference) answer's points and an AI-generated response's points. "
            "Respond only with JSON."
        )

        labeled_bp = {f"bp{i + 1}": p for i, p in enumerate(benchmark_points)}
        labeled_ap = {f"ap{i + 1}": p for i, p in enumerate(ai_points)}

        user = (
            "Benchmark points:\n"
            f"{json.dumps(labeled_bp, indent=2)}\n\n"
            "Agent response points:\n"
            f"{json.dumps(labeled_ap, indent=2)}\n\n"
            "Compare them and identify, using the ids shown above:\n"
            "1. matched_ids: ids of benchmark points (bpX) that are also present/covered in the agent response\n"
            "2. missing_ids: ids of benchmark points (bpX) that are NOT present in the agent response\n"
            "3. extra_ids: ids of agent response points (apX) that are NOT supported by any benchmark point\n\n"
            "Every bpX id must appear in exactly one of matched_ids or missing_ids. "
            "Every apX id that has no matching benchmark point should appear in extra_ids.\n\n"
            'Respond as JSON exactly in this shape: '
            '{"matched_ids": ["bp1", "bp3"], "missing_ids": ["bp2"], "extra_ids": ["ap4"]}'
        )
        result = self.client.call_structured(system, user)
        return {
            "matched_ids": result.get("matched_ids", []) or [],
            "missing_ids": result.get("missing_ids", []) or [],
            "extra_ids": result.get("extra_ids", []) or [],
            "benchmark_points": labeled_bp,
            "agent_points": labeled_ap,
        }

    def answer_precision_recall(self, ai_answer: str, benchmark_answer: str) -> tuple:
        ai_points = self._decompose_points(ai_answer)
        benchmark_points = self._decompose_points(benchmark_answer)

        if not ai_points and not benchmark_points:
            empty_detail = {"matched_ids": [], "missing_ids": [], "extra_ids": [],
                             "benchmark_points": {}, "agent_points": {}}
            return 0.0, 0.0, empty_detail

        if not ai_points:
            detail = {
                "matched_ids": [],
                "missing_ids": [f"bp{i + 1}" for i in range(len(benchmark_points))],
                "extra_ids": [],
                "benchmark_points": {f"bp{i + 1}": p for i, p in enumerate(benchmark_points)},
                "agent_points": {},
            }
            return 0.0, 0.0, detail

        detail = self._compare_answer_points(ai_points, benchmark_points)
        matched_bp = {i for i in detail["matched_ids"] if i in detail["benchmark_points"]}
        extra_ap = {i for i in detail["extra_ids"] if i in detail["agent_points"]}

        precision = (len(ai_points) - len(extra_ap)) / len(ai_points) if ai_points else 0.0
        recall = len(matched_bp) / len(benchmark_points) if benchmark_points else 0.0
        return precision, recall, detail

    def _judge_chunk_relevance(self, query: str, chunks: list) -> list:
        system = (
            "You are a strict evaluator judging whether retrieved text chunks are relevant "
            "to a user's query. Respond only with JSON."
        )
        numbered = [{"id": i, "text": c} for i, c in enumerate(chunks)]
        user = (
            f"User query:\n{query}\n\n"
            "Retrieved chunks:\n"
            f"{json.dumps(numbered, indent=2)}\n\n"
            "For each chunk, decide if it is genuinely relevant and useful for answering the query.\n"
            'Respond as JSON exactly in this shape: '
            '{"verdicts": [{"id": 0, "relevant": true}, {"id": 1, "relevant": false}]}'
        )
        result = self.client.call_structured(system, user)
        return result.get("verdicts", []) or []

    def context_precision_recall(self, query: str, retrieved_chunks: list, benchmark_chunk_count: int) -> tuple:
        if not retrieved_chunks:
            return 0.0, 0.0

        verdicts = self._judge_chunk_relevance(query, retrieved_chunks)
        relevant_count = sum(1 for v in verdicts if v.get("relevant") is True)

        precision = relevant_count / len(retrieved_chunks)
        recall = relevant_count / benchmark_chunk_count if benchmark_chunk_count else 0.0
        return precision, recall

    def evaluate_single(self, query: str, ai_answer: str, benchmark_answer: str,
                         retrieved_chunks: list, benchmark_chunks: list) -> dict:
        ans_p, ans_r, ans_detail = self.answer_precision_recall(ai_answer, benchmark_answer)
        ctx_p, ctx_r = self.context_precision_recall(query, retrieved_chunks, len(benchmark_chunks))
        return {
            "answer_precision": ans_p,
            "answer_recall": ans_r,
            "context_precision": ctx_p,
            "context_recall": ctx_r,
            "answer_detail": ans_detail,
        }

def evaluate_dataset(evaluator: RAGMetricsEvaluator, dataset: list, verbose: bool = True) -> dict:
    results = []
    for i, item in enumerate(dataset):
        metrics = evaluator.evaluate_single(
            query=item["query"],
            ai_answer=item["ai_answer"],
            benchmark_answer=item["benchmark_answer"],
            retrieved_chunks=item["retrieved_chunks"],
            benchmark_chunks=item["benchmark_chunks"],
        )
        metrics["query"] = item["query"]
        results.append(metrics)
        if verbose:
            print(
                f"[{i + 1}/{len(dataset)}] {item['query'][:60]!r} -> "
                f"AnsP={metrics['answer_precision']:.2f} AnsR={metrics['answer_recall']:.2f} "
                f"CtxP={metrics['context_precision']:.2f} CtxR={metrics['context_recall']:.2f}"
            )

    def avg(key):
        return sum(r[key] for r in results) / len(results) if results else 0.0

    summary = {
        "num_questions": len(results),
        "avg_answer_precision": avg("answer_precision"),
        "avg_answer_recall": avg("answer_recall"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        "per_question": results,
    }
    return summary

def evaluate_with_judge(api_key: str, dataset: list, provider: str = "opencode", model: Optional[str] = None) -> dict:
    client = OpenCodeClient(api_key=api_key, model=model or OPENCODE_MODEL)
    evaluator = RAGMetricsEvaluator(client)
    return evaluate_dataset(evaluator, dataset)
