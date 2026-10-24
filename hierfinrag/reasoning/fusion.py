import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Literal
import torch.nn as nn

class SymbolicCalculator:
    """Executes precise arithmetic operations on financial figures."""
    def __init__(self, precision=2):
        self.precision = precision
        
    def compute(self, operation: str, values: Dict[str, float]) -> float:
        try:
            if operation == "add":
                return round(sum(values.values()), self.precision)
            elif operation == "subtract":
                vals = list(values.values())
                if len(vals) < 2:
                    return 0.0
                return round(vals[0] - vals[1], self.precision)
            elif operation in ("divide", "ratio"):
                num = values.get("numerator", list(values.values())[0])
                den = values.get("denominator", list(values.values())[1])
                if den == 0:
                    return 0.0
                return round(num / den, self.precision)
            elif operation == "percentage_change":
                old = values.get("old_value", values.get("previous", list(values.values())[0]))
                new = values.get("new_value", values.get("current", list(values.values())[1]))
                if old == 0:
                    return 0.0
                return round(((new - old) / old) * 100, self.precision)
            else:
                return 0.0
        except Exception as e:
            return 0.0

class ReasoningRouter:
    """Classifies queries into Neural, Symbolic, or Hybrid reasoning paths."""
    def __init__(self):
        self.symbolic_keywords = [
            "calculate", "sum", "total", "difference", "growth", "increase", 
            "decrease", "margin", "ratio", "percentage", "average", "compute", "how much"
        ]
        
    def predict(self, query: str, context_types: List[str]) -> Dict[str, float]:
        query_lower = query.lower()
        probs = {"neural": 0.35, "symbolic": 0.25, "hybrid": 0.40}
        
        symbolic_score = sum(1 for k in self.symbolic_keywords if k in query_lower)
        if symbolic_score > 0:
            probs["symbolic"] += 0.3
            probs["hybrid"] += 0.2
            probs["neural"] -= 0.3
            
        has_table = any("table" in str(t).lower() or "cell" in str(t).lower() for t in context_types)
        if has_table:
            probs["symbolic"] += 0.15
            probs["hybrid"] += 0.15
            probs["neural"] -= 0.1
            
        total = sum(probs.values())
        return {k: max(0.0, v / total) for k, v in probs.items()}

    def determine_mode(self, query: str, context_types: List[str]) -> str:
        probs = self.predict(query, context_types)
        return max(probs, key=probs.get)

class SymbolicNeuralFusion(nn.Module):
    """
    Blends LLM reasoning with symbolic execution for precise financial answers.
    Supports Gemini, OpenCode/DeepSeek, OpenAI, or grounded extractive fallback.
    """
    def __init__(
        self, 
        provider: str = "opencode",
        api_key: Optional[str] = None, 
        model: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        super().__init__()
        self.calculator = SymbolicCalculator()
        self.router = ReasoningRouter()
        
        # Load keys from config.py if present
        config_keys = {}
        try:
            import config
            for k in ["OPENCODE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY"]:
                if hasattr(config, k):
                    config_keys[k] = getattr(config, k)
        except ImportError:
            pass

        self.opencode_key = api_key or config_keys.get("OPENCODE_API_KEY") or os.getenv("OPENCODE_API_KEY", "")
        self.openai_key = api_key or config_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.gemini_key = ""  # Enforce OpenCode API only; Gemini disabled
        
        # Resolve provider: Strictly prefer opencode
        if provider in ("auto", "opencode", "gemini"):
            if provider == "gemini":
                print("[SymbolicNeuralFusion] Notice: Gemini API disabled. Using OpenCode API.")
            if self.opencode_key:
                self.provider = "opencode"
            elif self.openai_key:
                self.provider = "openai"
            else:
                self.provider = "extractive"
        elif provider == "openai":
            self.provider = "openai"
        else:
            self.provider = "extractive"

        # Default models & endpoints
        if self.provider == "opencode":
            self.model = model or "deepseek-v4.1-flash"
            self.api_key = self.opencode_key
            self.base_url = base_url or "https://opencode.ai/inference/openai/v1"
        elif self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            self.api_key = self.openai_key
            self.base_url = base_url or config_keys.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        else:
            self.model = "extractive"
            self.api_key = ""
            self.base_url = None

    def _call_gemini(self, prompt: str, system_prompt: str) -> Optional[str]:
        # Gemini API disabled as per configuration; redirecting to OpenCode
        return self._call_openai_compatible(prompt, system_prompt)

    def _call_openai_compatible(self, prompt: str, system_prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import openai
                kwargs = {"api_key": self.api_key, "timeout": 45.0}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                client = openai.OpenAI(**kwargs)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=900,
                )
                return response.choices[0].message.content
            except Exception as e:
                # Fallback to direct requests if needed
                try:
                    import requests
                    endpoint = f"{self.base_url.rstrip('/')}/chat/completions" if self.base_url else "https://opencode.ai/inference/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 900,
                    }
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                except Exception as req_err:
                    if attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    print(f"[SymbolicNeuralFusion] OpenCode API call error (attempt {attempt+1}/{max_retries}): {e} | req_fallback: {req_err}")
                    return None
        return None

    def _call_llm(self, prompt: str, system_prompt: str = "You are a financial analyst. Answer concisely and precisely based ONLY on the excerpts. No preamble. Start directly with the answer.") -> Optional[str]:
        # Enforce OpenCode API only (no Gemini API)
        return self._call_openai_compatible(prompt, system_prompt)

    def _fallback_grounded_synthesis(self, query: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
        """Extractive synthesis when an LLM API key is not present."""
        query_words = set(re.findall(r"\w{3,}", query.lower()))
        scored_snippets = []
        
        for item in retrieved_contexts:
            text = item.get("text", "")
            meta = item.get("metadata", {})
            page = meta.get("page_number", meta.get("page_numbers", ""))
            doc = meta.get("document", item.get("document", ""))
            
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                s_words = set(re.findall(r"\w{3,}", s_clean.lower()))
                overlap = len(query_words & s_words)
                if overlap > 0:
                    cite = f"[{doc} p.{page}]" if page else f"[{doc}]"
                    scored_snippets.append((overlap, f"- {s_clean} {cite}"))
                    
            if item.get("type") == "table":
                scored_snippets.append((len(query_words) * 2, f"\n{text}\n"))

        scored_snippets.sort(key=lambda x: x[0], reverse=True)
        top_snippets = [snip for _, snip in scored_snippets[:8]]
        
        if not top_snippets:
            top_snippets = [f"- {item.get('text', '')[:300]}..." for item in retrieved_contexts[:3]]
            
        summary = (
            f"### Financial Summary in response to: {query}\n\n"
            f"Based on the retrieved documentation:\n\n" +
            "\n".join(top_snippets)
        )
        return summary

    def _extract_numeric_operands(
        self, query: str, combined_evidence: str
    ) -> Optional[Dict]:
        """
        Try to extract two numeric operands from evidence for deterministic verification.
        Returns None if not confident enough to auto-verify.
        Only triggers for simple single-operation questions.
        """
        pattern = re.compile(
            r"([\w\s]{3,30}?)\s*[:\-\u2013]\s*(?:rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:crore|cr|lakh|million|%)?",
            re.IGNORECASE,
        )
        matches = pattern.findall(combined_evidence[:4000])
        if len(matches) < 2:
            return None

        label_vals: Dict[str, float] = {}
        for label, val_str in matches[:20]:
            key = label.strip().lower()
            try:
                label_vals[key] = float(val_str.replace(",", ""))
            except ValueError:
                pass

        if len(label_vals) < 2:
            return None

        q_lower = query.lower()
        if any(k in q_lower for k in ["change", "growth", "increase", "decrease", "yoy"]):
            operation = "percentage_change"
        elif any(k in q_lower for k in ["difference", "subtract", "how much more", "how much less"]):
            operation = "subtract"
        elif any(k in q_lower for k in ["ratio", "margin", "divide", "per"]):
            operation = "divide"
        elif any(k in q_lower for k in ["sum", "total", "add", "combined"]):
            operation = "add"
        else:
            return None

        vals = list(label_vals.values())
        labels = list(label_vals.keys())
        if operation == "percentage_change":
            return {"operation": operation,
                    "values": {"old_value": vals[0], "new_value": vals[1]},
                    "labels": labels[:2]}
        elif operation == "subtract":
            return {"operation": operation,
                    "values": {"a": vals[0], "b": vals[1]},
                    "labels": labels[:2]}
        elif operation == "divide":
            return {"operation": operation,
                    "values": {"numerator": vals[0], "denominator": vals[1]},
                    "labels": labels[:2]}
        elif operation == "add":
            return {"operation": operation,
                    "values": {str(i): v for i, v in enumerate(vals[:4])},
                    "labels": labels[:4]}
        return None

    def forward(self, query: str, retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes reasoning over retrieved contexts."""
        context_types = [c.get("type", "text") for c in retrieved_contexts]
        mode = self.router.determine_mode(query, context_types)
        
        evidence_blocks = []
        chunk_texts = []
        for idx, c in enumerate(retrieved_contexts, start=1):
            txt = c.get("text", "")
            chunk_texts.append(txt)
            meta = c.get("metadata", {})
            page = meta.get("page_number", meta.get("page_numbers", "N/A"))
            doc = c.get("document", meta.get("document", "Doc"))
            evidence_blocks.append(f"[Excerpt {idx} | {doc} | Page: {page}]\n{txt}")
            
        combined_evidence = "\n\n".join(evidence_blocks)

        # ── Deterministic numerical verification ──────────────────────────
        # When mode is symbolic/hybrid, verify arithmetic using SymbolicCalculator.
        symbolic_note = ""
        symbolic_verified = False
        if mode in ("symbolic", "hybrid"):
            operand_info = self._extract_numeric_operands(query, combined_evidence)
            if operand_info:
                try:
                    verified_result = self.calculator.compute(
                        operand_info["operation"], operand_info["values"]
                    )
                    labels = operand_info.get("labels", [])
                    symbolic_note = (
                        f"\n\n[VERIFIED CALCULATION — Symbolic Engine]\n"
                        f"Operation: {operand_info['operation']}\n"
                        f"Operands: {dict(zip(labels, list(operand_info['values'].values())))}\n"
                        f"Result: {verified_result}\n"
                        f"Use this verified figure in your answer; do not recompute it differently.\n"
                    )
                    symbolic_verified = True
                except Exception:
                    pass
        
        # Truncate each excerpt to keep total prompt under ~1500 tokens
        MAX_EXCERPT_CHARS = 500
        evidence_blocks_trimmed = []
        for idx, c in enumerate(retrieved_contexts, start=1):
            txt = c.get("text", "")
            meta = c.get("metadata", {})
            page = meta.get("page_number", meta.get("page_numbers", "N/A"))
            doc = c.get("document", meta.get("document", "Doc"))
            txt_trimmed = txt[:MAX_EXCERPT_CHARS] + ("…" if len(txt) > MAX_EXCERPT_CHARS else "")
            evidence_blocks_trimmed.append(f"[{idx}|{doc}|p.{page}] {txt_trimmed}")

        combined_evidence = "\n".join(evidence_blocks_trimmed)

        llm_prompt = f"""Q: {query}

EXCERPTS:
{combined_evidence}
{symbolic_note}
Answer the question using ONLY these excerpts. Be specific: include figures, dates, entities. Cover all relevant facts. Cite [excerpt number] inline."""

        llm_answer = self._call_llm(llm_prompt)
        if not llm_answer:
            llm_answer = self._fallback_grounded_synthesis(query, retrieved_contexts)
            
        return {
            "mode": mode,
            "answer": llm_answer,
            "retrieved_chunks": chunk_texts,
            "provider": self.provider,
            "model": self.model,
            "symbolic_verified": symbolic_verified,
        }

