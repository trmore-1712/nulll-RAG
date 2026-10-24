"""
Langfuse Observability Wrapper
================================
Provides a thin, optional wrapper around Langfuse for tracing query latency,
token costs, and per-step pipeline details.

If Langfuse is not installed or LANGFUSE_* env vars are missing, all calls
silently no-op so the rest of the system keeps working.

Environment variables (or set in config.py):
  LANGFUSE_PUBLIC_KEY  – Langfuse project public key
  LANGFUSE_SECRET_KEY  – Langfuse project secret key
  LANGFUSE_HOST        – e.g. "https://cloud.langfuse.com" (default)
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional


def _load_langfuse_keys() -> Dict[str, str]:
    keys = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
        "host":       os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    }
    try:
        import config
        for attr, env in [
            ("LANGFUSE_PUBLIC_KEY", "public_key"),
            ("LANGFUSE_SECRET_KEY", "secret_key"),
            ("LANGFUSE_HOST",       "host"),
        ]:
            if hasattr(config, attr) and getattr(config, attr):
                keys[env] = getattr(config, attr)
    except ImportError:
        pass
    return keys


class _NoOpSpan:
    """Silently discards all calls."""
    def update(self, **kwargs): pass
    def end(self, **kwargs):   pass
    def generation(self, **kwargs): return self
    def score(self, **kwargs):      pass


class ObservabilityTracer:
    """
    Wraps a single pipeline execution (one /query call) with Langfuse tracing.

    Usage:
        tracer = ObservabilityTracer(query="What was the net loss?")
        tracer.start()

        with tracer.span("metadata_filter") as sp:
            result = do_filter(...)
            sp.update(output=result, metadata={"docs_before": 35, "docs_after": 3})

        tracer.record_generation(
            model="deepseek-v4.1-flash",
            prompt_tokens=1200,
            completion_tokens=400,
            cost_usd=0.0015,
        )
        tracer.finish()
        print(tracer.trace_url)
    """

    def __init__(self, query: str, collection: str = ""):
        self._query      = query
        self._collection = collection
        self._trace_id   = str(uuid.uuid4())
        self._lf         = None          # Langfuse client
        self._trace      = None
        self._start      = None
        self.trace_url   = ""
        self.total_latency_ms: float = 0.0
        self.total_cost_usd:   float = 0.0
        self._active = False

        keys = _load_langfuse_keys()
        if keys["public_key"] and keys["secret_key"]:
            try:
                from langfuse import Langfuse  # type: ignore
                self._lf = Langfuse(
                    public_key=keys["public_key"],
                    secret_key=keys["secret_key"],
                    host=keys["host"],
                )
                self._active = True
            except Exception as e:
                print(f"[Langfuse] Init skipped: {e}")

    def start(self) -> "ObservabilityTracer":
        self._start = time.time()
        if self._active:
            try:
                self._trace = self._lf.trace(
                    id=self._trace_id,
                    name="nullrag_query",
                    input={"query": self._query, "collection": self._collection},
                    metadata={"collection": self._collection},
                )
                self.trace_url = f"https://cloud.langfuse.com/trace/{self._trace_id}"
            except Exception as e:
                print(f"[Langfuse] trace() failed: {e}")
        return self

    class _Span:
        def __init__(self, trace, name: str, active: bool):
            self._name   = name
            self._span   = None
            self._start  = time.time()
            self._active = active
            if active and trace is not None:
                try:
                    self._span = trace.span(name=name)
                except Exception:
                    pass

        def update(self, **kwargs):
            if self._active and self._span:
                try:
                    self._span.update(**kwargs)
                except Exception:
                    pass

        def end(self, **kwargs):
            if self._active and self._span:
                try:
                    self._span.end(end_time=time.time(), **kwargs)
                except Exception:
                    pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.end()

    def span(self, name: str) -> "_Span":
        return self._Span(self._trace, name, self._active)

    def record_generation(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.total_cost_usd += cost_usd
        if self._active and self._trace:
            try:
                self._trace.generation(
                    name="llm_generation",
                    model=model,
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                    },
                    metadata={"cost_usd": cost_usd},
                )
            except Exception:
                pass

    def finish(self, answer: str = "") -> None:
        if self._start:
            self.total_latency_ms = (time.time() - self._start) * 1000
        if self._active and self._trace:
            try:
                self._trace.update(
                    output={"answer": answer[:500]},
                    metadata={
                        "latency_ms":  self.total_latency_ms,
                        "cost_usd":    self.total_cost_usd,
                    },
                )
                self._lf.flush()
            except Exception as e:
                print(f"[Langfuse] finish() failed: {e}")

    def summary(self) -> Dict[str, Any]:
        return {
            "trace_id":        self._trace_id,
            "trace_url":       self.trace_url,
            "latency_ms":      round(self.total_latency_ms, 2),
            "cost_usd":        round(self.total_cost_usd, 6),
        }
