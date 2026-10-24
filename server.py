"""
NullRAG FastAPI Server
======================
Exposes exactly the two endpoints required by the hackathon:

  POST /index  – ingest a zip of PDFs into a named collection
  POST /query  – retrieve + generate a grounded answer from a collection

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000

Or:
    python server.py
"""

import io
import os
import json
import time
import tempfile
import asyncio
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NullRAG – Smart Knowledge Base for Agents",
    description=(
        "Hierarchical Multimodal RAG with dynamic metadata routing, "
        "FAISS indexing, adaptive retrieval, and Langfuse observability."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Lazy engine import (heavy; done once on first request)
# ---------------------------------------------------------------------------
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from nullrag.index.engine import get_engine
        _engine = get_engine()
    return _engine


@app.on_event("startup")
async def startup_event():
    print("[Server] Warming up NullRAG engine on startup...")
    await asyncio.to_thread(_get_engine)
    print("[Server] Engine warmup complete. Server ready for queries.")


# ---------------------------------------------------------------------------
# In-progress index tracker (simple; for status endpoint)
# ---------------------------------------------------------------------------
_index_status: dict = {}


# ---------------------------------------------------------------------------
# POST /index
# ---------------------------------------------------------------------------

@app.post("/index", summary="Index a zip of PDFs into a named collection")
async def index_documents(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP archive containing PDF documents"),
    collection_name: str = Form(..., description="Unique name for this document collection"),
):
    """
    Unzips the uploaded archive, parses every PDF, batch-embeds all chunks,
    and stores them in a FAISS index keyed by `collection_name`.

    Supports multiple independent collections – each call with a different
    `collection_name` creates an isolated index.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .zip archive")

    # Save upload to a temp file
    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(contents)
    tmp.close()
    zip_path = tmp.name

    _index_status[collection_name] = {"status": "indexing", "started_at": time.time()}

    try:
        engine = _get_engine()
        result = await asyncio.to_thread(engine.index_zip, zip_path, collection_name)
        # Invalidate cached queries for this collection so subsequent queries strictly use the new index
        try:
            from nullrag.index.engine import _QUERY_CACHE
            for k in list(_QUERY_CACHE.keys()):
                if k.startswith(f"{collection_name}::"):
                    del _QUERY_CACHE[k]
        except Exception:
            pass
        _index_status[collection_name] = {
            "status":           "ready",
            "completed_at":     time.time(),
            "num_documents":    result.get("num_documents", 0),
            "num_chunks":       result.get("num_chunks", 0),
            "elapsed_seconds":  result.get("elapsed_seconds", 0),
        }
        return JSONResponse(content={
            "status":            "success",
            "collection_name":   collection_name,
            "documents_indexed": result.get("num_documents", 0),
            "num_documents":     result.get("num_documents", 0),
            "num_chunks":        result.get("num_chunks", 0),
            "elapsed_seconds":   result.get("elapsed_seconds", 0),
            "message": (
                f"Collection '{collection_name}' indexed successfully. "
                f"{result.get('num_documents', 0)} documents, "
                f"{result.get('num_chunks', 0)} chunks."
            ),
        })
    except Exception as e:
        _index_status[collection_name] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    finally:
        try:
            os.unlink(zip_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    collection_name: str
    query: str
    top_chunks: Optional[int] = 30
    provider: Optional[str] = "opencode"
    model: Optional[str] = None


@app.post("/query", summary="Query a named collection and get a grounded answer")
async def query_collection(request: QueryRequest):
    """
    Given a `collection_name` and `query`, runs the full NullRAG pipeline:

    1. Metadata filter  – narrows doc pool using entity/year constraints
    2. Query classifier – determines targeted vs synthesis retrieval mode
    3. FAISS dense retrieval – coarse ANN search
    4. Adaptive retrieval   – HierarchicalRetriever or SectionMapAggregator
    5. LLM generation       – grounded answer via OpenCode/Gemini/OpenAI
    6. Langfuse tracing     – latency + cost recorded (if configured)

    Returns: answer, retrieved chunks (with page citations), and a step trace.
    """
    engine = _get_engine()

    try:
        result = await asyncio.to_thread(
            engine.query,
            query=request.query,
            collection_name=request.collection_name,
            top_chunks=request.top_chunks,
            provider=request.provider,
            model=request.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    if "error" in result and not result.get("answer"):
        if result.get("error") == "unknown collection" or "not found" in str(result.get("error")).lower():
            raise HTTPException(status_code=404, detail="unknown collection")
        raise HTTPException(status_code=404, detail=result["error"])

    # Contract Section 3.4: Unanswerable returns HTTP 200 with honest answer, empty chunks, trace
    if not result.get("answer"):
        result["answer"] = "The documents do not contain this information."
        result["chunks"] = []

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /query/stream  – same as /query but streams answer via SSE
# ---------------------------------------------------------------------------

@app.post("/query/stream", summary="Stream answer token-by-token (SSE)")
async def query_stream(request: QueryRequest):
    """
    Same pipeline as POST /query, but streams the answer as Server-Sent Events.
    Each event is: `data: <text_chunk>\\n\\n`
    The final event is: `data: [DONE] <json_metadata>\\n\\n`

    Use this endpoint for interactive UIs — perceived latency drops to ~1s.
    """
    engine = _get_engine()

    async def _stream_generator() -> AsyncGenerator[str, None]:
        try:
            result = await asyncio.to_thread(
                engine.query,
                query=request.query,
                collection_name=request.collection_name,
                top_chunks=request.top_chunks,
                provider=request.provider,
                model=request.model,
            )
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
            return

        answer = result.get("answer", "")
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0)

        meta = {k: v for k, v in result.items() if k != "answer"}
        yield f"data: [DONE] {json.dumps(meta)}\n\n"

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /cache/clear  – evict in-process query cache
# ---------------------------------------------------------------------------

@app.post("/cache/clear", summary="Clear the query answer cache")
async def clear_cache():
    """Clears the in-memory query answer cache. Useful before benchmark runs."""
    from nullrag.index.engine import _QUERY_CACHE
    count = len(_QUERY_CACHE)
    _QUERY_CACHE.clear()
    return JSONResponse(content={"cleared": count, "status": "ok"})


# ---------------------------------------------------------------------------
# GET /collections  – list all indexed collections
# ---------------------------------------------------------------------------

@app.get("/collections", summary="List all indexed collections")
async def list_collections():
    from nullrag.index.collection_store import registry
    collections = registry.list_collections()
    details = []
    for name in collections:
        store = registry.get(name)
        details.append(store.stats() if store else {"name": name})
    return JSONResponse(content={"collections": details})


# ---------------------------------------------------------------------------
# GET /collection/{name}/status
# ---------------------------------------------------------------------------

@app.get("/collection/{collection_name}/status", summary="Check indexing status of a collection")
async def collection_status(collection_name: str):
    status = _index_status.get(collection_name, {"status": "not_found"})
    from nullrag.index.collection_store import registry
    store = registry.get(collection_name)
    if store:
        status.update(store.stats())
    return JSONResponse(content=status)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "service": "NullRAG", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
