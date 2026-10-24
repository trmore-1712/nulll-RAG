# HierFinRAG — Hierarchical Multimodal RAG for Financial Document Understanding

> **Treat financial reports as structured evidence networks, not flat text.**

HierFinRAG is a specialised Retrieval-Augmented Generation framework built for complex financial documents — 10-Ks, Annual Reports, credit disclosures — where the evidence needed to answer a question is distributed across narrative sections, tables, footnotes and different reporting periods.

---

## The Problem with Conventional RAG

A standard RAG pipeline asks: *"Which chunks are most similar to the query?"*

For a question like:

> *"What was the company's credit exposure, what provisions were made, and what percentage of the exposure was covered?"*

A vanilla retriever may return five semantically relevant passages about credit risk — yet never retrieve the provision table on page 88 or the footnote on page 90. The answer looks plausible but is numerically incomplete or wrong.

### Four Failure Modes We Eliminate

| Failure | Description |
|---|---|
| **Table context loss** | A retrieved value loses its row label, column header, unit or footnote |
| **Arithmetic hallucination** | The LLM calculates an incorrect ratio or percentage from financial values |
| **Redundant retrieval** | Many similar passages consume the context window without covering all requirements |
| **Disconnected evidence** | The correct answer requires connecting facts from different pages or documents |

---

## Architecture Overview

```
PDF / ZIP
    |
    v
+-------------------------------+
|  Stage 1 · Hierarchical       |
|  Ingestion & Parsing          |  PyMuPDF -> Sections, Paragraphs,
|  (hierfinrag/parsing)         |  Tables, Cells + stable Evidence IDs
+---------------+---------------+
                |
                v
+-------------------------------+
|  Stage 2 · Structural         |
|  Evidence Graph (SEG)         |  Nodes: Doc / Section / Chunk / Table / Cell / Note
|  (hierfinrag/graph)           |  Edges: parent-child, sequential, cross-ref, semantic
+---------------+---------------+
                |
                v
+-------------------------------+
|  Stage 3 · Hybrid Retrieval   |  FAISS dense (semantic)
|  (hierfinrag/retrieval)       |  + BM25 sparse (lexical)
|                               |  -> Reciprocal Rank Fusion (RRF)
+---------------+---------------+
                |
                v
+-------------------------------+
|  Stage 4 · Adaptive Query     |  Targeted mode  (precise metric lookup)
|  Routing                      |  Synthesis mode (multi-section evidence)
|  (hierfinrag/routing)         |
+---------------+---------------+
                |
                v
+-------------------------------+
|  Stage 5 · Obligation-Aware   |  Decomposes query into required facts
|  Evidence Ledger              |  Selects complementary evidence; follows
|  (hierfinrag/evidence)        |  graph cross-references to recover missing data
+---------------+---------------+
                |
                v
        Grounded Answer
        + Source chunks
        + Page citations
        + Execution trace
```

---

## Stage Descriptions

### Stage 1 — Hierarchical Ingestion & Parsing

PyMuPDF parses every PDF page into a labelled hierarchy: **Document → Section → Paragraph / Table → Cell**. Each chunk carries stable metadata (document name, page number, section, type) so every retrieved fact can be traced back to its exact source page.

Instead of a flat string `"Assets Cash 5000 Loans 20000 Total 25000"`, the system preserves:

| Particulars | 2024 (Rs Cr) | 2023 (Rs Cr) |
|---|---|---|
| Gross NPAs  | 3,120.50 | 4,210.80 |
| Provisions  | 2,450.00 | 3,100.00 |

Row labels, column headers, units and footnotes are kept intact.

---

### Stage 2 — Structural Evidence Graph (SEG)

Nodes represent document elements; edges capture relationships:

- **Parent-child** — paragraph belongs to a section
- **Sequential** — element order within the document
- **Cross-reference** — "See Schedule 14.2" links to that schedule
- **Structural** — cell belongs to a row and column
- **Semantic** — passage and table row share a subject

When retrieval surfaces a paragraph that says *"See Schedule 14.2 for provisions"*, the graph traversal fetches the actual provision table — not just the pointer.

---

### Stage 3 — Hybrid Retrieval with RRF

Two complementary search channels feed into Reciprocal Rank Fusion:

```
RRF(d) = 1 / (60 + r_dense(d)) + 1 / (60 + r_BM25(d))
```

| Channel | Strength |
|---|---|
| **FAISS** (dense, semantic) | Matches intent even when wording differs |
| **BM25** (sparse, lexical) | Exact account labels, company names, financial identifiers |

---

### Stage 4 — Adaptive Query Routing

The query classifier routes to one of two modes:

| Mode | Example query | Behaviour |
|---|---|---|
| **Targeted** | "Capital adequacy ratio FY2025?" | Narrow doc pool; locate specific metric + footnote |
| **Synthesis** | "Explain main credit-risk issues and mitigations" | Collect complementary multi-section evidence |

Metadata filters are only applied when constraints are reliably present in the document.

---

### Stage 5 — Obligation-Aware Evidence Ledger

The ledger decomposes the query into individual **obligations** (required facts) and tracks coverage:

| Obligation | Evidence required |
|---|---|
| Entity identity | Identification of the named company |
| NPA status | Classification + reporting period |
| Credit exposure | Source value, unit, scope, page |
| Provisions | Provision amount + supporting table or note |
| Coverage ratio | Verified numerator and denominator |

Each candidate chunk is scored on **marginal utility**:

```
U(c | S) = NewCoverage(c) - lambda * Redundancy(c, S) + gamma * UsefulGraphLinks(c)
```

Where `S` is the currently selected evidence set. This means:
- A passage that covers a **missing obligation** beats fifteen near-identical paragraphs.
- Graph links can recover a referenced schedule that similarity search never surfaced.

---

## API Reference

The server exposes exactly the two endpoints required by the hackathon specification, plus utility endpoints.

### `POST /index`

Ingest a ZIP archive of PDFs into a named collection.

```bash
curl -X POST http://localhost:8000/index \
  -F "file=@annual_reports.zip" \
  -F "collection_name=bank_fy2025"
```

**Response**
```json
{
  "status": "success",
  "collection_name": "bank_fy2025",
  "num_documents": 3,
  "num_chunks": 1842,
  "elapsed_seconds": 47.3
}
```

---

### `POST /query`

Query a named collection and receive a grounded answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "bank_fy2025",
    "query": "What was the NPA coverage ratio and provisions made in FY2025?",
    "top_chunks": 30,
    "provider": "opencode"
  }'
```

**Response**
```json
{
  "answer": "The NPA coverage ratio was 78.5%...",
  "chunks": [
    {
      "text": "...",
      "source": "annual_report_2025.pdf",
      "page": 88,
      "section": "Asset Quality"
    }
  ],
  "trace": {}
}
```

---

### Additional Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query/stream` | Stream answer token-by-token via SSE |
| `POST` | `/cache/clear` | Clear the in-memory query cache |
| `GET` | `/collections` | List all indexed collections |
| `GET` | `/collection/{name}/status` | Check indexing status |
| `GET` | `/health` | Health check |

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Answer precision** | Of all factual claims generated, what proportion is correct? |
| **Answer recall** | Of all benchmark answer points, what proportion did we include? |
| **Context precision** | Of all retrieved chunks, what proportion is relevant? |
| **Context recall** | Of all benchmark-relevant chunks, what proportion did we retrieve? |

The graph-based evidence recovery targets **context recall** (finding what similarity search misses). The ledger targets **context precision** (fewer redundant chunks).

---

## Project Structure

```
NullRAG/
├── server.py                    # FastAPI server (POST /index, POST /query)
├── config.py                    # API keys and Langfuse config
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # End-to-end pipeline demo
├── run_benchmark_eval.py        # Local benchmark evaluation
├── run_api_benchmark_eval.py    # API-based benchmark evaluation
├── run_opencode_judge.py        # LLM-as-judge evaluation
│
├── hierfinrag/                  # Core pipeline modules
│   ├── parsing/                 # PDF ingestion, hierarchy extraction
│   ├── graph/                   # Structural Evidence Graph (SEG)
│   ├── index/                   # FAISS + BM25 indexing
│   ├── retrieval/               # Hybrid retrieval, RRF fusion
│   ├── routing/                 # Adaptive query router
│   ├── intent/                  # Query intent classification
│   ├── evidence/                # Obligation-Aware Evidence Ledger
│   ├── reasoning/               # Numerical reasoning helpers
│   ├── generation/              # LLM answer generation with citations
│   ├── evaluation/              # Metric computation
│   └── observability/           # Langfuse tracing
│
├── nullrag/                     # Engine and collection store
├── tests/                       # Unit and integration tests
└── results/                     # Benchmark output logs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config.py`:
```python
OPENCODE_API_KEY = "your-key-here"   # Primary LLM provider
GEMINI_API_KEY   = ""                 # Optional
OPENAI_API_KEY   = ""                 # Optional

# Langfuse observability (optional)
LANGFUSE_PUBLIC_KEY = ""
LANGFUSE_SECRET_KEY  = ""
LANGFUSE_HOST        = "https://cloud.langfuse.com"
```

### 3. Start the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or:
```bash
python server.py
```

### 4. Index Documents

```bash
curl -X POST http://localhost:8000/index \
  -F "file=@your_reports.zip" \
  -F "collection_name=my_collection"
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "my_collection", "query": "What is the NPA ratio?"}'
```

### 6. Run Pipeline Demo

```bash
python run_pipeline.py
```

### 7. Run Benchmark Evaluation

```bash
python run_benchmark_eval.py
```

---

## The 60-Second Pitch

Financial documents are difficult for conventional RAG systems because their information is distributed across narrative sections, tables, footnotes and different reporting periods. A system may retrieve many relevant-looking passages yet still miss the exact evidence required to answer a question completely.

**HierFinRAG** addresses this by treating the document as a structured evidence network rather than a collection of text chunks:

1. **Preserve hierarchy & provenance** — every chunk is traceable to its source page
2. **Structural Evidence Graph** — connects sections, tables, cells and cross-references
3. **Obligation-Aware Ledger** — tracks what the question *requires*, selects complementary evidence, follows graph links to recover missing information
4. **Grounded generation** — LLM produces cited explanations anchored to retrieved evidence

**Goal**: improve evidence completeness and factual precision while maintaining traceability, controlled latency and measurable cost.

---

## Questions Judges May Ask

**Q: How is this different from ordinary hybrid RAG?**
Hybrid retrieval is only our candidate-discovery stage. Our distinct contribution is the combination of structural evidence relationships and obligation-aware evidence selection.

**Q: How does the graph improve recall?**
A retrieved passage can contain a cross-reference to a table that similarity search did not return. The graph can recover that evidence and make it available to the coverage selector.

**Q: How do you avoid retrieving many repeated paragraphs?**
The ledger tracks which requirements are already supported. A passage covering a missing fact receives greater marginal value than a redundant one.


**Q: How do you handle conflicting figures or reporting periods?**
We retain provenance for candidate values and check entity, period, units and scope. If a discrepancy cannot be resolved, we preserve it as unresolved.

---

## License

Developed for the HierFinRAG Hackathon. All financial figures in documentation are illustrative only and do not represent real company data.
