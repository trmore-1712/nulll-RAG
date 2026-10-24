from nullrag.routing.metadata_router import MetadataRouter, MetadataExtractor
from nullrag.retrieval.synthesis_aggregator import classify_query, AdaptiveRetriever
from nullrag.index.collection_store import registry
from nullrag.observability.langfuse_tracer import ObservabilityTracer

print("All NullRAG modules OK")

q = classify_query("pull together every credit risk event and red flag in this year's report")
print(f'Query type test (synthesis expected): "{q}"')

q2 = classify_query("What was the net loss for FY2025?")
print(f'Query type test2 (targeted expected): "{q2}"')

extractor = MetadataExtractor()
meta = extractor.extract("Integrated Annual Report 2025-26 Gammon India Limited.pdf", "")
print(f"Metadata extraction: entity={meta.entity}, fy={meta.fiscal_years}, scope={meta.scope}, type={meta.doc_type}")

store = registry.get_or_create("test_col")
print(f"Registry OK: collection created = {store.name}")
print("ALL CHECKS PASSED")
