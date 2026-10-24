"""
nullRAG Package
===============
Hierarchical Multimodal RAG for Financial Document Understanding.
"""
import sys
import importlib

# Ensure hierfinrag is loaded and mirror its namespaces
import hierfinrag

# Forward all attributes from hierfinrag
for attr in dir(hierfinrag):
    if not attr.startswith("__"):
        globals()[attr] = getattr(hierfinrag, attr)

# Custom import hook or module aliasing for subpackages
_submodules = [
    "evaluation", "evidence", "generation", "graph", 
    "index", "intent", "observability", "parsing", 
    "reasoning", "retrieval", "routing"
]

for sub in _submodules:
    try:
        mod = importlib.import_module(f"hierfinrag.{sub}")
        sys.modules[f"nullrag.{sub}"] = mod
        globals()[sub] = mod
    except ImportError:
        pass
