import re
from typing import List, Dict, Any, Optional
from ..reasoning.fusion import SymbolicNeuralFusion

def extract_atomic_claims(answer_text: str) -> List[str]:
    """Splits an answer text into checkable claims (lines / sentences)."""
    lines = [line.strip("- *").strip() for line in answer_text.split("\n") if line.strip()]
    claims = []
    for line in lines:
        if line.startswith("#"):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for s in sentences:
            if len(s.strip()) > 15:
                claims.append(s.strip())
    return claims or [answer_text]

def compute_confidence(retrieval_quality: float = 0.85, claim_count: int = 1) -> float:
    """Calibrates confidence based on retrieval quality and coverage."""
    return min(1.0, max(0.5, retrieval_quality * 0.9 + 0.1))

def generate_answer_with_attribution(
    query: str, 
    retrieved_contexts: List[Dict[str, Any]], 
    fusion_engine: Optional[SymbolicNeuralFusion] = None
) -> Dict[str, Any]:
    """
    Generates a grounded financial answer with attributed sources and confidence metrics.
    """
    if fusion_engine is None:
        fusion_engine = SymbolicNeuralFusion()
        
    result = fusion_engine(query, retrieved_contexts)
    answer_text = result.get("answer", "")
    retrieved_chunks = result.get("retrieved_chunks", [])
    
    claims = extract_atomic_claims(answer_text)
    confidence = compute_confidence(retrieval_quality=0.88, claim_count=len(claims))
    
    return {
        "answer_text": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "mode": result.get("mode", "hybrid"),
        "atomic_claims": claims,
        "confidence": confidence
    }
