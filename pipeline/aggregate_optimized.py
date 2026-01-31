# pipeline/aggregate_optimized.py
"""
Optimized classification pipeline with async/parallel processing and caching.

This module provides the core classification functionality with:
- Parallel execution of independent operations
- In-memory caching for repeated queries
- Batch processing support
- Graceful error handling
"""

import asyncio
import hashlib
from typing import Dict, Optional
from pipeline.extract_claim_async import extract_claim_async
from pipeline.retrieve_context import retrieve_context
from pipeline.map_trope_async import map_trope_async
from pipeline.counterfactual_async import counterfactual_test_async


# Simple in-memory cache (can be upgraded to Redis for production)
_cache: Dict[str, Dict] = {}
CACHE_SIZE = 1000


def _get_cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.md5(text.encode()).hexdigest()


def _get_from_cache(text: str) -> Optional[Dict]:
    """Get result from cache."""
    key = _get_cache_key(text)
    return _cache.get(key)


def _set_cache(text: str, result: Dict):
    """Set result in cache."""
    key = _get_cache_key(text)
    if len(_cache) >= CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        _cache.pop(next(iter(_cache)))
    _cache[key] = result


async def classify_text_async(text: str, use_cache: bool = True) -> dict:
    """Classify text for antisemitic content using async parallel processing.

    Parameters
    ----------
    text
        Input text to analyze for antisemitic rhetoric.
    use_cache
        If True, use in-memory cache for repeated queries (default: True).

    Returns
    -------
    dict
        Classification result containing:
        - verdict: Risk level classification
        - risk_score: Numeric risk score (0.0-1.0)
        - trope: Detected antisemitic trope type
        - trope_strength: Strength of trope match (0.0-1.0)
        - explanation: Alternative non-antisemitic interpretation
        - reasoning: Brief explanation of assessment
        - details: Additional analysis details
    """
    # Check cache first
    if use_cache:
        cached = _get_from_cache(text)
        if cached:
            return cached
    
    # Early exit: very short or obviously benign text
    if len(text.strip()) < 10:
        result = {
            "verdict": "Low-risk / non-identity-based",
            "risk_score": 0.0,
            "trope": "none",
            "trope_strength": 0.0,
            "explanation": "Text too short for meaningful analysis.",
            "reasoning": "",
            "details": {
                "extracted_claim": text,
                "target": "unclear",
                "explicitness": "implicit",
                "counterfactual": "",
                "meaning_preserved": True,
                "counterfactual_explanation": ""
            }
        }
        if use_cache:
            _set_cache(text, result)
        return result
    
    # Run independent operations in parallel
    try:
        # These can all run in parallel since they don't depend on each other
        claim_task = extract_claim_async(text)
        counterfactual_task = counterfactual_test_async(text)
        
        # Wait for claim extraction first (needed for retrieval)
        claim_data = await claim_task
        claim = claim_data.get("claim", text)
        target = claim_data.get("target", "unclear")
        explicitness = claim_data.get("explicitness", "implicit")
        
        # Now run retrieval and counterfactual in parallel
        # retrieve_context is synchronous, so run it in thread pool
        docs_task = asyncio.to_thread(retrieve_context, claim)
        counterfactual_task_result = await counterfactual_task
        
        # Wait for retrieval (runs in parallel with counterfactual)
        docs = await docs_task
        
        # Now run trope mapping (needs docs)
        trope_map = await map_trope_async(claim, docs)
        
    except Exception as e:
        # Fallback on error
        claim = text
        target = "unclear"
        explicitness = "implicit"
        trope_map = {
            "mapped_trope": "none",
            "trope_strength": 0.0,
            "alternative_interpretation": "Error in analysis",
            "reasoning": f"Error: {str(e)}"
        }
        counterfactual_task_result = {
            "counterfactual_claim": "",
            "meaning_preserved": True,
            "explanation": f"Error: {str(e)}"
        }
        docs = []

    # Calculate risk score (same logic as before)
    trope_strength = trope_map.get("trope_strength", 0.0)
    meaning_preserved = counterfactual_task_result.get("meaning_preserved", True)
    mapped_trope = trope_map.get("mapped_trope", "none")

    # Improved risk score calculation
    base_score = trope_strength
    
    if meaning_preserved:
        counterfactual_multiplier = 0.3
    else:
        counterfactual_multiplier = 1.0
    
    if target == "explicit_jews":
        target_multiplier = 1.2
    elif target == "implicit_jews":
        target_multiplier = 1.0
    else:
        target_multiplier = 0.8
    
    if explicitness == "explicit":
        explicitness_multiplier = 1.1
    else:
        explicitness_multiplier = 1.0
    
    risk_score = min(1.0, base_score * counterfactual_multiplier * target_multiplier * explicitness_multiplier)
    
    if mapped_trope != "none" and trope_strength > 0.7:
        if meaning_preserved:
            risk_score = max(risk_score, trope_strength * 0.5)

    if risk_score > 0.6:
        verdict = "High-risk trope-based rhetoric"
    elif risk_score > 0.3:
        verdict = "Ambiguous — requires context"
    else:
        verdict = "Low-risk / non-identity-based"

    result = {
        "verdict": verdict,
        "risk_score": round(risk_score, 2),
        "trope": mapped_trope,
        "trope_strength": round(trope_strength, 2),
        "explanation": trope_map.get(
            "alternative_interpretation",
            "No clear identity-based implication detected."
        ),
        "reasoning": trope_map.get("reasoning", ""),
        "details": {
            "extracted_claim": claim,
            "target": target,
            "explicitness": explicitness,
            "counterfactual": counterfactual_task_result.get("counterfactual_claim", ""),
            "meaning_preserved": meaning_preserved,
            "counterfactual_explanation": counterfactual_task_result.get("explanation", "")
        }
    }
    
    # Cache result
    if use_cache:
        _set_cache(text, result)
    
    return result


def classify_text(text: str, use_cache: bool = True) -> dict:
    """Classify text for antisemitic content (synchronous wrapper).

    Parameters
    ----------
    text
        Input text to analyze for antisemitic rhetoric.
    use_cache
        If True, use in-memory cache for repeated queries (default: True).

    Returns
    -------
    dict
        Classification result containing:
        - verdict: Risk level classification
        - risk_score: Numeric risk score (0.0-1.0)
        - trope: Detected antisemitic trope type
        - trope_strength: Strength of trope match (0.0-1.0)
        - explanation: Alternative non-antisemitic interpretation
        - reasoning: Brief explanation of assessment
        - details: Additional analysis details
    """
    try:
        # Check if we're in an async context (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
        # If we're in an async context, try to use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(classify_text_async(text, use_cache))
        except ImportError:
            raise RuntimeError(
                "classify_text() cannot be called from an async context. "
                "Install nest-asyncio (pip install nest-asyncio) or use await classify_text_async() instead."
            )
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(classify_text_async(text, use_cache))


async def classify_texts_batch(texts: list[str], use_cache: bool = True) -> list[dict]:
    """Process multiple texts in parallel for batch classification.

    Parameters
    ----------
    texts
        List of input texts to analyze for antisemitic rhetoric.
    use_cache
        If True, use in-memory cache for repeated queries (default: True).

    Returns
    -------
    list[dict]
        List of classification results, one per input text. Each result
        contains the same structure as classify_text_async().
    """
    tasks = [classify_text_async(text, use_cache) for text in texts]
    return await asyncio.gather(*tasks)

