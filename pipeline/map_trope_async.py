# pipeline/map_trope_async.py
"""
Async trope mapping module with optimized parsing.
"""

import json
import re
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o",
    temperature=0
)

SYSTEM_PROMPT = """
You are a contextual reasoning agent specialized in identifying antisemitic tropes.

Your task is to determine whether a claim resembles a known antisemitic trope,
using ONLY the provided reference material.

You must:
- Ground your reasoning in the reference material
- Allow for ambiguity and context
- Always provide a plausible non-antisemitic interpretation
- Consider both explicit and implicit (coded) language
- Be precise in your trope_strength assessment

Reasoning process:
1. First, identify which trope (if any) the claim most closely resembles
2. Assess the strength of resemblance (0.0 = no match, 1.0 = clear match)
3. Consider alternative non-antisemitic interpretations
4. Provide clear reasoning for your assessment

If the claim does not clearly match any trope, return "none" with trope_strength 0.0.
"""

HUMAN_PROMPT = """
Claim:
"{claim}"

Reference material:
{context}

Instructions:
1. Carefully analyze the claim against the reference material
2. Determine whether the claim resembles a known antisemitic trope
3. If so, identify the specific trope from the available options
4. Estimate how strong the resemblance is on a scale from 0.0 to 1.0:
   - 0.0-0.2: No clear resemblance or very weak
   - 0.3-0.5: Some resemblance but ambiguous
   - 0.6-0.8: Strong resemblance with some ambiguity
   - 0.9-1.0: Clear, strong resemblance
5. Provide a plausible alternative interpretation that is NOT antisemitic
6. Include brief reasoning for your assessment

Available tropes:
- elite_control: Claims about Jews controlling institutions (media, finance, government)
- dual_loyalty: Claims about divided allegiance or loyalty conflicts
- collective_guilt: Holding all Jews responsible for actions of individuals/Israel
- financial_conspiracy: Financial manipulation conspiracies targeting Jews
- blood_libel: Accusations of harm or ritualistic violence
- holocaust_denial: Denial, minimization, or distortion of the Holocaust
- proxy_figures: Using specific individuals as stand-ins for broader conspiracies
- dogwhistle: Coded language that may convey antisemitic meaning
- religious_demonization: Framing Jews as evil or satanic
- none: No clear trope match

Return ONLY valid JSON with the following fields:
- mapped_trope (elite_control | dual_loyalty | collective_guilt | financial_conspiracy | blood_libel | holocaust_denial | proxy_figures | dogwhistle | religious_demonization | none)
- trope_strength (number between 0.0 and 1.0)
- alternative_interpretation (string)
- reasoning (string, brief explanation of your assessment)
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def safe_json_load(text: str) -> dict:
    """Extract and parse JSON from text."""
    text = text.strip()
    # Try direct JSON first (faster)
    if text.startswith('{'):
        try:
            return json.loads(text)
        except:
            pass
    
    # Fallback to regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())


async def map_trope_async(claim: str, retrieved_docs) -> dict:
    """Map a claim to a known antisemitic trope using retrieved knowledge base context (async version).

    Parameters
    ----------
    claim
        Extracted claim text to analyze.
    retrieved_docs
        List of retrieved knowledge base documents from RAG system.

    Returns
    -------
    dict
        Trope mapping result containing:
        - mapped_trope: Detected trope type or "none"
        - trope_strength: Strength of match (0.0-1.0)
        - alternative_interpretation: Plausible non-antisemitic interpretation
        - reasoning: Brief explanation of assessment
    """
    # Combine retrieved docs into a single context string
    context = "\n\n".join(
        [doc.text if hasattr(doc, "text") else str(doc) for doc in retrieved_docs]
    )

    try:
        # Invoke LLM asynchronously
        response = await llm.ainvoke(
            prompt.format_messages(
                claim=claim,
                context=context
            )
        )

        # Parse structured output safely
        return safe_json_load(response.content)
    except Exception as e:
        return {
            "mapped_trope": "none",
            "trope_strength": 0.0,
            "alternative_interpretation": "Unable to determine due to parsing error.",
            "reasoning": f"Error: {str(e)}",
            "error": str(e),
            "raw_output": str(e)
        }


def map_trope(claim: str, retrieved_docs) -> dict:
    """Map a claim to a known antisemitic trope using retrieved knowledge base context (synchronous wrapper).

    Parameters
    ----------
    claim
        Extracted claim text to analyze.
    retrieved_docs
        List of retrieved knowledge base documents from RAG system.

    Returns
    -------
    dict
        Trope mapping result containing:
        - mapped_trope: Detected trope type or "none"
        - trope_strength: Strength of match (0.0-1.0)
        - alternative_interpretation: Plausible non-antisemitic interpretation
        - reasoning: Brief explanation of assessment
    """
    import asyncio
    return asyncio.run(map_trope_async(claim, retrieved_docs))

