# pipeline/extract_claim_async.py
"""
Async claim extraction module with optimized JSON parsing.
"""

import json
import os
import re
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

SYSTEM_PROMPT = """You are a linguistic analysis agent specialized in extracting claims and identifying targets.

Your task is to:
- Extract the main claim from the text
- Identify the target of the claim (if any)
- Assess the explicitness of identity references

You must:
- Be precise and objective
- Do NOT judge intent or harm
- Focus on what is stated or implied, not interpretation
- Consider both explicit mentions and implicit references

Target categories:
- explicit_jews: Text explicitly mentions Jews, Jewish people, or Jewish identity
- implicit_jews: Text uses coded language, tropes, or references that commonly target Jews
- other: Text targets a different group or entity
- unclear: Target cannot be determined or is ambiguous

Explicitness:
- explicit: Direct, clear mention of identity or group
- implicit: Indirect, coded, or implied reference
"""

HUMAN_PROMPT = """Text:
"{text}"

Analyze the text and extract:
1. The main claim being made
2. The target of the claim (if any)
3. Whether identity references are explicit or implicit

Return ONLY valid JSON with:
- claim (string): The main claim extracted from the text
- target (explicit_jews | implicit_jews | other | unclear): Who or what the claim targets
- explicitness (explicit | implicit): How directly identity is referenced
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def safe_json_load(text: str) -> dict:
    """Extract and parse JSON from text."""
    # Try direct JSON first (faster)
    text = text.strip()
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


async def extract_claim_async(text: str) -> dict:
    """Extract main claim and identify target from text (async version).

    Parameters
    ----------
    text
        Input text to analyze.

    Returns
    -------
    dict
        Extracted claim data containing:
        - claim: Main claim extracted from text
        - target: Target category (explicit_jews | implicit_jews | other | unclear)
        - explicitness: Explicitness level (explicit | implicit)
    """
    try:
        response = await llm.ainvoke(prompt.format_messages(text=text))
        return safe_json_load(response.content)
    except Exception as e:
        # Return safe defaults on error
        return {
            "claim": text,
            "target": "unclear",
            "explicitness": "implicit",
            "error": str(e)
        }


def extract_claim(text: str) -> dict:
    """Extract main claim and identify target from text (synchronous wrapper).

    Parameters
    ----------
    text
        Input text to analyze.

    Returns
    -------
    dict
        Extracted claim data containing:
        - claim: Main claim extracted from text
        - target: Target category (explicit_jews | implicit_jews | other | unclear)
        - explicitness: Explicitness level (explicit | implicit)
    """
    import asyncio
    return asyncio.run(extract_claim_async(text))

