# pipeline/counterfactual.py
"""
Counterfactual reasoning module.

Tests whether claims rely on implicit identity-based meaning by creating
counterfactual versions with neutral actors.
"""

import json
import re
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# LLM setup
# -----------------------------
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o",
    temperature=0
)


# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a counterfactual reasoning agent specialized in detecting identity-based implications.

Your task is to test whether a claim relies on implicit identity-based meaning by creating
a counterfactual version with neutral actors.

You must:
- Rewrite the claim using neutral, non-identity-based actors (e.g., "powerful groups", "certain individuals", "institutions")
- Compare meanings carefully between original and counterfactual
- Allow ambiguity where appropriate
- Be precise: if the claim's meaning fundamentally changes when identity is removed, it likely depends on identity

Key principle: If a claim can be rewritten with neutral actors while preserving its core meaning,
it may not be identity-dependent. If the meaning is lost or fundamentally changed, the claim likely
depends on implicit identity assumptions.
"""

HUMAN_PROMPT = """
Original claim:
"{claim}"

Steps:
1. Rewrite the claim using neutral, non-identity-based actors (avoid any ethnic, religious, or identity-specific references).
2. Compare the rewritten claim to the original carefully.
3. Decide whether the original meaning depends on an implied identity group:
   - If meaning is preserved: the claim may not be identity-dependent
   - If meaning is lost or fundamentally changed: the claim likely depends on identity assumptions
4. Provide clear explanation of your reasoning.

Examples:
- "They control the media" → "A group controls the media" (meaning preserved = True, not identity-dependent)
- "Jews control the media" → "A group controls the media" (meaning preserved = True, but explicit identity)
- "The Rothschilds control banking" → "A family controls banking" (meaning may change if Rothschilds are a proxy)

Return ONLY valid JSON with the following fields:
- counterfactual_claim (string)
- meaning_preserved (true | false)
- explanation (string, explain why meaning is or isn't preserved)
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# -----------------------------
# Helper: safe JSON parsing
# -----------------------------
def safe_json_load(text: str) -> dict:
    """
    Extracts the first JSON object from a string and parses it.
    Raises ValueError if no JSON is found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())


# -----------------------------
# Main function
# -----------------------------
def counterfactual_test(claim: str) -> dict:
    """Test whether a claim relies on implicit identity-based meaning using counterfactual reasoning.

    Parameters
    ----------
    claim
        Claim text to test for identity-dependence.

    Returns
    -------
    dict
        Counterfactual test result containing:
        - counterfactual_claim: Rewritten claim with neutral actors
        - meaning_preserved: Whether meaning is preserved in counterfactual (bool)
        - explanation: Explanation of why meaning is or isn't preserved
    """

    response = llm.invoke(
        prompt.format_messages(
            claim=claim
        )
    )

    try:
        return safe_json_load(response.content)
    except Exception as e:
        return {
            "counterfactual_claim": "",
            "meaning_preserved": None,
            "explanation": "Unable to determine due to parsing error.",
            "error": str(e),
            "raw_output": response.content
        }
