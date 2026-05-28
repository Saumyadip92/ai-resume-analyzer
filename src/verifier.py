# src/verifier.py
"""Verify a single JD requirement against retrieved resume chunks.

Uses Gemini with structured output (JSON schema) to produce a grounded
verdict: did we find evidence in the retrieved chunks, and if so, what
literal quotes support it?
"""

import json
from typing import List, Tuple, Dict, Any

from google.genai import types

from src.llm_client import generate_with_retry



RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["strong", "partial", "weak", "missing"],
            "description": (
                "strong = direct, demonstrated evidence with depth; "
                "partial = mentioned but thin (e.g. listed without project); "
                "weak = inferable from adjacent evidence but not directly shown; "
                "missing = no evidence at all in the provided chunks."
            ),
        },
        "evidence_quotes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Literal quotes from the chunks that support the verdict. "
                "Empty array when verdict=missing. Each quote must appear "
                "verbatim in the provided chunks."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One or two sentences explaining the verdict and how the "
                "evidence (or lack of it) supports the chosen verdict."
            ),
        },
    },
    "required": ["verdict", "evidence_quotes", "reasoning"],
}


SYSTEM_INSTRUCTION = """You are a strict resume auditor.
Your job: given ONE job-description requirement and a small set of retrieved
chunks from a candidate's resume, decide whether the resume contains evidence
of that requirement.

Verdict scale:
- strong:  direct, demonstrated evidence with specifics or depth.
           e.g. requirement "Python" with chunk "Built Flask API in Python serving 10k req/day"
- partial: skill is mentioned but evidence is thin (listed without context, no project).
           e.g. requirement "Python" with chunk "Programming Languages: Java, Python, C++"
- weak:    inferable from related evidence but not directly stated.
           e.g. requirement "REST APIs" with chunk "deployed Flask service for mobile team"
- missing: no evidence appears in the chunks.

Critical rules:
1. Quote ONLY text that appears verbatim in the provided chunks. Do NOT
   paraphrase, summarize, or invent. If you cannot find a literal quote,
   the verdict is missing.
2. When verdict=missing, evidence_quotes MUST be an empty array.
3. Be honest about thin evidence. Listing a skill in a skills section is
   partial at best, never strong.
4. Reasoning must reference the actual chunk content or its absence.
5. Do NOT speculate about what the candidate "probably" knows beyond what
   the chunks show.
6. When a requirement contains a quantifier (e.g. "2+ years", "5+ projects", "production-scale"), 
    the verdict must reflect whether THAT quantifier is met, not just whether the underlying skill is mentioned.
    A skill listed without the required quantifier evidence is at most weak, never partial or strong
"""


def verify_requirement(
    requirement: str,
    chunks_with_scores: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """Verify one requirement against a list of (chunk, similarity_score) tuples.

    Returns a dict with keys: verdict, evidence_quotes, reasoning.
    """
    if not requirement or not requirement.strip():
        raise ValueError("verify_requirement called with empty requirement")
    if not chunks_with_scores:
        # No retrieval results -> nothing to verify against.
        return {
            "verdict": "missing",
            "evidence_quotes": [],
            "reasoning": "No relevant resume chunks were retrieved for this requirement.",
        }

    # Format the chunks for the LLM. Numbering helps the model reason about
    # multiple chunks and lets the user trace quotes back if needed.
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}, similarity={score:.2f}]\n{chunk}"
        for i, (chunk, score) in enumerate(chunks_with_scores)
    )

    user_prompt = f"""Requirement to verify:
\"\"\"
{requirement}
\"\"\"

Retrieved resume chunks (the only source of truth — quote only from these):
\"\"\"
{chunks_text}
\"\"\"

Return your verdict as JSON matching the response schema."""

    response = generate_with_retry(
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    return json.loads(response.text)
