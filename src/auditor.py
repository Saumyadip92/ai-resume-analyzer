# src/auditor.py
"""Audit a resume for vague or unsubstantiated claims.

JD-agnostic. Given the full resume text, returns a list of findings:
specific weak claims with a category, explanation, and concrete rewrite.
"""

import json
from typing import List, Dict, Any

from google.genai import types

from src.llm_client import generate_with_retry


WEAKNESS_TYPES = [
    "vague_metric",
    "unquantified_impact",
    "weak_verb",
    "unspecified_scope",
    "buzzword_soup",
    "unclear_role",
]


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "The literal vague claim from the resume. Must "
                            "appear verbatim in the input text."
                        ),
                    },
                    "weakness": {
                        "type": "string",
                        "enum": WEAKNESS_TYPES,
                        "description": "Category of the weakness.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": (
                            "One sentence explaining why this specific claim "
                            "is weak."
                        ),
                    },
                    "suggestion": {
                        "type": "string",
                        "description": (
                            "A concrete, sharpened rewrite of the claim. "
                            "Should be a model bullet the user could copy "
                            "after filling in their own real metrics."
                        ),
                    },
                },
                "required": ["claim", "weakness", "explanation", "suggestion"],
            },
        }
    },
    "required": ["findings"],
}


SYSTEM_INSTRUCTION = """You are a strict resume reviewer. Your job is to find
vague, unsubstantiated, or weak claims in a candidate's resume that an
experienced interviewer would push back on.

Weakness categories (use exactly these labels):
- vague_metric:        a number with no baseline, scope, or measurement context.
                       e.g. "Improved performance by 50%"
- unquantified_impact: a claim of impact with no numbers at all.
                       e.g. "Significantly improved user experience"
- weak_verb:           passive or generic action verbs that hide what was done.
                       e.g. "Was responsible for...", "Helped with...", "Worked on..."
- unspecified_scope:   words like "team", "users", "data" without size or context.
                       e.g. "Led a team", "Worked with large datasets"
- buzzword_soup:       multiple jargon words strung together with little substance.
                       e.g. "Leveraged synergistic AI-powered solutions"
- unclear_role:        the reader cannot tell what THIS person did vs. the team.
                       e.g. "Our team built X" without specifying the candidate's role

Rules:
1. Quote ONLY claims that appear verbatim in the resume. Do NOT paraphrase.
2. SKIP section headers, contact info, dates, education facts, and bare
   skill lists. Audit only sentences that make a claim of action or impact.
3. Find between 3 and 10 findings. If the resume is genuinely strong with
   no weak claims, return fewer (even zero). Do not invent weakness.
4. The "suggestion" must be a concrete rewrite the user could adapt. Do
   not write "add metrics" — write the rewritten bullet with placeholder
   metrics like "[X%]" or "[N users]" if specifics aren't knowable.
5. Each finding addresses ONE weakness. If a single claim has two issues,
   pick the most serious one.
"""


def audit_resume(resume_text: str) -> List[Dict[str, Any]]:
    """Audit a full resume for vague or unsubstantiated claims.

    Returns a list of findings, each a dict with:
    claim, weakness, explanation, suggestion.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("audit_resume called with empty resume text")

    user_prompt = f"""Audit the following resume. Find vague claims an
interviewer would push back on.

\"\"\"
{resume_text}
\"\"\"

Return your findings as JSON matching the response schema."""

    response = generate_with_retry(
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.3,
        ),
    )

    parsed = json.loads(response.text)
    return parsed["findings"]
