# src/jd_decomposer.py
"""Decompose a job description into a list of atomic, testable requirements.

Uses Gemini with structured output (JSON schema) to guarantee a parseable
response shape regardless of how the JD is phrased.
"""

import json
from typing import List, Dict, Any

from google.genai import types


from src.llm_client import generate_with_retry



# JSON schema for the response. The Gemini API will guarantee that
# response.text is a JSON string conforming to this schema.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "Atomic, testable statement extracted from the JD."
                    },
                    "type": {
                        "type": "string",
                        "enum": ["technical", "experience", "education", "soft_skill"],
                        "description": "Category of the requirement."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["must_have", "nice_to_have"],
                        "description": "Whether the JD marks this as required or preferred."
                    }
                },
                "required": ["requirement", "type", "priority"],
            }
        }
    },
    "required": ["requirements"],
}


SYSTEM_INSTRUCTION = """You are a strict job-description analyzer.
Your only job is to decompose a job description into a list of atomic,
testable requirements that could each be verified against a candidate's
resume.

Rules:
- Each requirement must be a single, concrete claim (one skill, one
  qualification, one experience type per item). Split compound bullets like
  "Python and Java" into two separate requirements.
- Skip marketing fluff (company culture, perks, "join our team!").
- Skip vague aspirations ("passionate about X") unless they correspond to a
  concrete verifiable behavior.
- Preserve quantifiers: "3+ years experience with Python" stays as a single
  requirement; do not drop the "3+ years" part.
- Mark items the JD explicitly calls "required" / "must" as must_have;
  items under "preferred" / "nice to have" / "bonus" as nice_to_have.
- If priority is unclear, default to must_have.
- Return between 5 and 25 requirements depending on JD length.
"""


def decompose_jd(jd_text: str) -> List[Dict[str, Any]]:
    """Decompose a JD into a list of structured requirements.

    Returns a list of dicts with keys: requirement, type, priority.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("decompose_jd called with empty JD text")

    response = generate_with_retry(
        contents=jd_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    # response.text is guaranteed to be a JSON string matching RESPONSE_SCHEMA.
    parsed = json.loads(response.text)
    return parsed["requirements"]
