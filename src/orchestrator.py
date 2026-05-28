# src/orchestrator.py
"""Top-level orchestration: run the full audit pipeline end to end.

This is the only function the UI needs to know about. Internally it stitches
together extraction, chunking, embeddings, retrieval, JD decomposition,
per-requirement verification, and the resume audit.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.pdf_extract import extract_text
from src.chunker import chunk_text
from src.embedder import build_index, retrieve
from src.jd_decomposer import decompose_jd
from src.verifier import verify_requirement
from src.auditor import audit_resume


# How many resume chunks to retrieve per JD requirement and feed to the verifier.
# 3 is a good default: enough context for nuance, few enough to keep prompts short.
TOP_K = 3


def run_audit(
    pdf_path: str | Path,
    jd_text: str,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run the full audit pipeline end to end."""

    def progress(step: str, fraction: float):
        if progress_callback is not None:
            progress_callback(step, fraction)

    progress("Extracting resume text...", 0.05)
    resume_text = extract_text(pdf_path)

    progress("Chunking resume...", 0.10)
    chunks = chunk_text(resume_text)

    progress("Building embeddings index...", 0.20)
    index = build_index(chunks)

    progress("Decomposing job description...", 0.30)
    requirements = decompose_jd(jd_text)

    # Cache: maps (requirement_text, chunks_signature) -> verification result.
    # Lets us re-run an audit cheaply when only some calls failed last time,
    # or when the user hits "Run" again on the same inputs.
    verification_cache: Dict[str, Dict[str, Any]] = {}

    verifications: List[Dict[str, Any]] = []
    n = max(1, len(requirements))
    for i, req in enumerate(requirements):
        progress(
            f"Verifying requirement {i + 1}/{n}: {req['requirement'][:60]}...",
            0.30 + 0.55 * (i / n),
        )
        chunks_with_scores = retrieve(index, req["requirement"], k=TOP_K)

        # Cache key: requirement + the actual chunk texts retrieved.
        # If the same requirement retrieves the same chunks, skip the LLM call.
        cache_key = req["requirement"] + "||" + "||".join(c for c, _ in chunks_with_scores)
        if cache_key in verification_cache:
            result = verification_cache[cache_key]
        else:
            result = verify_requirement(req["requirement"], chunks_with_scores)
            verification_cache[cache_key] = result

        verifications.append({
            "requirement": req["requirement"],
            "type": req["type"],
            "priority": req["priority"],
            **result,
        })

    progress("Auditing resume for vague claims...", 0.90)
    audit_findings = audit_resume(resume_text)

    progress("Done", 1.0)

    summary = _summarize(verifications)

    return {
        "resume_text": resume_text,
        "chunk_count": len(chunks),
        "requirements": requirements,
        "verifications": verifications,
        "audit_findings": audit_findings,
        "summary": summary,
    }


def _summarize(verifications: List[Dict[str, Any]]) -> Dict[str, int]:
    """Tally verdicts for the summary card at the top of the UI."""
    counts = {"strong": 0, "partial": 0, "weak": 0, "missing": 0}
    for v in verifications:
        verdict = v.get("verdict", "missing")
        if verdict in counts:
            counts[verdict] += 1
    counts["total"] = sum(counts.values())
    return counts
