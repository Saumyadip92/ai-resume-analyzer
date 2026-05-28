# app.py
"""Streamlit UI for the Resume Auditor.

Two-pane app:
- Inputs: PDF resume upload + pasted job description text
- Outputs: Verification Table (per-requirement verdicts with citations)
            + Honesty Audit (vague claims with sharpening suggestions)
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.orchestrator import run_audit


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Auditor",
    page_icon="📄",
    layout="wide",
)


VERDICT_STYLES = {
    "strong":  {"emoji": "🟢", "label": "Strong", "color": "#22c55e"},
    "partial": {"emoji": "🟡", "label": "Partial", "color": "#eab308"},
    "weak":    {"emoji": "🟠", "label": "Weak", "color": "#f97316"},
    "missing": {"emoji": "🔴", "label": "Missing", "color": "#ef4444"},
}

# Sort order for verification results: show gaps first.
VERDICT_ORDER = {"missing": 0, "weak": 1, "partial": 2, "strong": 3}


WEAKNESS_LABELS = {
    "vague_metric":        "📊 Vague metric",
    "unquantified_impact": "📉 Unquantified impact",
    "weak_verb":           "💤 Weak verb",
    "unspecified_scope":   "🔍 Unspecified scope",
    "buzzword_soup":       "🍜 Buzzword soup",
    "unclear_role":        "👤 Unclear role",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _render_verification(v: dict):
    """Render a single verification entry as an expander."""
    style = VERDICT_STYLES.get(v["verdict"], VERDICT_STYLES["missing"])
    with st.expander(
        f"{style['emoji']} **{style['label']}** — {v['requirement']}"
    ):
        st.markdown(f"**Reasoning:** {v['reasoning']}")
        if v["evidence_quotes"]:
            st.markdown("**Evidence from resume:**")
            for quote in v["evidence_quotes"]:
                st.info(f"📝 {quote}")
        else:
            st.warning("No evidence found in the retrieved resume chunks.")


def _render_audit_finding(f: dict):
    """Render a single audit finding as an expander."""
    label = WEAKNESS_LABELS.get(f["weakness"], f["weakness"])
    # Smart truncation: cut at word boundary
    claim_preview = f["claim"][:70]
    if len(f["claim"]) > 70:
        claim_preview = claim_preview.rsplit(" ", 1)[0] + "..."
    with st.expander(f"{label} — \"{claim_preview}\""):
        st.markdown("**Original claim:**")
        st.warning(f["claim"])
        st.markdown(f"**Why it's weak:** {f['explanation']}")
        st.markdown("**Sharpened rewrite:**")
        st.code(f["suggestion"], language="markdown")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 Resume Auditor")
st.markdown(
    "*Evidence-based JD matching with grounded citations.* "
    "Not keyword matching — this auditor **reasons about evidence** "
    "and cites the exact resume lines that support each verdict."
)

st.divider()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Upload your resume")
    uploaded_pdf = st.file_uploader(
        "PDF only. Single-column resumes parse most reliably.",
        type=["pdf"],
        accept_multiple_files=False,
    )

with col_right:
    st.subheader("2. Paste the job description")
    jd_text = st.text_area(
        "Full JD text. The auditor will decompose this into atomic, "
        "testable requirements.",
        height=240,
        placeholder="Paste the full job description here...",
    )

st.divider()
run_clicked = st.button(
    "🔍 Run Audit", type="primary", use_container_width=True
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked:
    if uploaded_pdf is None:
        st.error("Please upload a resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Streamlit's file_uploader gives us an in-memory file. PyMuPDF's open()
    # accepts a path string, so we write the bytes to a temp file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = Path(tmp.name)

    progress_bar = st.progress(0.0, text="Starting...")

    def update_progress(step: str, fraction: float):
        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=step)

    try:
        result = run_audit(tmp_path, jd_text, progress_callback=update_progress)
    except Exception as e:
        st.error(f"Audit failed: {e}")
        st.stop()
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    progress_bar.empty()

    # -----------------------------------------------------------------------
    # Match Score (the single most impactful number)
    # -----------------------------------------------------------------------
    summary = result["summary"]
    total = summary["total"] or 1  # avoid division by zero

    # Weighted score: strong=1.0, partial=0.5, weak=0.25, missing=0.0
    score = (
        summary["strong"] * 1.0
        + summary["partial"] * 0.5
        + summary["weak"] * 0.25
    ) / total * 100

    # Color the score based on value
    if score >= 70:
        score_color = "#22c55e"
    elif score >= 40:
        score_color = "#eab308"
    else:
        score_color = "#ef4444"

    st.markdown(
        f"<h2 style='text-align:center; margin-bottom:0;'>"
        f"Match Score: <span style='color:{score_color}'>{score:.0f}%</span>"
        f"</h2>"
        f"<p style='text-align:center; color:#888; margin-top:0;'>"
        f"Weighted: Strong=100%, Partial=50%, Weak=25%, Missing=0%"
        f"</p>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Summary breakdown
    # -----------------------------------------------------------------------
    st.subheader("Breakdown")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", summary["total"])
    c2.metric("🟢 Strong", summary["strong"])
    c3.metric("🟡 Partial", summary["partial"])
    c4.metric("🟠 Weak", summary["weak"])
    c5.metric("🔴 Missing", summary["missing"])

    st.divider()

    # -----------------------------------------------------------------------
    # Tabs for the two main panels
    # -----------------------------------------------------------------------
    tab_verify, tab_audit = st.tabs(
        ["📋 Verification Table", "🔎 Honesty Audit"]
    )

    # ---- Verification Table -------------------------------------------------
    with tab_verify:
        st.markdown(
            "Each JD requirement was verified against retrieved resume chunks. "
            "Evidence quotes are taken **verbatim** from the resume."
        )

        # Sort: missing first (gaps at top), strong last
        sorted_verifications = sorted(
            result["verifications"],
            key=lambda v: VERDICT_ORDER.get(v["verdict"], 99),
        )

        # Group by priority: must_have first
        must_have = [v for v in sorted_verifications if v["priority"] == "must_have"]
        nice_to_have = [v for v in sorted_verifications if v["priority"] != "must_have"]

        if must_have:
            st.markdown("#### 🔒 Required")
            for v in must_have:
                _render_verification(v)

        if nice_to_have:
            st.markdown("#### ✨ Nice to have")
            for v in nice_to_have:
                _render_verification(v)

    # ---- Honesty Audit ------------------------------------------------------
    with tab_audit:
        st.markdown(
            "Claims in your resume that an interviewer would push back on, "
            "with concrete sharpening suggestions."
        )
        if not result["audit_findings"]:
            st.success("No vague claims found. Resume reads strong. 💪")
        else:
            st.caption(
                f"Found {len(result['audit_findings'])} claims to sharpen."
            )
            for f in result["audit_findings"]:
                _render_audit_finding(f)

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.divider()
    st.caption(
        "Built by [Saumyadip](https://github.com/Saumyadip92) · "
        "Powered by Gemini + sentence-transformers · "
        "Evidence-based, not keyword-based."
    )

else:
    # Landing state — show how-it-works when no audit has been run yet
    st.markdown(
        """
        ### How it works

        1. **Upload** your resume as a PDF
        2. **Paste** the full job description
        3. **Click Run** — the auditor will:
           - Decompose the JD into atomic requirements
           - Search your resume for evidence of each requirement
           - Return a verdict with literal citations
           - Flag vague claims and suggest sharper rewrites

        *Typical run: 30-60 seconds depending on JD length.*
        """
    )
