# Resume Auditor — Evidence-Based JD Matcher

An LLM-powered tool that audits a resume against a job description by checking
for **evidence** of each requirement in the resume, not just keyword matches.

Most resume analyzers do keyword/cosine similarity matching, which misses the
forest for the trees: a candidate who built a Flask API serving 10k req/day
clearly knows REST APIs, even if the literal word "REST" never appears in
their resume. This tool reasons about evidence, with citations.

## Status

🚧 In active development. Stage 1/7 complete (Gemini API integration).

## Architecture

PDF Resume ─► Text extraction ─► Chunking ─► Embeddings ─► Vector store │ JD Text ─► JD Decomposer (LLM) ─► Atomic requirements ──┐ │ ▼ ▼ Retrieve top-k resume chunks per requirement │ ▼ Verifier (LLM): verdict + citations │ ▼ Verification Table

Resume ─► Auditor (LLM) ─► Vague-claim flags + sharpening suggestions



## Tech stack

- **Streamlit** — UI
- **pypdf** — PDF text extraction
- **Google Gemini (gemini-2.5-flash)** — LLM
- **sentence-transformers** — local embeddings
- **NumPy** — cosine similarity, vector ops

## Setup

\`\`\`bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
\`\`\`

Copy `.env.example` to `.env` and add your [Gemini API key](https://aistudio.google.com).

## Roadmap

- [x] Stage 1 — Project setup, Gemini API integration
- [x] Stage 2 — PDF extraction + text chunking
- [x] Stage 3 — Embeddings + retrieval
- [x] Stage 4 — JD Decomposer
- [x] Stage 5 — Verifier (the heart of the project)
- [x] Stage 6 — Auditor (vague claims)
- [x] Stage 7 — Streamlit UI + deployment

## Author

Saumyadip · [GitHub](https://github.com/Saumyadip92)
