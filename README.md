# Resume Auditor — Evidence-Based JD Matcher

An LLM-powered tool that audits a resume against a job description by checking
for **evidence** of each requirement, not just keyword matches.

Most resume analyzers do keyword/cosine similarity matching, which misses the
forest for the trees: a candidate who built a Flask API serving 10k req/day
clearly knows REST APIs, even if the literal word "REST" never appears in
their resume. This tool reasons about evidence, with citations.

## Live Demo

🚀 **[Try it here](https://saumyadip92-ai-resume-analyzer.streamlit.app)** — upload a resume PDF + paste a JD, get an evidence-based audit in 30 seconds.

## How it works

1. **Upload** a resume PDF + **paste** a job description
2. The JD is decomposed into atomic, testable requirements
3. Each requirement is matched against the resume using semantic retrieval (embeddings + cosine similarity)
4. A verifier LLM judges each match with a verdict (Strong / Partial / Weak / Missing) and cites the exact resume lines as evidence
5. A separate auditor flags vague claims in the resume and suggests concrete rewrites

## Architecture

```
PDF Resume ─► Text extraction ─► Chunking ─► Embeddings ─► Vector store
                                                                │
JD Text ─► JD Decomposer (LLM) ─► Atomic requirements ──┐       │
                                                        ▼       ▼
                              Retrieve top-k resume chunks per requirement
                                                        │
                                                        ▼
                                         Verifier (LLM): verdict + citations
                                                        │
                                                        ▼
                                            Verification Table

Resume ─► Auditor (LLM) ─► Vague-claim flags + sharpening suggestions
```

## Key design decisions

- **Evidence-based, not keyword-based.** Embeddings capture semantic similarity, so "Built Flask API serving 10k req/day" matches "REST API experience" even with zero keyword overlap.
- **Structured LLM outputs.** JSON schemas enforce response shape at the token-generation level — no regex parsing, no string surgery.
- **No vector database.** Resume corpus is 3-30 chunks; in-memory NumPy is faster and simpler than FAISS/Chroma at this scale.
- **No LangChain.** Hand-built pipeline for full transparency and learning value.
- **Retry-with-backoff + model fallback.** Tries gemini-2.5-flash → 2.5-flash-lite → 2.0-flash with exponential backoff on transient errors.

## Tech stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF (with Unicode ligature normalization) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local, free) |
| LLM | Google Gemini 2.5 Flash (free tier) |
| Vector math | NumPy |
| Secrets | python-dotenv (local) / Streamlit secrets (cloud) |

## Setup (local)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your [Gemini API key](https://aistudio.google.com).

```bash
streamlit run app.py
```

## Roadmap

- [x] Stage 1 — Project setup, Gemini API integration
- [x] Stage 2 — PDF extraction + text chunking
- [x] Stage 3 — Embeddings + retrieval
- [x] Stage 4 — JD Decomposer (structured output)
- [x] Stage 5 — Verifier (evidence-based verdicts with citations)
- [x] Stage 6 — Auditor (vague-claim detection + rewrites)
- [x] Stage 7 — Streamlit UI + deployment

## Author

Saumyadip · [GitHub](https://github.com/Saumyadip92)
