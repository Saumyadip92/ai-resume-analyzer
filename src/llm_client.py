# src/llm_client.py
"""Thin wrapper around the Gemini SDK with retry-with-backoff and a
model-fallback chain for transient cloud errors.

Centralizes client construction, model selection, and API key loading so
that downstream modules don't each re-implement the boilerplate.
"""

import os
import time
import random
from functools import lru_cache
from typing import Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors


# Models we try in order. If gemini-2.5-flash 503s, we fall back to
# gemini-2.5-flash-lite, then gemini-2.0-flash. Each is free-tier eligible.
DEFAULT_MODEL_CHAIN: List[str] = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
]
DEFAULT_MODEL = DEFAULT_MODEL_CHAIN[0]


# How many times to retry a single model before falling back to the next
# in the chain. Each retry waits longer (1s, 2s, 4s) plus jitter.
_MAX_RETRIES_PER_MODEL = 3
_BASE_DELAY_SECONDS = 1.0


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    """Return a singleton Gemini client.

    Checks for the API key in this order:
    1. Streamlit secrets (for Streamlit Cloud deployment)
    2. Environment variable (for local .env via python-dotenv)
    """
    load_dotenv()
    api_key = None

    # Try Streamlit secrets first (works on Streamlit Cloud)
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    # Fall back to environment variable (works locally with .env)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Set it in .env (local) or Streamlit secrets (cloud)."
        )
    return genai.Client(api_key=api_key)


def generate_with_retry(
    contents: Any,
    config: Any,
    models: Optional[List[str]] = None,
) -> Any:
    """Call generate_content with exponential backoff and model fallback.

    Args:
        contents: prompt content (string or structured)
        config: GenerateContentConfig instance
        models: ordered list of models to try. Defaults to DEFAULT_MODEL_CHAIN.

    Returns:
        The genai response object on success.

    Raises:
        The last exception encountered if all models + retries are exhausted.
    """
    client = get_client()
    models = models or DEFAULT_MODEL_CHAIN
    last_exception: Optional[Exception] = None

    for model in models:
        for attempt in range(_MAX_RETRIES_PER_MODEL):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                # 5xx (transient cloud) and 429 (rate-limited) -> backoff & retry.
                # 4xx (other) -> not retriable, bubble up.
                status = getattr(e, "code", None)
                is_retriable = (
                    isinstance(e, genai_errors.ServerError)
                    or status == 429
                )
                if not is_retriable:
                    raise

                last_exception = e
                # Exponential backoff with jitter to avoid thundering-herd retries.
                delay = _BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        # Exhausted retries for this model; loop to next model in chain.

    # All models, all retries exhausted.
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("generate_with_retry exhausted with no exception captured")
