"""
Application configuration.

Design notes:
- All knobs are env vars so the same code runs locally and on Render.
- We also load a local .env file at import time (via python-dotenv) so
  developers can put GROQ_API_KEY etc. in a .env file and not have to
  export it manually every shell session.
- Defaults are tuned for the SHL evaluator's constraints:
    * 30s per-call timeout -> we cap LLM calls at ~12s each, leaving
      budget for retrieval + guardrail + JSON parse.
    * 8-turn cap total -> we allow at most 2 clarification turns before
      committing to a shortlist, so the agent never times out the user.
"""

import os
from dataclasses import dataclass

# Load .env from the project root if present. On Render this file won't
# exist (env vars come from the dashboard), so we ignore missing-file.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed -- fall back to plain os.getenv. Render
    # deploys set vars in the environment directly, so this still works.
    pass


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    primary_model: str = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT", "12"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    # --- Retrieval ---
    embedding_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "30"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))  # standard RRF constant

    # --- Agent ---
    max_clarifications: int = int(os.getenv("MAX_CLARIFICATIONS", "2"))
    # On turn 2+, if the user has volunteered enough info, commit.
    # This guards against burning the 8-turn budget on clarifications.

    # --- Paths ---
    catalog_path: str = os.getenv("CATALOG_PATH", "data/catalog.json")


settings = Settings()