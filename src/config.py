"""Validated runtime settings and constants for Day 7 Agentic RAG."""
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
AGENT_CASES_PATH = DATA_DIR / "agent_cases.json"
RESULTS_DIR = REPO_ROOT / "results"

ALLOWED_FILTER_FIELDS: frozenset[str] = frozenset(
    {"document_type", "document_number", "effective_date", "document_id"}
)
ALLOWED_DOCUMENT_TYPES: frozenset[str] = frozenset({"luat", "nghi_dinh"})

DEFAULT_TOP_K = 3
DEFAULT_BUDGET_BASELINE = 800
DEFAULT_BUDGET_EXPANDED = 2400
DEFAULT_MAX_SEARCHES = 4
DEFAULT_CANDIDATE_K = 20
RERANK_CANDIDATE_K = 10
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

MIN_DEV_ANSWERABLE = 15
MIN_DEV_UNANSWERABLE = 2
MIN_TEST_ANSWERABLE = 5
MIN_TEST_UNANSWERABLE = 2


@dataclass(frozen=True)
class Settings:
    """Strictly validated runtime configuration loaded from .env."""

    database_url: str
    openrouter_api_key: str
    openrouter_base_url: str
    embedding_model: str
    chat_api_key: str
    chat_base_url: str
    chat_model: str
    day6_strategy: str


def load_settings(env_path: Path | None = None) -> Settings:
    """Load and validate required environment settings without silent fallbacks."""
    target = env_path or (REPO_ROOT / ".env")
    if target.exists():
        load_dotenv(target, override=False)

    values = {
        "DATABASE_URL": (os.getenv("DATABASE_URL") or "").strip(),
        "OPENROUTER_API_KEY": (os.getenv("OPENROUTER_API_KEY") or "").strip(),
        "OPENROUTER_BASE_URL": (os.getenv("OPENROUTER_BASE_URL") or "").strip(),
        "EMBEDDING_MODEL": (os.getenv("EMBEDDING_MODEL") or "").strip(),
        "CHAT_API_KEY": (os.getenv("CHAT_API_KEY") or "").strip(),
        "CHAT_BASE_URL": (os.getenv("CHAT_BASE_URL") or "").strip(),
        "CHAT_MODEL": (os.getenv("CHAT_MODEL") or "").strip(),
        "DAY6_STRATEGY": (os.getenv("DAY6_STRATEGY") or "").strip(),
    }
    missing = [key for key, val in values.items() if not val]
    if missing:
        raise RuntimeError(f"Missing required configuration in {target}: {', '.join(missing)}")

    return Settings(
        database_url=values["DATABASE_URL"],
        openrouter_api_key=values["OPENROUTER_API_KEY"],
        openrouter_base_url=values["OPENROUTER_BASE_URL"],
        embedding_model=values["EMBEDDING_MODEL"],
        chat_api_key=values["CHAT_API_KEY"],
        chat_base_url=values["CHAT_BASE_URL"],
        chat_model=values["CHAT_MODEL"],
        day6_strategy=values["DAY6_STRATEGY"],
    )
