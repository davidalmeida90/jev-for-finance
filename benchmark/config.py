"""Keys and the SEC User-Agent for the benchmark scripts.

Values come from environment variables, or from a .env file at the repo root when python-dotenv is
installed (see .env.example). Nothing here is ever printed.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()   # Jev and the Result 3 LLMs
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()       # DeepSeek Flash, direct
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()           # SEC asks for "Name email@domain"


def need(value, name):
    """Stop with a clear message when a call needs a value that is not set."""
    if not value:
        raise SystemExit(f"{name} is not set: copy .env.example to .env and fill it in")
    return value
