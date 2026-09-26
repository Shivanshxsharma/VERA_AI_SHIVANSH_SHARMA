import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Gemini Multi-Key Configuration (Tier 1 & Tier 2 with auto-rotation) ──
GEMINI_API_KEY1: str = os.getenv("GEMINI_API_KEY1", os.getenv("GEMINI_API_KEY_1", os.getenv("GEMINI_API_KEY", "")))
GEMINI_API_KEY2: str = os.getenv("GEMINI_API_KEY2", os.getenv("GEMINI_API_KEY_2", ""))
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SECONDS: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "6.0"))

# ── 3rd Backup Provider: Mistral AI (Tier 3) ──
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
MISTRAL_TIMEOUT_SECONDS: float = float(os.getenv("MISTRAL_TIMEOUT_SECONDS", "8.0"))

TICK_ACTION_CAP: int = 20
TICK_TIMEOUT_SECONDS: float = 25.0  # Hard tick deadline leaving 5s margin for the 30s challenge limit          # leave 2s headroom inside the 30s budget
PAYLOAD_MAX_BYTES: int = 500_000            # 500 KB
SUPPRESSION_TTL_SECONDS: int = 3600         # 1 hour default

# ── team metadata (returned by GET /v1/metadata) ──
TEAM_NAME = "Vera_Shivansh"
TEAM_MEMBERS = ["Shivansh Sharma"]
MODEL_NAME = "gemini-2.5-flash (with auto-rotating backup keys & mistral failover)"
APPROACH = (
    "Rule-based decision pipeline (suppression, expiry, ranking) with "
    "multi-tier prose composition: Primary Gemini 2.5 Flash with auto-rotating backup keys "
    "(auto-failover on HTTP 429 rate limits), 3rd backup Mistral AI failover, "
    "constrained strictly to a deterministic fact sheet, with a pure rule-based fallback composer. "
    "Multi-turn conversation state machine built with LangGraph."
)
CONTACT_EMAIL = "shivansh@example.com"
VERSION = "0.2.0"
SUBMITTED_AT = datetime.now(timezone.utc).isoformat()

# ── boot timestamp for uptime ──
BOOT_TIME = datetime.now(timezone.utc)
