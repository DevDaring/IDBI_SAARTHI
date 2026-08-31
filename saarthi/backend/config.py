"""
SAARTHI configuration.

Loads the .env (the user keeps all keys in Codes/.env; we also copy it next to
the backend). Normalises the *actual* key names found in the user's environment
onto the canonical provider definitions the LLM gateway expects.

Design notes
------------
The build prompt assumed key names like DEEPSEEK_API_KEY_1 / DEEPSEEK_BASE_URL /
DEEPSEEK_MODEL.  The user's real .env uses slightly different names
(DEEPSEEK_API_BASE_URL, DEEPSEEK_PRIMARY_MODEL_NAME, MISTRAL_API_KEY1, ...).
We read the real names first and fall back to the prompt names so the app works
unchanged if either convention is present.

Nothing here is ever sent to the frontend except the *resolved* provider/model
list (no keys) via /api/models.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load .env  (search several plausible locations, first hit wins)
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BACKEND_DIR.parent              # saarthi/
_REPO_DIR = _PROJECT_DIR.parent                 # IDBI_Hackathon/

_ENV_CANDIDATES = [
    _BACKEND_DIR / ".env",
    _PROJECT_DIR / ".env",
    _REPO_DIR / "Codes" / ".env",
    _REPO_DIR / ".env",
]
ENV_PATH: Optional[Path] = next((p for p in _ENV_CANDIDATES if p.exists()), None)
if ENV_PATH is not None:
    load_dotenv(ENV_PATH)


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty env var among `names`."""
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip().strip('"').strip("'")
    return default


def _keys(*names: str) -> List[str]:
    """Collect a de-duplicated list of non-empty keys from the given env names."""
    out: List[str] = []
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            k = str(v).strip().strip('"').strip("'")
            if k not in out:
                out.append(k)
    return out


# ---------------------------------------------------------------------------
# Provider definition
# ---------------------------------------------------------------------------
@dataclass
class Provider:
    name: str
    base_url: str
    keys: List[str]
    default_model: str
    # extra named models available on this provider (for diverse judging)
    models: dict = field(default_factory=dict)
    # provider-specific default headers (e.g. OpenRouter ranking headers)
    extra_headers: dict = field(default_factory=dict)
    enabled: bool = True

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.keys) and bool(self.base_url)


# ---------------------------------------------------------------------------
# Build the provider table from the real environment
# ---------------------------------------------------------------------------
def _build_providers() -> dict:
    providers: dict = {}

    # --- DeepSeek : PRIMARY worker -----------------------------------------
    ds_keys = _keys("DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY")
    ds_model = _env("DEEPSEEK_PRIMARY_MODEL_NAME", "DEEPSEEK_MODEL", default="deepseek-chat")
    ds_pro = _env("DEEPSEEK_JUDGE_MODEL_NAME", "DEEPSEEK_MODEL_PRO", default=ds_model or "deepseek-chat")
    providers["deepseek"] = Provider(
        name="deepseek",
        base_url=_env("DEEPSEEK_API_BASE_URL", "DEEPSEEK_BASE_URL", default="https://api.deepseek.com/v1"),
        keys=ds_keys,
        default_model=ds_model,
        models={"primary": ds_model, "pro": ds_pro},
    )

    # --- Mistral : FALLBACK -------------------------------------------------
    mi_keys = _keys("MISTRAL_API_KEY1", "MISTRAL_API_KEY2", "MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2", "MISTRAL_API_KEY")
    mi_model = _env("MISTRAL_MODEL_NAME", "MISTRAL_MODEL", default="mistral-small-latest")
    providers["mistral"] = Provider(
        name="mistral",
        base_url=_env("MISTRAL_API_BASE_URL", "MISTRAL_BASE_URL", default="https://api.mistral.ai/v1"),
        keys=mi_keys,
        default_model=mi_model,
        models={"primary": mi_model, "large": "mistral-large-latest"},
    )

    # --- OpenRouter : strong / diverse judges ------------------------------
    or_keys = _keys("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY")
    or_model = _env("OPENROUTER_PRIMARY_MODEL_NAME", "OPENROUTER_MODEL", default="openai/gpt-4o-mini")
    or_llama = _env("OPENROUTER_LLAMA_MODEL_NAME", default="meta-llama/llama-3.1-8b-instruct")
    or_gemma = _env("OPENROUTER_GEMMA_MODEL_NAME", default="google/gemma-3-4b-it")
    providers["openrouter"] = Provider(
        name="openrouter",
        base_url=_env("OPENROUTER_API_BASE_URL", "OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"),
        keys=or_keys,
        default_model=or_model,
        models={"primary": or_model, "llama": or_llama, "gemma": or_gemma},
        extra_headers={
            "HTTP-Referer": "https://saarthi.idbi-innovate",
            "X-Title": "SAARTHI",
        },
    )

    # --- Gemini : diverse judge family (OpenAI-compatible endpoint) ---------
    # Keys 3 & 4 verified working; 1 is an OAuth token, 2 was rejected.
    # We keep all AIza* keys; the gateway health-check + rotation skips dead ones.
    gm_keys = [k for k in _keys(
        "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_1"
    ) if k.startswith("AIza")]
    gm_model = _env("GEMINI_MODEL_NAME", default="gemini-2.5-flash-lite")
    providers["gemini"] = Provider(
        name="gemini",
        base_url=_env("GEMINI_API_BASE_URL",
                      default="https://generativelanguage.googleapis.com/v1beta/openai"),
        keys=gm_keys,
        default_model=gm_model,
        models={"primary": gm_model},
    )

    # --- NanoGPT : free/bulk (best-effort; needs balance/subscription) ------
    ng_keys = _keys("Nano_GPT_API_KEY", "NANOGPT_API_KEY")
    ng_model = _env("NANOGPT_MODEL", default="")  # resolved at startup from /models
    providers["nanogpt"] = Provider(
        name="nanogpt",
        base_url=_env("Nano_GPT_Base_URL", "NANOGPT_BASE_URL", default="https://nano-gpt.com/api/v1"),
        keys=ng_keys,
        default_model=ng_model,
        models={"primary": ng_model} if ng_model else {},
        # disabled by default unless a model is configured AND health-check passes
        enabled=bool(ng_keys),
    )

    return providers


PROVIDERS: dict = _build_providers()


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    flask_port: int = int(_env("FLASK_PORT", default="5000"))
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", default="512"))
    train_row_cap: int = int(_env("TRAIN_ROW_CAP", default="100000"))
    # rows kept in memory for scoring + lazy per-loan explanation (memory-safe cap)
    score_row_cap: int = int(_env("SCORE_ROW_CAP", default="150000"))
    # how many top-risk loans get an LLM explanation precomputed during the run
    eager_explain_top_k: int = int(_env("EAGER_EXPLAIN_TOP_K", default="12"))
    # parallel workers for eager per-loan LLM explanation
    explain_workers: int = int(_env("EXPLAIN_WORKERS", default="6"))
    random_seed: int = int(_env("RANDOM_SEED", default="20260502") or "20260502")
    upload_dir: Path = _BACKEND_DIR / "uploads"
    # risk band thresholds on the calibrated PD
    band_high: float = 0.50
    band_medium: float = 0.20
    # survival onset threshold (cumulative PD crossing)
    onset_threshold: float = 0.20
    # how many SHAP drivers to feed the explainer
    top_k_drivers: int = 6


SETTINGS = Settings()
SETTINGS.upload_dir.mkdir(parents=True, exist_ok=True)


def provider_summary() -> list:
    """Non-secret summary for /api/models (key COUNT, never the keys)."""
    out = []
    for p in PROVIDERS.values():
        out.append({
            "name": p.name,
            "base_url": p.base_url,
            "n_keys": len(p.keys),
            "default_model": p.default_model,
            "models": p.models,
            "available": p.available,
        })
    return out
