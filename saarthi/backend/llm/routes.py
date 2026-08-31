"""
Role-based fallback chains.

`call_llm(role, messages, want_json)` resolves a *role* to an ordered list of
(provider, model) attempts and walks the chain on any failure (HTTP error, 429,
timeout, empty content). The gateway handles per-provider key rotation + backoff;
this layer handles provider-to-provider fallback.

Diversity rule: a judge MUST use a different model *family* than the producer it
judges. DeepSeek produces explanations, so the faithfulness judge leads with
Gemini / OpenRouter (different families).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from llm.gateway import GATEWAY, LLMResult


# A route entry is (provider_name, model_selector) where model_selector picks
# the concrete model id from that provider's config at call time (so startup
# self-correction is respected).
def _model(provider: str, which: str = "primary") -> Tuple[str, str]:
    return (provider, which)


# role -> ordered list of (provider, model_key)
ROUTES = {
    "mapper": [
        _model("deepseek", "primary"),
        _model("mistral", "primary"),
        _model("openrouter", "primary"),
        _model("gemini", "primary"),
    ],
    "explainer": [
        _model("deepseek", "primary"),
        _model("mistral", "primary"),
        _model("openrouter", "llama"),
        _model("gemini", "primary"),
        _model("nanogpt", "primary"),
    ],
    # JSON repair: lead with a DIFFERENT family than the typical producer
    "json_judge": [
        _model("mistral", "primary"),
        _model("openrouter", "primary"),
        _model("gemini", "primary"),
        _model("deepseek", "pro"),
    ],
    # Anti-hallucination judge: diverse families, strongest first
    "faithfulness_judge": [
        _model("gemini", "primary"),
        _model("openrouter", "primary"),
        _model("mistral", "primary"),
        _model("deepseek", "pro"),
    ],
    # Consensus: two distinct families produce, a third judges/merges
    "consensus_producer_a": [_model("openrouter", "llama"), _model("mistral", "primary")],
    "consensus_producer_b": [_model("openrouter", "gemma"), _model("gemini", "primary")],
    "consensus_judge": [
        _model("openrouter", "primary"),
        _model("gemini", "primary"),
        _model("deepseek", "pro"),
    ],
}


def resolve_chain(role: str) -> List[Tuple[str, str]]:
    """Return concrete (provider, model_id) pairs that are currently available."""
    chain: List[Tuple[str, str]] = []
    for provider, which in ROUTES.get(role, []):
        p = GATEWAY.providers.get(provider)
        if not p or not p.available or not p.enabled:
            continue
        model_id = p.models.get(which) or p.default_model
        if not model_id:
            continue
        chain.append((provider, model_id))
    return chain


# per-role soft timeout (seconds): a slow primary fails over fast to the next
# provider so the UI never hangs. None = use the client default (60s).
ROLE_TIMEOUTS = {
    "mapper": 30.0,
    "explainer": 22.0,
    "json_judge": 20.0,
    "faithfulness_judge": 18.0,
    "consensus_judge": 22.0,
}


def call_llm(
    role: str,
    messages: List[dict],
    want_json: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 900,
    on_attempt: Optional[Callable[[str, str], None]] = None,
    timeout: Optional[float] = None,
) -> LLMResult:
    """Walk the role's fallback chain; return the first successful LLMResult.

    Raises RuntimeError only if every provider in the chain fails.
    """
    chain = resolve_chain(role)
    if not chain:
        raise RuntimeError(f"no available providers for role '{role}'")
    if timeout is None:
        timeout = ROLE_TIMEOUTS.get(role)
    last_err: Optional[Exception] = None
    for provider, model_id in chain:
        if on_attempt:
            on_attempt(provider, model_id)
        try:
            return GATEWAY.complete(
                provider=provider, model=model_id, messages=messages,
                want_json=want_json, temperature=temperature,
                max_tokens=max_tokens, role=role, retries_per_key=1,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001 - fall through to next provider
            last_err = e
            continue
    raise RuntimeError(f"all providers failed for role '{role}': {last_err}")


def routes_summary() -> dict:
    """Non-secret view of the resolved chains for /api/models."""
    out = {}
    for role in ROUTES:
        out[role] = [
            {"provider": prov, "model": model} for prov, model in resolve_chain(role)
        ]
    return out
