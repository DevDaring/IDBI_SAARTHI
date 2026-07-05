"""
LLM Gateway — the single choke-point for every LLM call in SAARTHI.

All providers (DeepSeek, Mistral, OpenRouter, Gemini, NanoGPT) are
OpenAI-compatible, so we use one `openai.OpenAI(base_url=, api_key=)` client per
(provider, key) and swap as needed.

Responsibilities
----------------
* Per-provider round-robin key rotation.
* Per-call retry with exponential backoff, then key rotation, then (at the
  routing layer) provider fallback.
* response_format=json_object when JSON is requested, with automatic retry
  WITHOUT it if a provider rejects the parameter.
* Startup health check that pings each provider, records availability and the
  real model ids it exposes (self-correcting against renamed model ids).
* A rolling, secret-free call trace so the UI can show a transparent model trace.

Nothing here parses JSON — that is json_safe.py's job. This module only returns
raw text content (or raises after exhausting a provider's keys).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from openai import OpenAI

from config import PROVIDERS, Provider


# ---------------------------------------------------------------------------
# Call result + trace
# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    content: str
    provider: str
    model: str
    key_index: int
    latency_ms: int
    finish_reason: str
    role: Optional[str] = None
    ok: bool = True
    error: Optional[str] = None


class CallTrace:
    """Bounded, thread-safe ring buffer of recent calls (no secrets)."""

    def __init__(self, maxlen: int = 200):
        self._buf: Deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}

    def add(self, r: LLMResult):
        with self._lock:
            self._buf.appendleft({
                "role": r.role,
                "provider": r.provider,
                "model": r.model,
                "key_index": r.key_index,
                "latency_ms": r.latency_ms,
                "finish_reason": r.finish_reason,
                "ok": r.ok,
                "error": (r.error or "")[:200] if r.error else None,
            })
            key = f"{r.provider}:{r.model}"
            self._counts[key] = self._counts.get(key, 0) + 1

    def recent(self, n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._buf)[:n]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)


TRACE = CallTrace()


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------
class Gateway:
    def __init__(self, providers: Dict[str, Provider]):
        self.providers = providers
        # client cache keyed by (provider, key_index)
        self._clients: Dict[Tuple[str, int], OpenAI] = {}
        # round-robin pointer per provider
        self._rr: Dict[str, int] = {p: 0 for p in providers}
        self._lock = threading.Lock()
        # populated by health_check(): provider -> set/list of available model ids
        self.available_models: Dict[str, List[str]] = {}
        self.health: Dict[str, dict] = {}

    # -- client management --------------------------------------------------
    def _client(self, provider: str, key_index: int) -> OpenAI:
        key = (provider, key_index)
        with self._lock:
            c = self._clients.get(key)
            if c is None:
                p = self.providers[provider]
                c = OpenAI(
                    base_url=p.base_url,
                    api_key=p.keys[key_index],
                    timeout=60.0,
                    max_retries=0,           # we manage retries ourselves
                    default_headers=p.extra_headers or None,
                )
                self._clients[key] = c
            return c

    def _next_key_index(self, provider: str) -> int:
        with self._lock:
            n = len(self.providers[provider].keys)
            if n == 0:
                return 0
            i = self._rr[provider] % n
            self._rr[provider] = (i + 1) % n
            return i

    # -- a single completion against ONE provider (rotates that provider's
    #    keys with backoff) ------------------------------------------------
    def complete(
        self,
        provider: str,
        model: str,
        messages: List[dict],
        want_json: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 900,
        role: Optional[str] = None,
        retries_per_key: int = 2,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        p = self.providers.get(provider)
        if p is None or not p.available:
            raise RuntimeError(f"provider '{provider}' unavailable")

        n_keys = max(1, len(p.keys))
        attempts = n_keys * retries_per_key
        last_err: Optional[Exception] = None
        backoff = 1.0

        for attempt in range(attempts):
            key_index = self._next_key_index(provider)
            client = self._client(provider, key_index)
            if timeout is not None:
                client = client.with_options(timeout=timeout)
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            if want_json:
                kwargs["response_format"] = {"type": "json_object"}
            t0 = time.time()
            try:
                resp = client.chat.completions.create(**kwargs)
                latency = int((time.time() - t0) * 1000)
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                finish = getattr(choice, "finish_reason", "stop") or "stop"
                if not content:
                    raise RuntimeError("empty content")
                res = LLMResult(content=content, provider=provider, model=model,
                                key_index=key_index, latency_ms=latency,
                                finish_reason=finish, role=role, ok=True)
                TRACE.add(res)
                return res
            except Exception as e:  # noqa: BLE001 - we want to catch everything
                latency = int((time.time() - t0) * 1000)
                msg = str(e)
                last_err = e
                # If the provider rejected response_format, retry once w/o it.
                if want_json and ("response_format" in msg or "json_object" in msg
                                  or "Unsupported" in msg or "not supported" in msg):
                    try:
                        resp = client.chat.completions.create(
                            model=model, messages=messages,
                            temperature=temperature, max_tokens=max_tokens)
                        content = (resp.choices[0].message.content or "").strip()
                        if content:
                            res = LLMResult(content=content, provider=provider,
                                            model=model, key_index=key_index,
                                            latency_ms=int((time.time() - t0) * 1000),
                                            finish_reason="stop", role=role, ok=True)
                            TRACE.add(res)
                            return res
                    except Exception as e2:  # noqa: BLE001
                        last_err = e2
                        msg = str(e2)
                TRACE.add(LLMResult(content="", provider=provider, model=model,
                                    key_index=key_index, latency_ms=latency,
                                    finish_reason="error", role=role, ok=False,
                                    error=msg))
                # backoff before next attempt (unless it's the last)
                if attempt < attempts - 1:
                    time.sleep(min(backoff, 8.0))
                    backoff *= 2
        raise RuntimeError(f"{provider} exhausted: {last_err}")

    # -- startup health check ----------------------------------------------
    def health_check(self) -> Dict[str, dict]:
        """Ping every provider's /models, verify configured ids, self-correct.

        Returns a per-provider health dict and (importantly) DISABLES providers
        that don't respond so the routing layer skips them cleanly.
        """
        for name, p in self.providers.items():
            if not p.keys or not p.base_url:
                self.health[name] = {"ok": False, "reason": "no keys/url"}
                p.enabled = False
                continue
            ids: List[str] = []
            ok = False
            reason = ""
            # try /models on the first key
            try:
                client = self._client(name, 0)
                listing = client.models.list()
                ids = [m.id for m in getattr(listing, "data", []) if getattr(m, "id", None)]
                ok = True
            except Exception as e:  # noqa: BLE001
                reason = f"models.list failed: {e}"
                # some providers gate /models; fall back to a tiny chat probe
                try:
                    self.complete(name, p.default_model or "gpt-4o-mini",
                                  [{"role": "user", "content": "ok"}],
                                  want_json=False, max_tokens=3, retries_per_key=1)
                    ok = True
                    reason = "models.list gated; chat probe ok"
                except Exception as e2:  # noqa: BLE001
                    ok = False
                    reason = f"probe failed: {e2}"

            self.available_models[name] = ids
            # verify / self-correct the configured default model id
            corrected = None
            if ok and ids and p.default_model and p.default_model not in ids:
                # try to find a sensible replacement
                cand = _closest_model(p.default_model, ids)
                if cand:
                    corrected = cand
                    p.default_model = cand
                    if "primary" in p.models:
                        p.models["primary"] = cand
            # self-correct any named model ids that no longer exist (e.g. the
            # 'pro' judge model after a provider rename)
            if ok and ids:
                for slot, mid in list(p.models.items()):
                    if mid and mid not in ids:
                        repl = _closest_model(mid, ids)
                        # prefer a 'pro'/'large'/'reasoner' variant for the pro slot
                        if slot == "pro":
                            pro = next((i for i in ids if any(
                                h in i.lower() for h in ("pro", "large", "reason"))), None)
                            repl = pro or repl
                        if repl:
                            p.models[slot] = repl
            # NanoGPT needs a model; if none configured, pick a cheap-looking one
            if name == "nanogpt" and not p.default_model and ids:
                pick = _pick_nanogpt_model(ids)
                if pick:
                    p.default_model = pick
                    p.models["primary"] = pick

            # NanoGPT (and any metered provider) must pass a real chat probe —
            # /models can succeed while the account has no balance (HTTP 402).
            if ok and name == "nanogpt":
                try:
                    self.complete(name, p.default_model,
                                  [{"role": "user", "content": "ok"}],
                                  want_json=False, max_tokens=2, retries_per_key=1)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    reason = f"chat probe failed (likely no balance): {str(e)[:80]}"

            p.enabled = ok and (bool(p.default_model) or name != "nanogpt")
            self.health[name] = {
                "ok": p.enabled,
                "n_models": len(ids),
                "default_model": p.default_model,
                "corrected_model": corrected,
                "reason": reason,
            }
        return self.health


def _closest_model(want: str, ids: List[str]) -> Optional[str]:
    """Pick the best available replacement for a renamed/missing model id."""
    want_l = want.lower()
    # exact suffix/prefix family match first
    fam = want_l.split("/")[-1].split("-")[0]  # 'deepseek', 'mistral', 'gpt', ...
    cands = [i for i in ids if fam and fam in i.lower()]
    if cands:
        # prefer ones containing 'chat'/'flash'/'small'/'mini' (cheaper/faster)
        for hint in ("chat", "flash", "small", "mini", "instruct"):
            for c in cands:
                if hint in c.lower():
                    return c
        return cands[0]
    return ids[0] if ids else None


def _pick_nanogpt_model(ids: List[str]) -> Optional[str]:
    prefs = ("free", "mini", "small", "flash", "8b", "qwen", "llama")
    low = [(i, i.lower()) for i in ids]
    for pref in prefs:
        for orig, l in low:
            if pref in l and "thinking" not in l:
                return orig
    return ids[0] if ids else None


# Singleton gateway
GATEWAY = Gateway(PROVIDERS)
