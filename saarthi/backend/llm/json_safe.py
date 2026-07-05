"""
JSON-safe calling — the 5-layer pipeline that turns flaky LLM output into a
validated pydantic object, or a structured degraded result (never a crash).

Layers
------
1. Prompt for strict JSON (response_format=json_object set by the gateway; the
   caller's prompt must contain the word "json", the schema, and an example).
2. Direct json.loads.
3. Cleanup parse — strip ``` fences, extract the first balanced {...} block.
4. Deterministic repair with the `json-repair` library.
5. Judge repair — hand the raw broken text to the json_judge role and parse that.

After any layer parses, validate against the expected pydantic model. If
validation fails, run ONE more json_judge pass with the validation error
included. If everything fails, return a degraded result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from llm.routes import call_llm

T = TypeVar("T", bound=BaseModel)


@dataclass
class JsonSafeResult:
    obj: Optional[BaseModel]
    status: str           # "ok" | "repaired" | "degraded"
    model_used: str       # "provider:model" of the producer
    judge_used: str       # "provider:model" of the json judge, or ""
    error: Optional[str] = None
    raw: str = ""


# ---------------------------------------------------------------------------
# pure-text parsing helpers (no LLM)
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    t = text.strip()
    # remove ```json ... ``` or ``` ... ``` fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring (handles strings/escapes)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _try_parse(text: str) -> Optional[dict]:
    """Layers 2-4: direct -> cleanup -> json-repair. Returns dict or None."""
    # Layer 2: direct
    try:
        v = json.loads(text)
        if isinstance(v, dict):
            return v
    except Exception:  # noqa: BLE001
        pass
    # Layer 3: cleanup (strip fences, take first balanced object)
    cleaned = _strip_fences(text)
    block = _first_balanced_object(cleaned) or cleaned
    try:
        v = json.loads(block)
        if isinstance(v, dict):
            return v
    except Exception:  # noqa: BLE001
        pass
    # Layer 4: deterministic repair
    try:
        repaired = repair_json(block)
        v = json.loads(repaired)
        if isinstance(v, dict):
            return v
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def call_json(
    role: str,
    messages: List[dict],
    schema: Type[T],
    schema_hint: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
    allow_judge_repair: bool = True,
) -> JsonSafeResult:
    """Call an LLM for JSON and return a validated `schema` instance (or degraded).

    `schema_hint` is a short human description of the schema, embedded in the
    judge-repair prompt so the judge knows the target shape.
    """
    status = "ok"
    judge_used = ""
    # ---- Layer 1: the producing call (strict json mode) -------------------
    try:
        res = call_llm(role, messages, want_json=True,
                       temperature=temperature, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001
        return JsonSafeResult(obj=None, status="degraded",
                              model_used="", judge_used="",
                              error=f"producer failed: {e}")
    model_used = f"{res.provider}:{res.model}"
    raw = res.content

    # ---- Layers 2-4: deterministic parsing --------------------------------
    parsed = _try_parse(raw)

    # ---- Layer 5: judge repair if still unparsed --------------------------
    if parsed is None and allow_judge_repair:
        status = "repaired"
        judged, judge_used = _judge_repair(raw, schema_hint)
        parsed = judged

    if parsed is None:
        return JsonSafeResult(obj=None, status="degraded", model_used=model_used,
                              judge_used=judge_used, error="unparseable", raw=raw)

    # ---- validate against pydantic ----------------------------------------
    try:
        obj = schema.model_validate(parsed)
        return JsonSafeResult(obj=obj, status=status, model_used=model_used,
                              judge_used=judge_used, raw=raw)
    except ValidationError as ve:
        # one more judge pass with the validation error included
        if allow_judge_repair:
            status = "repaired"
            judged, judge_used = _judge_repair(
                raw, schema_hint,
                validation_error=str(ve)[:1500],
                prior=json.dumps(parsed)[:2000],
            )
            if judged is not None:
                try:
                    obj = schema.model_validate(judged)
                    return JsonSafeResult(obj=obj, status="repaired",
                                          model_used=model_used,
                                          judge_used=judge_used, raw=raw)
                except ValidationError as ve2:
                    return JsonSafeResult(obj=None, status="degraded",
                                          model_used=model_used,
                                          judge_used=judge_used,
                                          error=f"validation failed: {ve2}",
                                          raw=raw)
        return JsonSafeResult(obj=None, status="degraded", model_used=model_used,
                              judge_used=judge_used,
                              error=f"validation failed: {ve}", raw=raw)


def _judge_repair(raw: str, schema_hint: str,
                  validation_error: str = "",
                  prior: str = "") -> tuple:
    """Layer 5: ask the json_judge role to convert text into valid JSON."""
    extra = ""
    if validation_error:
        extra = (f"\nThe previous attempt was:\n{prior}\n"
                 f"It FAILED schema validation with:\n{validation_error}\n"
                 f"Fix it to satisfy the schema.")
    prompt = (
        "You are a strict JSON repair tool. Convert the following text into a "
        "single valid JSON object that exactly matches this schema. "
        "Output JSON only — no prose, no code fences.\n\n"
        f"SCHEMA:\n{schema_hint}\n\n"
        f"TEXT TO CONVERT:\n{raw[:6000]}\n{extra}"
    )
    try:
        res = call_llm("json_judge",
                       [{"role": "user", "content": prompt}],
                       want_json=True, temperature=0.0, max_tokens=900)
        parsed = _try_parse(res.content)
        return parsed, f"{res.provider}:{res.model}"
    except Exception:  # noqa: BLE001
        return None, ""
