"""
Column mapping: arbitrary uploaded columns -> the fixed canonical schema.

Primary path: the `mapper` LLM role, given column names/dtypes/null%/samples,
returns the canonical mapping JSON (validated via json_safe). A deterministic
keyword + binary-target heuristic both (a) seeds the prompt and (b) is the
fallback if every provider fails. The user always confirms/edits in the UI.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

import pandas as pd

from llm.json_safe import call_json
from pipeline.canonical import CANONICAL, HINTS, PROTECTED_FIELDS
from schemas import MappingResult


# ---------------------------------------------------------------------------
# Deterministic heuristic (fallback + prompt seed)
# ---------------------------------------------------------------------------
def _looks_binary(profile_col: dict, df: Optional[pd.DataFrame]) -> bool:
    if df is not None and profile_col["name"] in df.columns:
        vals = df[profile_col["name"]].dropna().unique()
        if 1 <= len(vals) <= 2:
            return True
        if len(vals) <= 6:
            lowered = {str(v).strip().lower() for v in vals}
            binary_words = {"0", "1", "yes", "no", "y", "n", "true", "false",
                            "default", "paid", "chargeoff", "charged off",
                            "p i f", "pif", "clp", "good", "bad", "1.0", "0.0"}
            if lowered & binary_words:
                return True
    return profile_col.get("n_unique", 99) == 2


def _norm(s: str) -> str:
    """Lowercase + strip all non-alphanumerics, so 'YearsInBusiness',
    'years_in_business' and 'Years In Business' all collapse to the same token."""
    return re.sub(r"[^0-9a-z]", "", s.lower())


def heuristic_map(columns: List[dict], df: Optional[pd.DataFrame] = None) -> MappingResult:
    names = [c["name"] for c in columns]
    lower = {c["name"]: _norm(c["name"]) for c in columns}
    mapping = {k: None for k in CANONICAL}
    confidence = {}
    used = set()

    for canon, keys in HINTS.items():
        if canon == "target":
            continue  # target gets dedicated binary-aware detection below
        best, best_score = None, 0.0
        for c in names:
            if c in used:
                continue
            cl = lower[c]
            score = 0.0
            for kw_raw in keys:
                kw = _norm(kw_raw)
                if not kw:
                    continue
                if cl == kw:
                    score = max(score, 1.0)
                elif len(kw) < 3:
                    # too short for fuzzy matching (avoid 'y' matching 'years...')
                    continue
                elif cl.startswith(kw) or cl.endswith(kw):
                    score = max(score, 0.85)
                elif kw in cl:
                    score = max(score, 0.7)
            if score > best_score:
                best, best_score = c, score
        if best and best_score >= 0.7:
            mapping[canon] = best
            confidence[canon] = round(best_score, 2)
            used.add(best)

    # target: it MUST be (roughly) binary. Prefer a binary column whose name
    # hints at an outcome; otherwise the binary column with the best keyword
    # score. Exclude columns already mapped to a feature / id / protected attr.
    def _target_kw_score(name: str) -> float:
        cl = _norm(name)
        s = 0.0
        for kw_raw in HINTS["target"]:
            kw = _norm(kw_raw)
            if not kw:
                continue
            if cl == kw:
                s = max(s, 1.0)
            elif len(kw) < 3:
                continue
            elif cl.startswith(kw) or cl.endswith(kw):
                s = max(s, 0.85)
            elif kw in cl:
                s = max(s, 0.7)
        return s

    name_index = {c["name"]: i for i, c in enumerate(columns)}
    binary_cols = [c["name"] for c in columns
                   if _looks_binary(c, df) and c["name"] not in used]
    target = None
    if binary_cols:
        # rank by (keyword score, later position) — the outcome is often last
        target = max(binary_cols, key=lambda c: (_target_kw_score(c), name_index[c]))
    if target is None:
        # no clean binary column; fall back to best keyword match overall
        kw_ranked = sorted(((c["name"], _target_kw_score(c["name"])) for c in columns),
                           key=lambda t: -t[1])
        target = kw_ranked[0][0] if kw_ranked and kw_ranked[0][1] >= 0.7 else None
    mapping["target"] = target
    if target:
        used.add(target)
        confidence["target"] = round(max(0.6, _target_kw_score(target)), 2)

    protected = [mapping[p] and p for p in PROTECTED_FIELDS if mapping.get(p)]
    protected = [p for p in protected if p]
    return MappingResult(mapping=mapping, target=target, protected=protected,
                         confidence=confidence,
                         notes={"_method": "heuristic fallback"})


# ---------------------------------------------------------------------------
# LLM mapper
# ---------------------------------------------------------------------------
_SCHEMA_HINT = (
    '{"mapping": {<canonical_field>: <source_column or null>, ...}, '
    '"target": <source_column or null>, "protected": [<source_column>, ...], '
    '"confidence": {<canonical_field>: 0.0-1.0}, '
    '"notes": {<canonical_field>: "short reason"}}'
)


def _build_prompt(columns: List[dict], seed: MappingResult) -> List[dict]:
    canon_lines = "\n".join(f"- {k}: {meaning} [{role}]"
                            for k, (meaning, role) in CANONICAL.items())
    col_lines = []
    for c in columns:
        col_lines.append(
            f'- "{c["name"]}" (dtype={c["dtype"]}, null%={c["null_pct"]}, '
            f'unique={c.get("n_unique","?")}, samples={c["sample"]})')
    cols_block = "\n".join(col_lines)
    example = {
        "mapping": {"loan_amount": "DisbursementGross", "term_months": "Term",
                    "target": "MIS_Status", "sector": "NAICS", "region": "State",
                    "credit_score": None},
        "target": "MIS_Status",
        "protected": ["State"],
        "confidence": {"loan_amount": 0.95, "target": 0.9, "region": 0.8},
        "notes": {"target": "MIS_Status holds CHGOFF/P I F default labels"},
    }
    user = (
        "You map columns of an uploaded MSME credit dataset onto a FIXED canonical "
        "loan schema. Return strict JSON only.\n\n"
        "CANONICAL FIELDS (map each to at most one source column, or null):\n"
        f"{canon_lines}\n\n"
        "RULES:\n"
        "- `target` is the binary default / charge-off label and is REQUIRED; pick the "
        "column most likely to be a 0/1 or default/paid outcome.\n"
        "- `protected` = region, gender, community columns. These are audit-only and "
        "must be listed in `protected` even though they also appear in `mapping`.\n"
        "- Only use real source column names exactly as given; never invent columns.\n"
        "- Put a 0-1 confidence per mapped field and a one-line note for important ones.\n\n"
        f"UPLOADED COLUMNS:\n{cols_block}\n\n"
        f"A heuristic pre-guess (improve on it): {json.dumps(seed.mapping)}\n\n"
        f"Respond as JSON in EXACTLY this shape:\n{_SCHEMA_HINT}\n\n"
        f"Example output:\n{json.dumps(example)}"
    )
    return [
        {"role": "system", "content": "You are a precise data-mapping assistant. "
                                      "Output only valid JSON. The word json is required."},
        {"role": "user", "content": user},
    ]


def map_columns(columns: List[dict], df: Optional[pd.DataFrame] = None) -> dict:
    """Run the LLM mapper with deterministic seed + fallback. Returns a dict with
    `mapping/target/protected/confidence/notes` plus `_source` (llm|heuristic)."""
    seed = heuristic_map(columns, df)
    messages = _build_prompt(columns, seed)
    result = call_json("mapper", messages, MappingResult, _SCHEMA_HINT,
                       temperature=0.1, max_tokens=1200)
    valid_names = {c["name"] for c in columns}

    if result.obj is None:
        out = seed.model_dump()
        out["_source"] = "heuristic"
        out["_json_status"] = result.status
        return _sanitise(out, valid_names)

    obj: MappingResult = result.obj  # type: ignore
    # merge: keep LLM mapping but backfill anything it missed from the heuristic
    merged = dict(seed.mapping)
    for k, v in obj.mapping.items():
        if k in CANONICAL:
            merged[k] = v if (v in valid_names or v is None) else seed.mapping.get(k)
    target = obj.target if obj.target in valid_names else seed.target
    merged["target"] = target
    protected = [p for p in obj.protected if p in valid_names]
    if not protected:
        protected = seed.protected
    out = {
        "mapping": merged,
        "target": target,
        "protected": protected,
        "confidence": {**seed.confidence, **obj.confidence},
        "notes": obj.notes,
        "_source": "llm",
        "_model": result.model_used,
        "_json_status": result.status,
    }
    return _sanitise(out, valid_names)


def _sanitise(out: dict, valid_names: set) -> dict:
    """Ensure every canonical key exists and values are valid column names/None."""
    mapping = out.get("mapping", {})
    for k in CANONICAL:
        v = mapping.get(k)
        mapping[k] = v if (v in valid_names) else None
    out["mapping"] = mapping
    out["protected"] = [p for p in out.get("protected", []) if p in valid_names]
    out["target"] = out.get("target") if out.get("target") in valid_names else None
    return out
