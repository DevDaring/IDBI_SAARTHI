"""
Multi-model judge panel — a headline trust feature.

1. JSON-repair judge — wired into llm/json_safe.py (json_judge role).
2. Faithfulness judge (the differentiator) — receives the loan's ACTUAL SHAP
   drivers and the explainer's reason codes + prose, and verifies that every
   claimed driver is SHAP-supported and the direction matches. If a driver is
   invented or a sign is flipped, it returns faithful=false; the orchestrator
   regenerates the explanation once.
3. Consensus judge (optional toggle) — two different model families each write an
   explanation, a third judge picks/merges the clearer, more faithful one. Used
   for high-risk loans only.

Judges always use a DIFFERENT model family than the producer (DeepSeek), so the
chains lead with Gemini / OpenRouter.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from llm.json_safe import call_json
from llm.routes import call_llm
from schemas import ConsensusVerdict, FaithfulnessVerdict, ReasonCodeItem

_FAITH_HINT = (
    '{"faithful": true|false, "unsupported_claims": ["..."], '
    '"sign_flips": ["..."], "notes": "one line"}'
)


def faithfulness_check(drivers: List[dict], reason_codes: List[ReasonCodeItem],
                       explanation: str) -> Dict:
    """Verify the explanation against the SHAP evidence. Returns a dict with the
    verdict + the judge model used."""
    driver_view = [{
        "feature": d["feature"], "direction": d["direction"],
        "shap": d["shap"], "value": str(d.get("value", "")),
    } for d in drivers]
    rc_view = [{"feature": r.feature, "code": r.code, "direction": r.direction,
                "evidence": r.evidence} for r in reason_codes]

    # deterministic ground-truth check on the STRUCTURED reason codes first
    det = _deterministic_faithfulness(drivers, reason_codes)

    sys = ("You audit credit-risk explanations for HALLUCINATION only. You are "
           "given the model's ground-truth SHAP drivers and an explanation. "
           "Flag ONLY two things: (1) a NEW risk factor named in the prose that is "
           "absent from the drivers, or (2) a stated risk DIRECTION that is "
           "OPPOSITE to a driver's SHAP sign. Plain-English interpretation of a "
           "driver is ALWAYS fine — e.g. reading a low credit score as 'past "
           "repayment problems', or a low DSCR as 'tight cash flow', is faithful, "
           "not a new factor. Summarising or omitting drivers is fine. Be lenient "
           "on wording, strict on invented factors and flipped signs. JSON only (json).")
    user = (
        "GROUND-TRUTH SHAP DRIVERS (shap>0 => increases default risk; "
        "shap<0 => decreases it):\n"
        f"{json.dumps(driver_view, indent=2)}\n\n"
        "EXPLANATION REASON CODES (already checked structurally — focus on the prose):\n"
        f"{json.dumps(rc_view, indent=2)}\n\n"
        "EXPLANATION PROSE:\n"
        f"\"{explanation}\"\n\n"
        "Decide:\n"
        "- unsupported_claims: list any NEW named risk factor in the prose that is "
        "not one of the drivers above (interpretations of an existing driver do NOT "
        "count).\n"
        "- sign_flips: list any driver the prose describes in the direction "
        "OPPOSITE to its SHAP sign.\n"
        "Set faithful=true if BOTH lists are empty; otherwise false.\n\n"
        "Example faithful=true: drivers=[dscr shap>0], prose='weak debt-service "
        "coverage raises default risk'. Example faithful=false: prose='strong "
        "collateral fully secures the loan' when no collateral driver exists.\n\n"
        f"Respond as JSON: {_FAITH_HINT}"
    )
    res = call_json("faithfulness_judge",
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
                    FaithfulnessVerdict, _FAITH_HINT,
                    temperature=0.0, max_tokens=400)
    if res.obj is None:
        # judge unavailable: fall back to the deterministic structural check
        return {"faithful": det["faithful"], "unsupported_claims": det["unsupported"],
                "sign_flips": det["sign_flips"],
                "notes": "deterministic check (judge unavailable)",
                "judge": res.model_used or "deterministic",
                "structural_ok": det["faithful"]}
    v: FaithfulnessVerdict = res.obj  # type: ignore
    # ignore "vibes-based" verdicts: a false flag with no concrete claim cited
    # is treated as faithful (the judge must point to an actual violation).
    claims = [c for c in v.unsupported_claims if c and c.strip()]
    flips = [f for f in v.sign_flips if f and f.strip()]
    llm_faithful = bool(v.faithful) or (not claims and not flips)
    # combine with the deterministic structural check on the reason codes
    faithful = llm_faithful and det["faithful"]
    return {"faithful": faithful,
            "unsupported_claims": claims,
            "sign_flips": list(set(flips) | set(det["sign_flips"])),
            "notes": v.notes,
            "judge": res.model_used or res.judge_used or "faithfulness_judge",
            "structural_ok": det["faithful"]}


def _deterministic_faithfulness(drivers: List[dict],
                                reason_codes: List[ReasonCodeItem]) -> dict:
    by_feat = {d["feature"]: d for d in drivers}
    unsupported, flips = [], []
    for r in reason_codes:
        d = by_feat.get(r.feature)
        if d is None:
            unsupported.append(r.feature)
        elif d["direction"] != r.direction:
            flips.append(r.feature)
    return {"faithful": not unsupported and not flips,
            "unsupported": unsupported, "sign_flips": flips}


# ---------------------------------------------------------------------------
# Consensus judge (optional, high-risk loans)
# ---------------------------------------------------------------------------
_CONS_HINT = ('{"chosen": "a"|"b"|"merged", "explanation": "final 2-4 sentences", '
              '"reason": "why"}')


def consensus_explanation(drivers: List[dict], pd_value: float,
                          explanation_a: str, explanation_b: str) -> Optional[Dict]:
    """A third-family judge picks/merges the clearer, more faithful explanation."""
    driver_view = [{"feature": d["feature"], "direction": d["direction"],
                    "shap": d["shap"]} for d in drivers]
    sys = ("You are an adjudicator choosing the better credit-risk explanation. "
           "Prefer the one that is faithful to the SHAP drivers, concrete, and "
           "useful to a credit officer. Output JSON only (json).")
    user = (
        f"SHAP drivers: {json.dumps(driver_view)}\nPD={pd_value:.2f}\n\n"
        f"Explanation A:\n\"{explanation_a}\"\n\n"
        f"Explanation B:\n\"{explanation_b}\"\n\n"
        "Pick the better one (chosen='a' or 'b'), or write a merged version "
        f"(chosen='merged'). Respond as JSON: {_CONS_HINT}")
    res = call_json("consensus_judge",
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
                    ConsensusVerdict, _CONS_HINT, temperature=0.1, max_tokens=400)
    if res.obj is None:
        return None
    v: ConsensusVerdict = res.obj  # type: ignore
    return {"chosen": v.chosen, "explanation": v.explanation, "reason": v.reason,
            "judge": res.model_used}
