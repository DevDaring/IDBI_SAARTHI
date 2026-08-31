"""
End-to-end acceptance test for SAARTHI (no Flask server needed).

Runs the full pipeline on the synthetic MSME + German-style datasets and checks
the Section 14 acceptance criteria:
  * mapper finds the target; model trains; AUC printed; PD per loan
  * a high-PD loan has reason codes + recommended action + fairness flag
  * faithful plain-English explanation produced (faithfulness judge verdict)
  * fairness audit produces a disparity number across a protected attribute
  * malformed-JSON recovery via the 5-layer pipeline
  * provider-failure fallback (disable DeepSeek -> Mistral)
Run:  python scripts/smoke_test.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "backend")
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, BACKEND)

import config  # noqa: E402

config.SETTINGS.eager_explain_top_k = 6        # keep the test fast
config.SETTINGS.explain_workers = 6

from jobs import Job  # noqa: E402
from llm.gateway import GATEWAY  # noqa: E402
from pipeline.ingest import profile_columns, read_table  # noqa: E402
from pipeline.mapper import heuristic_map  # noqa: E402
from pipeline.orchestrator import build_loan_result, run_pipeline  # noqa: E402
from schemas import LoanResult, PortfolioResult  # noqa: E402

PASS, FAIL = "✓", "✗"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


def run_dataset(path, label):
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    df = read_table(path)
    hm = heuristic_map(profile_columns(df), df)
    check("mapper detected a target", bool(hm.target), f"target={hm.target}")
    check("mapper detected protected attrs", len(hm.protected) > 0, f"{hm.protected}")

    job = Job(job_id=f"smoke_{label}")
    t0 = time.time()
    run_pipeline(job, path, hm.mapping, hm.target, hm.protected, {"consensus": False})
    dt = time.time() - t0
    check("pipeline completed without error", job.error is None, job.error or "")
    if job.error:
        return

    p = PortfolioResult.model_validate(job.portfolio)
    print(f"  model={p.model.type} AUC={p.model.auc} PR-AUC={p.model.pr_auc} "
          f"ECE={p.model.ece} n={p.model.n_loans}  ({dt:.1f}s)")
    check("AUC computed", p.model.auc > 0.5, f"AUC={p.model.auc}")
    check("risk distribution non-empty",
          (p.risk_distribution.high + p.risk_distribution.medium +
           p.risk_distribution.low) == p.model.n_loans)
    check("fairness audit produced a disparity number",
          any(abs(s.dp_diff) > 0 or abs(s.eo_diff) > 0 for s in p.fairness_summary),
          ", ".join(f"{s.attribute}:dp={s.dp_diff},eo={s.eo_diff},{s.flag}"
                    for s in p.fairness_summary))
    check("top-risk loans listed", len(p.top_risk_loans) > 0)

    # high-PD loan must have reason codes + action + fairness + faithful prose
    top_id = p.top_risk_loans[0].loan_id
    lr_dict = job.loans.get(top_id) or build_loan_result(job, job.artifacts["id_to_idx"][top_id])
    lr = LoanResult.model_validate(lr_dict)
    check("high-PD loan has reason codes", len(lr.reason_codes) > 0,
          f"{len(lr.reason_codes)} codes, top={lr.reason_codes[0].code if lr.reason_codes else '-'}")
    check("high-PD loan has plain-English explanation", len(lr.explanation) > 30)
    check("high-PD loan has a recommended action", len(lr.recommended_action.action) > 5,
          f"pd {lr.pd:.2f} -> {lr.recommended_action.expected_pd_after:.2f}")
    check("high-PD loan has a fairness flag", lr.fairness.flag in ("pass", "review"))
    check("explanation quality recorded",
          lr.explanation_quality.json_status in ("ok", "repaired", "degraded"),
          f"faithful={lr.explanation_quality.faithful} "
          f"status={lr.explanation_quality.json_status} "
          f"model={lr.explanation_quality.model_used} judge={lr.explanation_quality.judge}")
    print(f"\n  Sample explanation ({top_id}):\n    {lr.explanation}")
    print(f"  Action: {lr.recommended_action.action} "
          f"(PD {lr.pd:.0%} -> {lr.recommended_action.expected_pd_after:.0%})")

    # faithfulness rate across the eager set
    faithful = [LoanResult.model_validate(v).explanation_quality.faithful
                for v in job.loans.values()]
    if faithful:
        rate = sum(faithful) / len(faithful)
        check("faithfulness rate >= 0.8 on explained loans", rate >= 0.8,
              f"{rate:.0%} of {len(faithful)}")

    # lazy build of a non-eager loan
    all_ids = list(job.artifacts["id_to_idx"].keys())
    lazy_id = next((i for i in all_ids if i not in job.loans), None)
    if lazy_id:
        t = time.time()
        lr2 = LoanResult.model_validate(build_loan_result(job, job.artifacts["id_to_idx"][lazy_id]))
        check("lazy per-loan build works", lr2.loan_id == lazy_id,
              f"{lazy_id} in {time.time()-t:.1f}s")


def test_json_layers():
    print("\n=== Malformed-JSON 5-layer recovery ===")
    from llm.json_safe import _try_parse
    cases = [
        '```json\n{"a": 1, "b": 2,}\n```',                 # fences + trailing comma
        'Here is the result: {"a": 1, "b": [1,2,3]} thanks',  # prose wrapper
        "{'a': 1, 'b': 'two'}",                              # single quotes
        '{"a": 1, "b": 2',                                   # missing brace
    ]
    for c in cases:
        parsed = _try_parse(c)
        check(f"recovered: {c[:32]!r}", parsed is not None, str(parsed))


def test_provider_fallback():
    print("\n=== Provider-failure fallback (disable DeepSeek -> Mistral) ===")
    from llm.routes import call_llm
    ds = GATEWAY.providers["deepseek"]
    saved = ds.enabled
    ds.enabled = False
    try:
        r = call_llm("explainer", [{"role": "user",
                     "content": 'return json {"ok": true}'}],
                     want_json=True, max_tokens=50)
        check("fell back to a non-DeepSeek provider", r.provider != "deepseek",
              f"used {r.provider}:{r.model}")
    except Exception as e:
        check("fell back to a non-DeepSeek provider", False, str(e))
    finally:
        ds.enabled = saved


if __name__ == "__main__":
    print("SAARTHI smoke test — checking providers...")
    GATEWAY.health_check()
    print("  providers up:", [n for n, h in GATEWAY.health.items() if h.get("ok")])

    run_dataset(os.path.join(DATA, "msme_demo.csv"), "MSME")
    run_dataset(os.path.join(DATA, "credit_applicants.csv"), "German-style")
    test_json_layers()
    test_provider_fallback()

    print("\n" + "=" * 50)
    n_pass = sum(1 for r in results if r)
    print(f"RESULT: {n_pass}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
