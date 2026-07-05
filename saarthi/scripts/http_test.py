"""
Live HTTP end-to-end test against a running SAARTHI backend.

Usage:
  scripts/run_backend.sh        # in one terminal
  python scripts/http_test.py   # in another

Exercises the full API surface the frontend uses: health, models, upload, map,
run (job), status polling, results, and a per-loan drill-down.
"""
import os
import sys
import time

import requests

BASE = os.environ.get("SAARTHI_BASE", "http://localhost:5000/api")
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "msme_demo.csv")
ok = []

# optional Basic Auth + self-signed TLS support (for the hosted HTTPS endpoint)
_user = os.environ.get("SAARTHI_USER")
_pass = os.environ.get("SAARTHI_PASS")
AUTH = (_user, _pass) if _user and _pass else None
VERIFY = os.environ.get("SAARTHI_VERIFY", "0") == "1"  # self-signed -> don't verify

_orig = requests.Session.request


def _patched(self, method, url, **kw):
    kw.setdefault("auth", AUTH)
    kw.setdefault("verify", VERIFY)
    return _orig(self, method, url, **kw)


requests.Session.request = _patched
requests.packages.urllib3.disable_warnings()  # quiet self-signed warnings


def check(name, cond, detail=""):
    ok.append(cond)
    print(f"  [{'✓' if cond else '✗'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    # health
    r = requests.get(f"{BASE}/health", timeout=10).json()
    check("GET /health", r.get("status") == "ok", str(r))

    # models
    m = requests.get(f"{BASE}/models", timeout=20).json()
    up = [p["name"] for p in m.get("providers", []) if p.get("ok")]
    check("GET /models lists providers", len(up) > 0, f"up={up}")
    check("routes resolved", len(m.get("routes", {})) > 0,
          f"{list(m.get('routes', {}).keys())}")

    # upload
    with open(DATA, "rb") as f:
        up = requests.post(f"{BASE}/upload", files={"file": ("msme_demo.csv", f, "text/csv")},
                           timeout=60).json()
    check("POST /upload profiles columns", len(up.get("columns", [])) > 0,
          f"{up.get('n_rows')} rows, {len(up.get('columns', []))} cols")
    upload_id = up["upload_id"]

    # map (LLM)
    mp = requests.post(f"{BASE}/map", json={"upload_id": upload_id}, timeout=90).json()
    check("POST /map returns a target", bool(mp.get("target")),
          f"target={mp.get('target')} protected={mp.get('protected')} src={mp.get('_source')}")

    # run
    rn = requests.post(f"{BASE}/run", json={
        "upload_id": upload_id, "mapping": mp["mapping"], "target": mp["target"],
        "protected": mp["protected"], "consensus": False}, timeout=30).json()
    job_id = rn.get("job_id")
    check("POST /run starts a job", bool(job_id), f"job_id={job_id}")

    # poll status
    t0 = time.time()
    last = ""
    done = False
    while time.time() - t0 < 300:
        st = requests.get(f"{BASE}/status/{job_id}", timeout=20).json()
        msg = f"{st.get('stage')}:{st.get('percent')}% {st.get('message')}"
        if msg != last:
            print(f"      … {msg}")
            last = msg
        if st.get("done"):
            done = True
            check("job finished", st.get("error") is None, st.get("error") or "")
            break
        time.sleep(1.5)
    check("status reached done within 5 min", done)

    # results
    res = requests.get(f"{BASE}/results/{job_id}", timeout=30).json()
    check("GET /results returns portfolio",
          "model" in res and "risk_distribution" in res,
          f"AUC={res.get('model', {}).get('auc')} "
          f"dist={res.get('risk_distribution')}")
    top = res.get("top_risk_loans", [])
    check("portfolio has top-risk loans", len(top) > 0)

    # loan drill-down
    if top:
        lid = top[0]["loan_id"]
        lr = requests.get(f"{BASE}/loan/{job_id}/{lid}", timeout=120).json()
        check("GET /loan returns reason codes + explanation",
              len(lr.get("reason_codes", [])) > 0 and len(lr.get("explanation", "")) > 20,
              f"pd={lr.get('pd')} band={lr.get('risk_band')} "
              f"faithful={lr.get('explanation_quality', {}).get('faithful')}")
        check("loan has recommended action + fairness",
              bool(lr.get("recommended_action", {}).get("action")) and
              lr.get("fairness", {}).get("flag") in ("pass", "review"))
        print(f"\n  Loan {lid}: {lr.get('explanation')}")
        ra = lr.get("recommended_action", {})
        print(f"  Action: {ra.get('action')} "
              f"(PD {lr.get('pd'):.0%} -> {ra.get('expected_pd_after'):.0%})")

    print("\n" + "=" * 50)
    n = sum(1 for x in ok if x)
    print(f"HTTP RESULT: {n}/{len(ok)} checks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Backend not reachable at", BASE, "— start it with scripts/run_backend.sh")
        sys.exit(2)
