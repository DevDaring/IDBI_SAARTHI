"""
SAARTHI Flask API.

Endpoints (Section 7.2):
  POST /api/upload          multipart CSV -> profile
  POST /api/map             {upload_id} -> LLM column mapping
  POST /api/run             {upload_id, mapping, target, protected, consensus?} -> {job_id}
  GET  /api/status/<job>    pipeline progress
  GET  /api/results/<job>   PortfolioResult
  GET  /api/loan/<job>/<id> LoanResult (lazy-built + cached)
  GET  /api/models          resolved provider/model routes + call trace
  GET  /api/health

All LLM keys stay server-side. Uploads are validated for type + size.
"""
from __future__ import annotations

import os
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import SETTINGS, provider_summary
from jobs import JOBS
from llm.gateway import GATEWAY, TRACE
from llm.routes import routes_summary
from pipeline.ingest import ALLOWED_EXT, profile_file
from pipeline.mapper import map_columns
from pipeline.orchestrator import build_loan_result, run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = SETTINGS.max_upload_mb * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

# in-memory upload registry: upload_id -> {path, filename, profile}
UPLOADS: dict = {}


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "saarthi"})


@app.get("/api/models")
def models():
    return jsonify({
        "routes": routes_summary(),
        "providers": [
            {"name": p["name"], "ok": GATEWAY.health.get(p["name"], {}).get("ok", p["available"]),
             "n_keys": p["n_keys"], "default_model": p["default_model"],
             "models": list(p["models"].values()) if isinstance(p["models"], dict) else [],
             "n_models": GATEWAY.health.get(p["name"], {}).get("n_models", 0)}
            for p in provider_summary()
        ],
        "health": GATEWAY.health,
        "trace": TRACE.recent(40),
        "call_counts": TRACE.counts(),
        "pretrained": _pretrained_info(),
    })


def _pretrained_info() -> dict:
    """Non-secret summary of the offline-trained global credit model."""
    try:
        from pipeline import pretrained
        return pretrained.info()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:120]}


@app.get("/api/pretrained")
def pretrained_endpoint():
    """What the shipped global model is and how it scored on held-out data."""
    return jsonify(_pretrained_info())


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file part (expected field 'file')"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported file type '{ext}'. "
                                 f"Allowed: {sorted(ALLOWED_EXT)}"}), 400
    upload_id = uuid.uuid4().hex[:12]
    safe = secure_filename(f.filename) or f"upload{ext}"
    path = str(SETTINGS.upload_dir / f"{upload_id}_{safe}")
    f.save(path)
    try:
        profile = profile_file(path, f.filename)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not read file: {e}"}), 400
    UPLOADS[upload_id] = {"path": path, "filename": f.filename, "profile": profile}
    return jsonify({"upload_id": upload_id, **profile})


@app.post("/api/map")
def do_map():
    body = request.get_json(force=True, silent=True) or {}
    upload_id = body.get("upload_id")
    rec = UPLOADS.get(upload_id)
    if rec is None:
        return jsonify({"error": "unknown upload_id"}), 404
    # use a small dataframe sample for binary-target detection
    from pipeline.ingest import read_table
    try:
        df = read_table(rec["path"], max_rows=5000)
    except Exception:
        df = None
    result = map_columns(rec["profile"]["columns"], df)
    return jsonify(result)


@app.post("/api/run")
def run():
    body = request.get_json(force=True, silent=True) or {}
    upload_id = body.get("upload_id")
    rec = UPLOADS.get(upload_id)
    if rec is None:
        return jsonify({"error": "unknown upload_id"}), 404
    mapping = body.get("mapping") or {}
    target = body.get("target")
    protected = body.get("protected") or []
    options = {"consensus": bool(body.get("consensus", False))}
    if not target:
        return jsonify({"error": "a target column is required to train"}), 400

    job_id = uuid.uuid4().hex[:12]
    job = JOBS.create(job_id)

    def _task(j):
        run_pipeline(j, rec["path"], mapping, target, protected, options)

    JOBS.submit(job, _task)
    return jsonify({"job_id": job_id})


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    return jsonify(job.status())


@app.get("/api/results/<job_id>")
def results(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    if job.error:
        return jsonify({"error": job.error}), 500
    if not job.done or job.portfolio is None:
        return jsonify({"error": "job not finished", "status": job.status()}), 409
    return jsonify(job.portfolio)


@app.get("/api/loan/<job_id>/<path:loan_id>")
def loan(job_id, loan_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    if job.portfolio is None:
        return jsonify({"error": "job not finished"}), 409
    # cached?
    if loan_id in job.loans:
        return jsonify(job.loans[loan_id])
    # lazy build
    idx = job.artifacts.get("id_to_idx", {}).get(str(loan_id))
    if idx is None:
        return jsonify({"error": f"unknown loan_id '{loan_id}'"}), 404
    try:
        lr = build_loan_result(job, int(idx))
        job.loans[loan_id] = lr
        return jsonify(lr)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not build loan result: {e}"}), 500


@app.get("/api/")
def root():
    return jsonify({"service": "SAARTHI", "endpoints": [
        "/api/health", "/api/models", "/api/upload", "/api/map", "/api/run",
        "/api/status/<job>", "/api/results/<job>", "/api/loan/<job>/<id>"]})


def _startup():
    print("=" * 60)
    print("SAARTHI backend starting — checking LLM providers...")
    health = GATEWAY.health_check()
    for name, h in health.items():
        flag = "OK " if h.get("ok") else "off"
        print(f"  [{flag}] {name:11} model={h.get('default_model')}  "
              f"models={h.get('n_models', '-')}  {h.get('reason', '')[:50]}")
    print("=" * 60)


_startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=SETTINGS.flask_port, debug=False, threaded=True)
