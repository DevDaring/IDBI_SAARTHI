"""
In-process job manager: a ThreadPoolExecutor + an in-memory job store with
status polling. No Redis required (that is an optional scale-up).

Each job tracks: stage, percent, message, done, error, and (when finished) the
assembled PortfolioResult + per-loan LoanResults.
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


# ordered pipeline stages, used to compute a baseline percent
STAGES = [
    "ingest", "map", "features", "train", "survival",
    "explain", "judge", "recourse", "fairness", "assemble",
]
STAGE_LABELS = {
    "ingest": "Reading dataset",
    "map": "Confirming column mapping",
    "features": "Building feature matrix",
    "train": "Training & calibrating model",
    "survival": "Computing 12-month risk curves",
    "explain": "Generating explanations",
    "judge": "Verifying with judge panel",
    "recourse": "Finding recommended actions",
    "fairness": "Auditing fairness",
    "assemble": "Assembling results",
}


@dataclass
class Job:
    job_id: str
    stage: str = "queued"
    percent: float = 0.0
    message: str = "Queued"
    done: bool = False
    error: Optional[str] = None
    portfolio: Optional[dict] = None          # PortfolioResult (dict)
    loans: Dict[str, dict] = field(default_factory=dict)  # loan_id -> LoanResult dict
    # heavy, NON-serialised in-memory context for lazy per-loan explanation
    # (scored frame, model artifacts, run options). Never returned by the API.
    artifacts: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, stage: Optional[str] = None, percent: Optional[float] = None,
               message: Optional[str] = None):
        with self._lock:
            if stage is not None:
                self.stage = stage
            if percent is not None:
                self.percent = max(0.0, min(100.0, percent))
            if message is not None:
                self.message = message

    def status(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "stage": self.stage,
                "stage_label": STAGE_LABELS.get(self.stage, self.stage.title()),
                "percent": round(self.percent, 1),
                "message": self.message,
                "done": self.done,
                "error": self.error,
            }


class JobManager:
    def __init__(self, max_workers: int = 3):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="saarthi-job")

    def create(self, job_id: str) -> Job:
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: Job, fn: Callable[[Job], None]):
        def _run():
            try:
                fn(job)
                job.done = True
                if job.error is None:
                    job.update(stage="assemble", percent=100.0, message="Complete")
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
                job.done = True
                job.update(message=f"Failed: {job.error}")
                traceback.print_exc()
        self._pool.submit(_run)


JOBS = JobManager()
