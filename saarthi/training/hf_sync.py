"""
HuggingFace sync helpers.

Two jobs:
  push_sequences  - upload the derived transaction-sequence parquets to a
                    PRIVATE dataset repo so a rented GPU can fetch them
                    without SSH. Private because Amex-derived data is under
                    Kaggle competition terms.
  push_models     - upload trained artifacts (CoLES checkpoint, GBDT bundles,
                    metrics) to a model repo.
  pull            - fetch a repo down onto whatever machine is running.

Token comes from HUGGINGFACE_TOKEN / HF_TOKEN in the environment.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi

USER = os.environ.get("HF_USER", "Debk")
SEQ_REPO = f"{USER}/saarthi-sequences"
MODEL_REPO = f"{USER}/saarthi-default-prediction"
SEQ_DIR = os.environ.get("SAARTHI_SEQ",
                         "/home/Debz/Hackathon/IDBI_Hackathon/Dataset/sequences")
MODEL_DIR = os.environ.get("SAARTHI_MODELS",
                           "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/models")


def _token() -> str:
    t = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not t:
        sys.exit("HUGGINGFACE_TOKEN not set")
    return t


def _api() -> HfApi:
    return HfApi(token=_token())


def push_sequences():
    api = _api()
    api.create_repo(SEQ_REPO, repo_type="dataset", private=True, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(SEQ_DIR)):
        if not f.endswith(".parquet"):
            continue
        p = os.path.join(SEQ_DIR, f)
        api.upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id=SEQ_REPO, repo_type="dataset")
        print(f"  uploaded {f} ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
        n += 1
    print(f"pushed {n} files -> https://huggingface.co/datasets/{SEQ_REPO}", flush=True)


def push_models(include_joblib: bool = True):
    api = _api()
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    exts = (".json", ".pt", ".txt", ".md", ".parquet")
    if include_joblib:
        exts += (".joblib",)
    n = 0
    for f in sorted(os.listdir(MODEL_DIR)):
        if not f.endswith(exts):
            continue
        p = os.path.join(MODEL_DIR, f)
        if os.path.getsize(p) > 4.5e9:
            print(f"  SKIP {f} (>4.5GB)", flush=True)
            continue
        api.upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id=MODEL_REPO, repo_type="model")
        print(f"  uploaded {f} ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
        n += 1
    print(f"pushed {n} files -> https://huggingface.co/{MODEL_REPO}", flush=True)


def pull(repo: str, repo_type: str, dest: str):
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id=repo, repo_type=repo_type, token=_token(),
                          local_dir=dest)
    print(f"pulled {repo} -> {p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push-sequences", "push-models",
                                       "pull-sequences", "pull-models"])
    ap.add_argument("--dest", default=".")
    a = ap.parse_args()
    if a.action == "push-sequences":
        push_sequences()
    elif a.action == "push-models":
        push_models()
    elif a.action == "pull-sequences":
        pull(SEQ_REPO, "dataset", a.dest)
    else:
        pull(MODEL_REPO, "model", a.dest)
