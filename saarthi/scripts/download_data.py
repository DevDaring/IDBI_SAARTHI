"""
SAARTHI dataset downloader — pulls the project's training datasets into a
"Dataset" folder that lives OUTSIDE the code repository.

Datasets:
  Kaggle community datasets : SBA, Berka, Lending Club, Kiva
  Kaggle competitions       : Home Credit Default Risk, GiveMeSomeCredit
  UCI (no Kaggle token)     : German Credit (id=144), Taiwan Default (id=350)

Usage:
  python scripts/download_data.py

Notes / quirks handled:
  * The `kaggle` library authenticates AT IMPORT TIME and only reads the
    UPPERCASE env vars KAGGLE_USERNAME / KAGGLE_KEY. This .env uses mixed-case
    names (Kaggle_username / Kaggle_key), so we read them case-insensitively and
    export the UPPERCASE vars BEFORE importing kaggle.
  * Competitions return HTTP 403 until you accept their rules in the browser;
    those are caught and skipped with a clear message.
  * Idempotent, rate-limit aware (429 backoff), per-dataset isolation, and a
    final summary table.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths: repo root + the EXTERNAL Dataset folder
# ---------------------------------------------------------------------------
SCRIPT = Path(__file__).resolve()


def find_repo_root() -> Path:
    """Walk up from this script to the hackathon project root.

    The project root is the directory that contains both the code app folder and
    the `Codes/` secrets folder (i.e. .../IDBI_Hackathon). Falls back to two
    levels up (saarthi/scripts -> saarthi -> project root).
    """
    for parent in SCRIPT.parents:
        if (parent / "Codes").is_dir() or (parent / ".git").is_dir() and (parent / "saarthi").is_dir():
            # prefer the dir that holds Codes/ (the hackathon root)
            if (parent / "Codes").is_dir():
                return parent
    return SCRIPT.parents[2] if len(SCRIPT.parents) > 2 else SCRIPT.parent


REPO_ROOT = find_repo_root()

# load .env (Codes/.env holds the keys; also try copies next to the app)
for cand in [REPO_ROOT / "Codes" / ".env", REPO_ROOT / ".env",
             SCRIPT.parents[1] / "backend" / ".env"]:
    if cand.exists():
        load_dotenv(cand)


def resolve_dataset_dir() -> Path:
    override = os.getenv("DATASET_DIR")
    if override and override.strip():
        return Path(override.strip()).expanduser().resolve()
    # default: a "Dataset" folder inside the project root (IDBI_Hackathon/Dataset).
    # (If an older sibling Dataset/ exists next to the repo, prefer that so a
    # re-run skips already-downloaded data instead of re-fetching it.)
    inside = (REPO_ROOT / "Dataset").resolve()
    sibling = (REPO_ROOT.parent / "Dataset").resolve()
    if not inside.exists() and sibling.exists():
        return sibling
    return inside


DATASET_DIR = resolve_dataset_dir()


# ---------------------------------------------------------------------------
# dependency bootstrap
# ---------------------------------------------------------------------------
def ensure(pkg: str, import_name: str | None = None):
    name = import_name or pkg
    try:
        __import__(name)
        return True
    except Exception:
        print(f"  installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg],
                       check=False)
        try:
            __import__(name)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# credentials -> UPPERCASE env BEFORE importing kaggle
# ---------------------------------------------------------------------------
def _ci_env(*names: str) -> str | None:
    """Read an env var case-insensitively across the given candidate names."""
    lowered = {k.lower(): v for k, v in os.environ.items()}
    for n in names:
        if n in os.environ and os.environ[n].strip():
            return os.environ[n].strip()
        if n.lower() in lowered and lowered[n.lower()].strip():
            return lowered[n.lower()].strip()
    return None


def setup_kaggle_creds() -> bool:
    user = _ci_env("Kaggle_username", "KAGGLE_USERNAME", "kaggle_username")
    key = _ci_env("Kaggle_key", "KAGGLE_KEY", "kaggle_key")
    if not user or not key:
        print("  ! Kaggle username/key not found in .env — Kaggle datasets will be skipped")
        return False
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key
    # never print the key
    print(f"  Kaggle user: {user}  (key: ***{key[-4:]})")
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
RESULTS: list[dict] = []
MAX_RETRIES = 3
BACKOFFS = [5, 15, 45]


def has_extracted_files(folder: Path) -> bool:
    if not folder.exists():
        return False
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".zip" and not p.name.startswith("."):
            return True
    return False


def unzip_all(folder: Path):
    for z in folder.glob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder)
            z.unlink()  # delete leftover zip
        except Exception as e:  # noqa: BLE001
            print(f"    ! could not unzip {z.name}: {e}")
    # also unzip any nested single-level zips produced by competitions
    for z in folder.rglob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            z.unlink()
        except Exception:  # noqa: BLE001
            pass


def main_csv_rows(folder: Path) -> int | None:
    """Fast newline-count of the largest CSV in the folder (minus header)."""
    csvs = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_size if p.exists() else 0,
                  reverse=True)
    if not csvs:
        return None
    try:
        with open(csvs[0], "rb") as f:
            count = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
        return max(0, count - 1)
    except Exception:  # noqa: BLE001
        return None


def record(name: str, status: str, detail: str = "", folder: Path | None = None):
    rows = main_csv_rows(folder) if folder else None
    RESULTS.append({"dataset": name, "status": status, "detail": detail, "rows": rows})
    tag = {"OK": "✓", "SKIPPED": "-", "FAILED": "✗"}.get(status, "?")
    print(f"  [{tag}] {name}: {status}" + (f" ({detail})" if detail else ""))


def _is_rate_limited(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "too many requests" in s or "rate limit" in s


def _is_forbidden(err: Exception) -> bool:
    s = str(err).lower()
    return "403" in s or "forbidden" in s or "not accepted" in s or "rules" in s


# ---------------------------------------------------------------------------
# Kaggle community datasets
# ---------------------------------------------------------------------------
def download_kaggle_dataset(api, name: str, ref: str, subdir: str):
    folder = DATASET_DIR / subdir
    if has_extracted_files(folder):
        record(name, "SKIPPED", "already present", folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RETRIES + 1):
        try:
            api.dataset_download_files(ref, path=str(folder), unzip=True, quiet=False)
            unzip_all(folder)
            if has_extracted_files(folder):
                record(name, "OK", ref, folder)
            else:
                record(name, "FAILED", "no files after download", folder)
            return
        except Exception as e:  # noqa: BLE001
            if _is_rate_limited(e) and attempt < MAX_RETRIES:
                wait = BACKOFFS[min(attempt, len(BACKOFFS) - 1)]
                print(f"    rate limited; retrying in {wait}s ...")
                time.sleep(wait)
                continue
            record(name, "FAILED", str(e)[:90], folder)
            return


# ---------------------------------------------------------------------------
# Kaggle competitions (need rules acceptance)
# ---------------------------------------------------------------------------
def download_kaggle_competition(api, name: str, comp: str, subdir: str):
    folder = DATASET_DIR / subdir
    if has_extracted_files(folder):
        record(name, "SKIPPED", "already present", folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/competitions/{comp}"
    for attempt in range(MAX_RETRIES + 1):
        try:
            api.competition_download_files(comp, path=str(folder), quiet=False)
            unzip_all(folder)
            if has_extracted_files(folder):
                record(name, "OK", comp, folder)
            else:
                record(name, "FAILED", "no files after download", folder)
            return
        except Exception as e:  # noqa: BLE001
            if _is_forbidden(e):
                print(f"    403 forbidden — accept the rules in your browser, then re-run:")
                print(f"      {url}/rules")
                record(name, "FAILED", "accept rules in browser then re-run", folder)
                return
            if _is_rate_limited(e) and attempt < MAX_RETRIES:
                wait = BACKOFFS[min(attempt, len(BACKOFFS) - 1)]
                print(f"    rate limited; retrying in {wait}s ...")
                time.sleep(wait)
                continue
            record(name, "FAILED", str(e)[:90], folder)
            return


# ---------------------------------------------------------------------------
# UCI datasets via ucimlrepo
# ---------------------------------------------------------------------------
def download_uci(name: str, uci_id: int, subdir: str, csv_name: str):
    folder = DATASET_DIR / subdir
    out = folder / csv_name
    if out.exists() and out.stat().st_size > 0:
        record(name, "SKIPPED", "already present", folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=uci_id)
        X = ds.data.features
        y = ds.data.targets
        import pandas as pd
        df = pd.concat([X, y], axis=1)
        df.to_csv(out, index=False)
        record(name, "OK", f"uci id={uci_id}", folder)
    except Exception as e:  # noqa: BLE001
        record(name, "FAILED", str(e)[:90], folder)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SAARTHI dataset downloader")
    print(f"  repo root   : {REPO_ROOT}")
    print(f"  Dataset dir : {DATASET_DIR}   (outside the code repo)")
    print("=" * 70)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # --- Kaggle ---
    print("\n[1/3] Kaggle community datasets")
    have_kaggle = setup_kaggle_creds() and ensure("kaggle")
    api = None
    if have_kaggle:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except Exception as e:  # noqa: BLE001
            print(f"  ! Kaggle auth failed: {str(e)[:120]}")
            api = None

    if api is not None:
        download_kaggle_dataset(api, "SBA (should-this-loan-be-approved)",
                                "mirbektoktogaraev/should-this-loan-be-approved-or-denied", "sba")
        download_kaggle_dataset(api, "Berka bank", "marceloventura/the-berka-dataset", "berka")
        download_kaggle_dataset(api, "Lending Club", "wordsforthewise/lending-club", "lending_club")
        download_kaggle_dataset(api, "Kiva crowdfunding",
                                "kiva/data-science-for-good-kiva-crowdfunding", "kiva")

        print("\n[2/3] Kaggle competitions (need rules acceptance)")
        download_kaggle_competition(api, "Home Credit Default Risk",
                                    "home-credit-default-risk", "home_credit")
        download_kaggle_competition(api, "GiveMeSomeCredit", "GiveMeSomeCredit", "gmsc")
    else:
        for nm in ["SBA", "Berka bank", "Lending Club", "Kiva crowdfunding",
                   "Home Credit Default Risk", "GiveMeSomeCredit"]:
            record(nm, "FAILED", "kaggle unavailable")

    # --- UCI ---
    print("\n[3/3] UCI datasets (ucimlrepo)")
    if ensure("ucimlrepo") and ensure("pandas"):
        download_uci("German Credit", 144, "german", "german.csv")
        download_uci("Taiwan Default", 350, "taiwan", "taiwan.csv")
    else:
        record("German Credit", "FAILED", "ucimlrepo unavailable")
        record("Taiwan Default", "FAILED", "ucimlrepo unavailable")

    # --- summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'dataset':<34} {'status':<9} {'rows':>10}   detail")
    print("-" * 70)
    for r in RESULTS:
        rows = f"{r['rows']:,}" if r["rows"] is not None else "-"
        print(f"{r['dataset']:<34} {r['status']:<9} {rows:>10}   {r['detail']}")
    n_ok = sum(1 for r in RESULTS if r["status"] == "OK")
    n_skip = sum(1 for r in RESULTS if r["status"] == "SKIPPED")
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAILED")
    print("-" * 70)
    print(f"  {n_ok} downloaded, {n_skip} skipped, {n_fail} failed")
    print(f"  Dataset folder: {DATASET_DIR}")

    print("\n" + "=" * 70)
    print("MANUAL STEP")
    print("=" * 70)
    print("Fannie Mae and Freddie Mac are NOT on Kaggle and cannot be downloaded")
    print("with a Kaggle token. Register and download manually if you need the")
    print("monthly survival panel:")
    print("  Fannie Mae  -> Data Dynamics portal")
    print("  Freddie Mac -> Clarity Data Intelligence")
    print("=" * 70)


if __name__ == "__main__":
    main()
