"""
Drive the live SAARTHI app in a real browser and capture prototype screenshots.

Walks the actual user journey — upload a loan book, confirm the LLM column
mapping, run the pipeline, open the dashboard, then drill into one loan's
reasoning — and saves a PNG at each step for the submission deck.
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("SAARTHI_URL", "https://koushikdeb.duckdns.org")
CSV = os.environ.get("SAARTHI_CSV",
                     "/home/Debz/Hackathon/IDBI_Hackathon/saarthi/data/msme_demo.csv")
OUT = os.environ.get("SAARTHI_SHOTS",
                     "/home/Debz/Hackathon/IDBI_Hackathon/deck_figs/shots")
os.makedirs(OUT, exist_ok=True)


def shot(page, name, full=False):
    p = f"{OUT}/{name}.png"
    page.screenshot(path=p, full_page=full)
    print(f"  saved {name}.png", flush=True)
    return p


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--no-sandbox"])
        page = b.new_page(viewport={"width": 1440, "height": 900},
                          device_scale_factor=2, ignore_https_errors=True)
        page.set_default_timeout(60_000)

        # ---- 1. landing -------------------------------------------------
        print("1. landing", flush=True)
        page.goto(URL, wait_until="networkidle")
        time.sleep(2)
        shot(page, "01_landing")

        # ---- 2. upload --------------------------------------------------
        print("2. upload", flush=True)
        page.set_input_files("input[type=file]", CSV)
        page.wait_for_selector("text=/Column profile|Continue to mapping/i",
                               timeout=120_000)
        time.sleep(2)
        shot(page, "02_profile")

        # ---- 3. mapping -------------------------------------------------
        print("3. mapping", flush=True)
        page.click("text=/Continue to mapping/i")
        page.wait_for_selector("text=/target|Run analysis|protected/i", timeout=120_000)
        time.sleep(8)                      # let the LLM mapper settle
        shot(page, "03_mapping")

        # ---- 4. run -----------------------------------------------------
        print("4. run", flush=True)
        for sel in ["button:has-text('Run')", "text=/Run analysis/i",
                    "button:has-text('analysis')"]:
            try:
                page.click(sel, timeout=8_000)
                break
            except Exception:
                continue
        try:
            page.wait_for_selector("text=/Building your early-warning|Reading dataset|Training/i",
                                   timeout=60_000)
            time.sleep(3)
            shot(page, "04_processing")
        except Exception:
            print("   (processing view passed too quickly)", flush=True)

        # ---- 5. dashboard ----------------------------------------------
        print("5. dashboard (waiting for pipeline)", flush=True)
        page.wait_for_url("**/dashboard/**", timeout=600_000)
        page.wait_for_selector("text=/AUC|Risk|Portfolio|High/i", timeout=300_000)
        time.sleep(10)
        shot(page, "05_dashboard")
        shot(page, "05_dashboard_full", full=True)

        # ---- 6. loan detail --------------------------------------------
        print("6. loan detail", flush=True)
        try:
            page.click("table tbody tr", timeout=30_000)
            page.wait_for_url("**/loan/**", timeout=60_000)
            page.wait_for_selector("text=/reason|recourse|explanation|Verified/i",
                                   timeout=180_000)
            time.sleep(12)                 # LLM explanation + judge
            shot(page, "06_loan_detail")
            shot(page, "06_loan_detail_full", full=True)
        except Exception as e:
            print(f"   loan detail skipped: {type(e).__name__}", flush=True)

        # ---- 7. how it works -------------------------------------------
        try:
            page.goto(f"{URL}/how-it-works", wait_until="networkidle")
            time.sleep(3)
            shot(page, "07_how_it_works")
        except Exception:
            pass

        b.close()
    print("CAPTURE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
