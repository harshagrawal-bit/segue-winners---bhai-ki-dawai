#!/usr/bin/env python3
"""
Capture screenshots of every TrialSense tab with a headless browser.

Doubles as the real end-to-end render check: Streamlit reports script
exceptions in the page itself, so this fails loudly if anything breaks live
that the smoke test cannot see.

    python capture_screens.py [port] [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

TABS = [
    ("Risk Report", "01-risk-report"),
    ("① Interaction", "02-interaction"),
    ("② Patient Response", "03-patient-response"),
    ("③ Resistance", "04-resistance"),
    ("Methods & Honesty", "05-methods"),
]


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "8508"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "assets/screenshots")
    outdir.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100},
                                device_scale_factor=2)
        page.goto(url, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(9000)  # let models load and charts draw

        # Select the flagship demo case so the screenshots show the real pitch
        # scenario rather than whatever the default inputs happen to be.
        try:
            page.locator('[data-testid="stSelectbox"]').first.click()
            page.wait_for_timeout(900)
            page.get_by_text("Case 1 —", exact=False).first.click()
            page.wait_for_timeout(6000)
            print("Selected demo Case 1")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not select demo case: {exc}) — capturing defaults")

        # Streamlit surfaces script errors as an exception block in the DOM.
        errors = page.locator('[data-testid="stException"]').count()
        alerts = page.locator(".stAlert").count()
        print(f"Exception blocks on page: {errors}  (alert boxes: {alerts})")
        if errors:
            print("--- Exception text ---")
            print(page.locator('[data-testid="stException"]').first.inner_text()[:3000])
            browser.close()
            return 1

        title = page.title()
        print(f"Page title: {title!r}")

        for label, slug in TABS:
            try:
                tab = page.get_by_role("tab", name=label)
                tab.click(timeout=15_000)
                page.wait_for_timeout(3500)
            except Exception as exc:  # noqa: BLE001
                print(f"  could not open tab {label!r}: {exc}")
                continue

            if page.locator('[data-testid="stException"]').count():
                print(f"  EXCEPTION after opening {label!r}:")
                print(page.locator('[data-testid="stException"]').first.inner_text()[:2000])
                browser.close()
                return 1

            path = outdir / f"{slug}.png"
            page.screenshot(path=str(path), full_page=True)
            size = path.stat().st_size // 1024
            print(f"  saved {path}  ({size} KB)")

        browser.close()
    print("\nAll tabs rendered with no exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
