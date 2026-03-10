import argparse
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def safe_url(url: str) -> bool:
    blocked_tokens = ["127.0.0.1", "localhost", "0.0.0.0", "192.168.", "10.", "172.16."]
    lower = (url or "").lower()
    if not lower.startswith(("http://", "https://")):
        return False
    return not any(token in lower for token in blocked_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture website screenshots from URL list")
    parser.add_argument("--input_csv", required=True, help="CSV with url column")
    parser.add_argument("--output_dir", default="data/raw/screenshots")
    parser.add_argument("--url_col", default="url")
    parser.add_argument("--wait_sec", type=float, default=2.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.url_col not in df.columns:
        raise ValueError(f"Column '{args.url_col}' not found in {args.input_csv}")

    driver = make_driver()
    captured = 0

    try:
        for idx, row in df.iterrows():
            url = str(row[args.url_col]).strip()
            if not safe_url(url):
                continue
            try:
                driver.set_page_load_timeout(15)
                driver.get(url)
                time.sleep(args.wait_sec)
                driver.save_screenshot(str(output_dir / f"site_{idx}.png"))
                captured += 1
            except WebDriverException:
                continue
    finally:
        driver.quit()

    print(f"Captured {captured} screenshots into: {output_dir}")


if __name__ == "__main__":
    main()
