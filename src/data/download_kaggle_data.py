import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_kaggle_download(dataset_slug: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_path),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for phishing detector")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. owner/name")
    parser.add_argument("--output", default="data/raw", help="Download directory")
    args = parser.parse_args()

    required_vars = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        missing_str = ", ".join(missing)
        raise EnvironmentError(
            f"Missing Kaggle credentials in environment: {missing_str}. "
            "Set them or use ~/.kaggle/kaggle.json"
        )

    run_kaggle_download(args.dataset, args.output)
    print(f"Downloaded dataset '{args.dataset}' to {args.output}")


if __name__ == "__main__":
    main()
