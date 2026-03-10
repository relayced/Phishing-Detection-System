import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.text_cleaning import clean_email_text


TEXT_CANDIDATES = ["text", "email", "body", "message", "content", "email text"]
LABEL_CANDIDATES = ["label", "target", "class", "is_phishing", "email type"]


def find_column(columns: list[str], candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError(f"Could not find a column from candidates: {candidates}")


def normalize_label(value) -> int:
    if isinstance(value, str):
        val = value.strip().lower()
        if "phish" in val:
            return 1
        if "legit" in val or "safe" in val or "ham" in val:
            return 0
        if val in {"phishing", "spam", "malicious", "1", "true", "yes"}:
            return 1
        if val in {"legitimate", "ham", "safe", "0", "false", "no"}:
            return 0
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split email phishing dataset")
    parser.add_argument("--input", required=True, help="CSV path")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    text_col = find_column(df.columns.tolist(), TEXT_CANDIDATES)
    label_col = find_column(df.columns.tolist(), LABEL_CANDIDATES)

    df = df[[text_col, label_col]].dropna().copy()
    df["text"] = df[text_col].astype(str).map(clean_email_text)
    df["label"] = df[label_col].map(normalize_label).astype(int)
    df = df[["text", "label"]]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Prepared dataset:")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()
