import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.fusion.ensemble import weighted_ensemble


def evaluate_scores(df: pd.DataFrame, score_col: str, label_col: str = "label") -> tuple[float, str]:
    probs = df[score_col].astype(float)
    labels = df[label_col].astype(int)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    report = classification_report(labels, preds, digits=4)
    return auc, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unimodal vs multimodal phishing scores")
    parser.add_argument("--input_csv", required=True, help="CSV with columns: label,nlp_score,vision_score")
    parser.add_argument("--nlp_weight", type=float, default=0.55)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_csv))
    needed = {"label", "nlp_score", "vision_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    final_scores = []
    for _, row in df.iterrows():
        result = weighted_ensemble(row["nlp_score"], row["vision_score"], nlp_weight=args.nlp_weight)
        final_scores.append(result.final_score)
    df["final_score"] = final_scores

    for col in ["nlp_score", "vision_score", "final_score"]:
        auc, report = evaluate_scores(df, col)
        print(f"\n=== {col} ===")
        print(f"ROC-AUC: {auc:.4f}")
        print(report)


if __name__ == "__main__":
    main()
