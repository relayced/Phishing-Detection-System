import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline


def load_split(path: Path):
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression phishing classifier")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_out", default="artifacts/nlp_tfidf_lr.joblib")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train, y_train = load_split(data_dir / "train.csv")
    X_val, y_val = load_split(data_dir / "val.csv")

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    print("Validation classification report")
    print(classification_report(y_val, val_preds, digits=4))
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
