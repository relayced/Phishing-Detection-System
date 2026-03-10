from pathlib import Path

import joblib


class TfidfNLPInferencer:
    def __init__(self, model_path: str = "artifacts/nlp_tfidf_lr.joblib"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"NLP model not found: {path}")
        self.model = joblib.load(path)

    def predict_score(self, text: str) -> float:
        score = self.model.predict_proba([text])[0, 1]
        return float(score)
