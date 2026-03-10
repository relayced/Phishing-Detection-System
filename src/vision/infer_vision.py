from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class VisionInferencer:
    def __init__(self, model_path: str = "artifacts/vision_mobilenet.keras"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Vision model not found: {path}")
        self.model = tf.keras.models.load_model(path)

    def predict_score(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        score = self.model.predict(arr, verbose=0)[0][0]
        return float(score)
