import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model() -> tf.keras.Model:
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Rescaling(1.0 / 255),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for phishing screenshot classification")
    parser.add_argument("--data_dir", required=True, help="Directory with class subfolders phishing/legitimate")
    parser.add_argument("--output", default="artifacts/vision_mobilenet.keras")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved vision model to: {out_path}")


if __name__ == "__main__":
    main()
