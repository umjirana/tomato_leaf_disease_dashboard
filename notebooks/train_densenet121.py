import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = BASE_DIR / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


TRAIN_DIR = Path("/Users/umjirana/tomatoleaf/tomato/train")
VAL_DIR = Path("/Users/umjirana/tomatoleaf/tomato/val")
MODEL_PATH = BASE_DIR / "models" / "densenet121_tomato.keras"
CLASS_NAMES_PATH = BASE_DIR / "data" / "class_names.txt"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32


def load_dataset(path: Path) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    return dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int) -> tf.keras.Model:
    conv_base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    conv_base.trainable = False

    model = Sequential(
        [
            conv_base,
            BatchNormalization(),
            Dense(256, activation="relu"),
            Dropout(0.35),
            BatchNormalization(),
            Dense(120, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    train_data = load_dataset(TRAIN_DIR)
    val_data = load_dataset(VAL_DIR)
    class_names = train_data.class_names if hasattr(train_data, "class_names") else None

    if class_names is None:
        class_names = sorted(path.name for path in TRAIN_DIR.iterdir() if path.is_dir())

    CLASS_NAMES_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    model = build_model(num_classes=len(class_names))
    model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
