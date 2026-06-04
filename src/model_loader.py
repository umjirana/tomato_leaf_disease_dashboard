from pathlib import Path

import tensorflow as tf

from src.config import CLASS_NAMES_PATH, MODEL_PATH


def load_class_names(path: Path = CLASS_NAMES_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"클래스 이름 파일을 찾을 수 없습니다: {path}")

    class_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    class_names = [name for name in class_names if name]

    if not class_names:
        raise ValueError(f"클래스 이름 파일이 비어 있습니다: {path}")

    return class_names


def load_tomato_model(path: Path = MODEL_PATH) -> tf.keras.Model:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            "저장된 모델 파일이 없거나 비어 있습니다. "
            "노트북 학습 마지막에 model.save('../models/densenet121_tomato.keras')를 실행하세요."
        )

    return tf.keras.models.load_model(path)
