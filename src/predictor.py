import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


def predict_leaf_disease(model, image: Image.Image, class_names: list[str]) -> dict:
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "class_name": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
    }
