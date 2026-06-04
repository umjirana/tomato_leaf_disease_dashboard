import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            nested = _find_last_conv_layer(layer)
            if nested is not None:
                return nested

        output_shape = getattr(layer, "output", None)
        if output_shape is not None and len(layer.output.shape) == 4:
            return layer

    return None


def _call_layer(layer, inputs):
    try:
        return layer(inputs, training=False)
    except TypeError:
        return layer(inputs)


def make_gradcam_heatmap(model, image_batch: np.ndarray, last_conv_layer_name: str | None = None) -> np.ndarray:
    base_model = model.layers[0] if model.layers and isinstance(model.layers[0], tf.keras.Model) else model
    last_conv_layer = base_model.get_layer(last_conv_layer_name) if last_conv_layer_name else _find_last_conv_layer(base_model)

    if last_conv_layer is None:
        raise ValueError("Grad-CAM을 만들 수 있는 4D convolution layer를 찾지 못했습니다.")

    feature_model = tf.keras.models.Model(base_model.inputs, [last_conv_layer.output, base_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(image_batch)
        predictions = features
        if base_model is not model:
            for layer in model.layers[1:]:
                predictions = _call_layer(layer, predictions)

        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_value = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (max_value + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(image, heatmap: np.ndarray, alpha: float = 0.42):
    from PIL import Image

    base = image.convert("RGB")
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(base.size, Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0

    red = np.clip(1.8 * heatmap_array, 0, 1)
    green = np.clip(1.8 * (1 - np.abs(heatmap_array - 0.55) * 2), 0, 1)
    blue = np.clip(1.4 * (1 - heatmap_array), 0, 1)
    color = np.stack([red, green, blue], axis=-1)

    base_array = np.asarray(base, dtype=np.float32) / 255.0
    blended = np.clip((1 - alpha) * base_array + alpha * color, 0, 1)
    return Image.fromarray(np.uint8(blended * 255))
