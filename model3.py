import tensorflow as tf
import numpy as np
import os

# Load the trained Keras model
model = tf.keras.models.load_model("models/mnist_fp32.keras")

# Representative dataset for INT8 calibration
(x_train, _), _ = tf.keras.datasets.mnist.load_data()

# Normalize to 0–1
x_train = x_train.astype("float32") / 255.0

# Add grayscale channel
x_train = x_train[..., np.newaxis]


def representative_dataset():
    for i in range(100):
        image = x_train[i:i+1]
        yield [image]


# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force INT8 input and output
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_int8_model = converter.convert()

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

# Save INT8 model
with open("models/mnist_int8.tflite", "wb") as f:
    f.write(tflite_int8_model)

print("INT8 LiteRT model saved successfully.")