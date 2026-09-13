import tensorflow as tf
import os

# Load the trained Keras model
model = tf.keras.models.load_model("models/mnist_fp32.keras")

# Convert Keras model to LiteRT
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Create models directory if it does not exist
os.makedirs("models", exist_ok=True)

# Save the FP32 LiteRT model
with open("models/mnist_fp32.tflite", "wb") as f:
    f.write(tflite_model)

print("FP32 LiteRT model saved successfully.")