
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Load the existing INT8 model
model_path = Path(__file__).resolve().parent / "models" / "mnist_int8.tflite"
interpreter = Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

# Connect to the Samsung phone's IP Webcam stream
url = "http://10.54.146.15:8080/video"
cap = cv2.VideoCapture(url)

try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open phone camera stream")

    # Allow time to position the handwritten digit.
    # Keep reading frames so we use a fresh frame, not an old buffered one.
    deadline = time.monotonic() + 2.5
    frame = None

    while time.monotonic() < deadline:
        ok, current = cap.read()
        if ok:
            frame = current

    if frame is None:
        raise RuntimeError("Could not receive a camera frame")

    cv2.imwrite("captured.jpg", frame)
finally:
    cap.release()

# Match the working laptop preprocessing
h, w = frame.shape[:2]
roi = frame[int(h * 0.05):int(h * 0.90),
            int(w * 0.25):int(w * 0.75)]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = [c for c in contours if cv2.contourArea(c) > 50]

if not contours:
    raise RuntimeError("No digit detected; check lighting and positioning")

largest = max(contours, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(largest)
digit = thresh[y:y + bh, x:x + bw]

# Center the digit in a square, then fit it into a 28x28 MNIST image
side = max(bw, bh)
square = np.zeros((side, side), dtype=np.uint8)
offset_x = (side - bw) // 2
offset_y = (side - bh) // 2
square[offset_y:offset_y + bh, offset_x:offset_x + bw] = digit

small = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
image28 = np.zeros((28, 28), dtype=np.float32)
image28[4:24, 4:24] = small.astype(np.float32) / 255.0
input_data = image28.reshape(1, 28, 28, 1)

# Quantize input if required by the model
if np.issubdtype(input_details["dtype"], np.integer):
    input_data = np.round(
        input_data / input_scale + input_zero_point
    )
    limits = np.iinfo(input_details["dtype"])
    input_data = np.clip(input_data, limits.min, limits.max)

input_data = input_data.astype(input_details["dtype"])

interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

# Dequantize output when the model returns integer values
if np.issubdtype(output_details["dtype"], np.integer):
    output = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

prediction = int(np.argmax(output))
print(f"Prediction: {prediction}")

