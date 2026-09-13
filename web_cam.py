import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# 1. Load INT8 LiteRT model
# --------------------------------------------------

model_path = "models/mnist_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

print("Model loaded successfully.")
print("Input:", input_details[0]["shape"], input_details[0]["dtype"])


# --------------------------------------------------
# 2. Start webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam started.")
print("Place the handwritten digit inside the green box.")
print("Press C to capture and predict.")
print("Press Q to quit.")


# --------------------------------------------------
# 3. Webcam loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------
    # Fixed Region of Interest
    # --------------------------------------------------

    roi_size = 300

    x1 = width // 2 - roi_size // 2
    y1 = height // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2]

    # Draw green box
    display_frame = frame.copy()

    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "Place digit here",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Webcam", display_frame)

    key = cv2.waitKey(1) & 0xFF


    # ==================================================
    # CAPTURE
    # ==================================================

    if key == ord("c"):

        # --------------------------------------------------
        # 4. Convert to grayscale
        # --------------------------------------------------

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------------------------------
        # 5. Threshold
        # --------------------------------------------------

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # --------------------------------------------------
        # 6. THICKEN THE HANDWRITING
        # --------------------------------------------------

        kernel = np.ones((3, 3), np.uint8)

        threshold = cv2.dilate(
            threshold,
            kernel,
            iterations=1
        )

        # --------------------------------------------------
        # 7. Find contours
        # --------------------------------------------------

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print("No digit detected.")
            continue

        # Largest contour
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        # Ignore tiny objects
        if w < 10 or h < 10:
            print("Digit is too small.")
            continue

        # --------------------------------------------------
        # 8. Crop digit
        # --------------------------------------------------

        digit = threshold[
            y:y+h,
            x:x+w
        ]

        # --------------------------------------------------
        # 9. Preserve aspect ratio
        # --------------------------------------------------

        max_dimension = max(w, h)

        square = np.zeros(
            (max_dimension, max_dimension),
            dtype=np.uint8
        )

        x_offset = (max_dimension - w) // 2
        y_offset = (max_dimension - h) // 2

        square[
            y_offset:y_offset+h,
            x_offset:x_offset+w
        ] = digit

        # --------------------------------------------------
        # 10. Resize to 20 × 20
        # --------------------------------------------------

        digit_20 = cv2.resize(
            square,
            (20, 20),
            interpolation=cv2.INTER_AREA
        )

        # --------------------------------------------------
        # 11. Put inside 28 × 28 MNIST image
        # --------------------------------------------------

        digit_28 = np.zeros(
            (28, 28),
            dtype=np.uint8
        )

        digit_28[4:24, 4:24] = digit_20

        # --------------------------------------------------
        # 12. Display processed image
        # --------------------------------------------------

        cv2.imshow(
            "Processed Digit",
            digit_28
        )

        # --------------------------------------------------
        # 13. Normalize
        # --------------------------------------------------

        digit_normalized = (
            digit_28.astype(np.float32) / 255.0
        )

        digit_input = digit_normalized[
            np.newaxis,
            ...,
            np.newaxis
        ]

        # --------------------------------------------------
        # 14. FLOAT32 → INT8
        # --------------------------------------------------

        digit_int8 = (
            digit_input / input_scale
            + input_zero_point
        )

        digit_int8 = np.round(
            digit_int8
        ).astype(np.int8)

        # --------------------------------------------------
        # 15. INT8 inference
        # --------------------------------------------------

        interpreter.set_tensor(
            input_index,
            digit_int8
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        # Dequantize output
        output_float = (
            output.astype(np.float32)
            - output_zero_point
        ) * output_scale

        # --------------------------------------------------
        # 16. Prediction
        # --------------------------------------------------

        prediction = int(
            np.argmax(output_float)
        )

        score = float(
            np.max(output_float)
        ) * 100

        print(
            f"Predicted digit: {prediction}"
        )

        print(
            f"Prediction score: {score:.2f}%"
        )

        # --------------------------------------------------
        # 17. Display prediction
        # --------------------------------------------------

        result_frame = frame.copy()

        cv2.rectangle(
            result_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            result_frame,
            f"Score: {score:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Prediction",
            result_frame
        )


    # ==================================================
    # QUIT
    # ==================================================

    elif key == ord("q"):
        break


# --------------------------------------------------
# 18. Close
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Webcam application closed.")