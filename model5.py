import tensorflow as tf
import numpy as np
import time
import os
import psutil
import gc


# ==================================================
# Load MNIST test dataset
# ==================================================

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., np.newaxis]


# ==================================================
# Benchmark function
# ==================================================

def benchmark_model(model_path, quantized=False):

    print("\n====================================")
    print("Model:", model_path)
    print("====================================")

    gc.collect()

    process = psutil.Process(os.getpid())

    # Memory before loading model
    memory_before = process.memory_info().rss

    # Load LiteRT model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    print("Input type:", input_dtype)
    print("Output type:", output_dtype)

    # Memory after loading
    memory_after_load = process.memory_info().rss

    # ==================================================
    # Warm-up
    # ==================================================

    for i in range(100):

        image = x_test[i:i+1]

        if quantized:
            scale, zero_point = input_details[0]["quantization"]

            image = image / scale + zero_point
            image = np.round(image).astype(np.int8)

        interpreter.set_tensor(input_index, image)
        interpreter.invoke()

    # ==================================================
    # Accuracy on all 10,000 images
    # ==================================================

    correct = 0

    for i in range(len(x_test)):

        image = x_test[i:i+1]

        if quantized:
            scale, zero_point = input_details[0]["quantization"]

            image = image / scale + zero_point
            image = np.round(image).astype(np.int8)

        interpreter.set_tensor(input_index, image)
        interpreter.invoke()

        output = interpreter.get_tensor(output_index)

        if quantized:
            output_scale, output_zero_point = (
                output_details[0]["quantization"]
            )

            output = (
                output.astype(np.float32) - output_zero_point
            ) * output_scale

        prediction = np.argmax(output)

        if prediction == y_test[i]:
            correct += 1

    accuracy = correct / len(x_test) * 100

    # ==================================================
    # Latency benchmark
    # ==================================================

    num_latency_runs = 2000

    image = x_test[0:1]

    if quantized:
        scale, zero_point = input_details[0]["quantization"]

        latency_image = image / scale + zero_point
        latency_image = np.round(latency_image).astype(np.int8)
    else:
        latency_image = image

    start_time = time.perf_counter()

    for _ in range(num_latency_runs):

        interpreter.set_tensor(
            input_index,
            latency_image
        )

        interpreter.invoke()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    average_latency_ms = (
        total_time / num_latency_runs
    ) * 1000

    # ==================================================
    # Memory after inference
    # ==================================================

    memory_after_inference = process.memory_info().rss

    memory_load = (
        memory_after_load - memory_before
    ) / (1024 * 1024)

    memory_inference = (
        memory_after_inference - memory_after_load
    ) / (1024 * 1024)

    # ==================================================
    # Display results
    # ==================================================

    print(f"\nAccuracy: {accuracy:.2f}%")

    print(
        f"Average inference latency: "
        f"{average_latency_ms:.4f} ms"
    )

    print(
        f"Memory increase after model loading: "
        f"{memory_load:.2f} MB"
    )

    print(
        f"Additional memory after inference: "
        f"{memory_inference:.2f} MB"
    )

    return {
        "accuracy": accuracy,
        "latency": average_latency_ms,
        "memory": memory_load
    }


# ==================================================
# Benchmark FP32
# ==================================================

fp32_results = benchmark_model(
    "models/mnist_fp32.tflite",
    quantized=False
)


# ==================================================
# Benchmark INT8
# ==================================================

int8_results = benchmark_model(
    "models/mnist_int8.tflite",
    quantized=True
)


# ==================================================
# Final comparison
# ==================================================

print("\n\n====================================")
print("FINAL FP32 vs INT8 COMPARISON")
print("====================================")

print("\n                 FP32       INT8")
print("------------------------------------")

print(
    f"Accuracy       "
    f"{fp32_results['accuracy']:.2f}%     "
    f"{int8_results['accuracy']:.2f}%"
)

print(
    f"Latency        "
    f"{fp32_results['latency']:.4f} ms    "
    f"{int8_results['latency']:.4f} ms"
)

print(
    f"Memory         "
    f"{fp32_results['memory']:.2f} MB     "
    f"{int8_results['memory']:.2f} MB"
)