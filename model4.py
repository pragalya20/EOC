import os

# Model paths
fp32_path = "models/mnist_fp32.tflite"
int8_path = "models/mnist_int8.tflite"

# Get file sizes
fp32_size = os.path.getsize(fp32_path)
int8_size = os.path.getsize(int8_path)

# Convert bytes to KB
fp32_kb = fp32_size / 1024
int8_kb = int8_size / 1024

# Calculate size reduction
size_reduction = (1 - int8_size / fp32_size) * 100

print("Model Size Comparison")
print("---------------------")
print(f"FP32 model: {fp32_kb:.2f} KB")
print(f"INT8 model: {int8_kb:.2f} KB")
print(f"Size reduction: {size_reduction:.2f}%")