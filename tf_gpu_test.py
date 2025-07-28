import tensorflow as tf
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Enable device placement logging to see where ops run
# This is crucial for verifying GPU usage in the logs
tf.debugging.set_log_device_placement(True)

print("\n--- Starting WSL2 TensorFlow GPU Verification Test ---")

# 1. List physical GPUs
physical_gpus = tf.config.list_physical_devices('GPU')
print(f"Num Physical GPUs Available: {len(physical_gpus)}")
for gpu in physical_gpus:
    print(f"  {gpu}")

# 2. List logical GPUs (should appear if physical are found and initialized)
logical_gpus = tf.config.list_logical_devices('GPU')
print(f"Num Logical GPUs Available: {len(logical_gpus)}")
for gpu in logical_gpus:
    print(f"  {gpu}")

# 3. Check if TensorFlow was built with CUDA support (should be True)
print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

# 4. Perform a small operation explicitly on GPU and check its device
try:
    # tf.test.gpu_device_name() is a quick check, but actual device placement logs are more definitive
    device_name = tf.test.gpu_device_name()
    print(f"tf.test.gpu_device_name() reports: {device_name}")

    # Explicitly place a small operation on the GPU:0
    with tf.device('/GPU:0'): # Explicitly target GPU 0 in case it's not default
         a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
         b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
         c = tf.matmul(a, b)
         print(f"Explicit GPU MatMul result: {c.numpy()}")
         # Check the device of the tensor after the operation
         print(f"Explicit GPU MatMul ran on device: {c.device}")


    # 5. Perform a slightly larger operation to observe GPU utilization
    # This part is more likely to show up on nvidia-smi
    matrix_size = 2048 # Adjust this size if you want more/less load
    A = tf.random.normal((matrix_size, matrix_size), dtype=tf.float32)
    B = tf.random.normal((matrix_size, matrix_size), dtype=tf.float32)

    start_time = time.time()
    with tf.device('/GPU:0'): # Again, explicitly targeting GPU 0
        C = tf.matmul(A, B)
        D = tf.nn.relu(C)
        E = tf.reduce_sum(D)
    end_time = time.time()

    print(f"\nLarger tensor 'E' is on device: {E.device}")
    print(f"Large matrix ops took {end_time - start_time:.4f} seconds.")
    print("TensorFlow GPU operations seem to be working!")

except Exception as e:
    print(f"\n--- GPU ERROR: {e} ---")
    print("TensorFlow failed to run operations on GPU in WSL2. Error details:")
    print(e) # Print the actual error message

print("\n--- End of WSL2 TensorFlow GPU Verification Test ---")