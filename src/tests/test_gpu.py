import tensorflow as tf
import sys

def check_gpu():
    print("--- System Information ---")
    print(f"Python Version: {sys.version}")
    

    # Check TensorFlow GPU support
    print("\n--- TensorFlow Check ---")
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs Found: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"Device {i}: {gpu}")
    else:
        print("TensorFlow cannot find a GPU.")

if __name__ == "__main__":
    check_gpu()
