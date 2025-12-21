import torch
import tensorflow as tf
import sys

def check_gpu():
    print("--- System Information ---")
    print(f"Python Version: {sys.version}")
    
    # Check PyTorch GPU support
    print("\n--- PyTorch Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("PyTorch cannot find a GPU. Check your CUDA/cuDNN installation.")

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
