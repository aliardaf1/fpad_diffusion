import sys
import torch # or import tensorflow as tf

def check_gpu_status():
    # Check if a GPU is available for PyTorch
    print(f"Python version: {sys.version}")

    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Check your drivers and toolkit.")

if name == "main":
    check_gpu_status()
