import torch
import subprocess
import sys

def check_nvidia_smi():
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            print("NVIDIA-SMI Output:")
            print(nvidia_smi.stdout)
        else:
            print("nvidia-smi command failed or GPU not found")
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA driver might not be installed.")

def check_system_config():
    print("\n=== System Configuration ===")
    
    # Python version
    print(f"\nPython version: {sys.version.split()[0]}")
    
    # PyTorch version and configuration
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability and details
    print(f"\n=== CUDA Configuration ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Additional CUDA details
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Details:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    # ROCm availability (for AMD GPUs)
    print(f"\n=== ROCm Configuration ===")
    print(f"ROCm available: {hasattr(torch.version, 'hip')}")

if __name__ == "__main__":
    print("=== GPU and PyTorch Configuration Check ===\n")
    
    # Check nvidia-smi
    check_nvidia_smi()
    
    # Check system configuration
    check_system_config() 