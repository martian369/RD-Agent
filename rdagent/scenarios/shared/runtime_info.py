import platform
import subprocess
import sys
from importlib.metadata import distributions


def print_runtime_info():
    """
    打印Python运行时信息
    
    显示Python版本和操作系统信息
    """
    print("=== Python Runtime Info ===")
    print(f"Python {sys.version} on {platform.system()} {platform.release()}")


def get_gpu_info():
    """
    获取并打印GPU信息
    
    支持多种GPU检测方式：
    1. PyTorch CUDA GPU检测
    2. PyTorch Apple Silicon MPS检测
    3. nvidia-smi工具检测NVIDIA GPU
    4. system_profiler工具检测Apple Silicon GPU
    
    该函数会按优先级依次尝试不同的检测方法
    """
    # Option 1: 使用PyTorch检测GPU（支持CUDA和Apple Silicon MPS）
    try:
        import torch

        # 检查CUDA GPU
        if torch.cuda.is_available():
            print("\n=== GPU Info (via PyTorch - CUDA) ===")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        # 检查Apple Silicon GPU (MPS)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("\n=== Apple Silicon GPU Info (via PyTorch MPS) ===")
            print(f"PyTorch version: {torch.__version__}")
            
            # 获取硬件信息
            try:
                # 获取系统硬件信息
                result = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    chip_info = ""
                    memory_info = ""
                    for line in lines:
                        if "chip" in line.lower() or "processor" in line.lower():
                            chip_info = line.strip()
                        elif "memory" in line.lower() and "gb" in line.lower():
                            memory_info = line.strip()
                    
                    if chip_info:
                        print(f"Hardware: {chip_info}")
                    if memory_info:
                        print(f"System Memory: {memory_info}")
            except:
                pass
            
        else:
            print("\nNo CUDA or Apple Silicon GPU detected (PyTorch).")

    except ImportError:
        # Option 2: 使用nvidia-smi检测NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("\n=== GPU Info (via nvidia-smi) ===")
                print(result.stdout.strip())
            else:
                print("\nNo NVIDIA GPU detected (nvidia-smi not available).")
        except FileNotFoundError:
            # Option 3: 在macOS上使用system_profiler检测Apple Silicon GPU
            if platform.system() == "Darwin":  # macOS
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        # 检查是否为Apple Silicon
                        if "Apple M" in result.stdout or "Apple Graphics" in result.stdout:
                            print("\n=== GPU Info (via system_profiler) ===")
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if "Chip" in line or "Graphics" in line:
                                    if "Apple M" in line or "Apple Graphics" in line:
                                        print(line.strip())
                        else:
                            print("\nNo Apple Silicon GPU detected (system_profiler).")
                    else:
                        print("\nNo Apple Silicon GPU detected (system_profiler).")
                except FileNotFoundError:
                    print("\nNo GPU detection method available.")
            else:
                print("\nNo GPU detected (nvidia-smi not installed).")


if __name__ == "__main__":
    """
    主函数
    
    当直接运行此脚本时，会打印运行时信息和GPU信息
    """
    print_runtime_info()
    get_gpu_info()