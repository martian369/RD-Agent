import subprocess
import sys
import platform
from pathlib import Path
from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env
from functools import lru_cache


@lru_cache(maxsize=3)
def get_runtime_environment_by_env(env: Env) -> str:
    # Get runtime info and GPU info
    runtime_info = print_runtime_info()
    gpu_info = get_gpu_info()
    
    # Combine all info
    combined_info = runtime_info + "\n" + gpu_info
    return combined_info

@lru_cache(maxsize=1)
def get_runtime_environment_by_env() -> str:
    # Get runtime info and GPU info
    runtime_info = print_runtime_info()
    gpu_info = get_gpu_info()
    
    # Combine all info
    combined_info = runtime_info + "\n" + gpu_info
    return combined_info

def print_runtime_info() -> str:
    """
    获取Python运行时信息字符串
    
    显示Python版本和操作系统信息
    """
    info = "=== Python Runtime Info ===\n"
    info += f"Python {sys.version} on {platform.system()} {platform.release()}"
    return info


def get_gpu_info() -> str:
    """
    获取GPU信息字符串
    
    支持多种GPU检测方式：
    1. PyTorch CUDA GPU检测
    2. PyTorch Apple Silicon MPS检测
    3. nvidia-smi工具检测NVIDIA GPU
    4. system_profiler工具检测Apple Silicon GPU
    
    该函数会按优先级依次尝试不同的检测方法
    """
    info = ""
    
    # Option 1: 使用PyTorch检测GPU（支持CUDA和Apple Silicon MPS）
    try:
        import torch

        # 检查CUDA GPU
        if torch.cuda.is_available():
            info += "\n=== GPU Info (via PyTorch - CUDA) ===\n"
            info += f"CUDA Version: {torch.version.cuda}\n"
            info += f"GPU Device: {torch.cuda.get_device_name(0)}\n"
            info += f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n"
            info += f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB\n"
            info += f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB\n"
        # 检查Apple Silicon GPU (MPS)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            info += "\n=== Apple Silicon GPU Info (via PyTorch MPS) ===\n"
            info += f"PyTorch version: {torch.__version__}\n"
            
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
                        info += f"Hardware: {chip_info}\n"
                    if memory_info:
                        info += f"System Memory: {memory_info}\n"
            except:
                pass
            
        else:
            info += "\nNo CUDA or Apple Silicon GPU detected (PyTorch).\n"

    except ImportError:
        # Option 2: 使用nvidia-smi检测NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info += "\n=== GPU Info (via nvidia-smi) ===\n"
                info += result.stdout.strip() + "\n"
            else:
                info += "\nNo NVIDIA GPU detected (nvidia-smi not available).\n"
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
                            info += "\n=== GPU Info (via system_profiler) ===\n"
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if "Chip" in line or "Graphics" in line:
                                    if "Apple M" in line or "Apple Graphics" in line:
                                        info += line.strip() + "\n"
                        else:
                            info += "\nNo Apple Silicon GPU detected (system_profiler).\n"
                    else:
                        info += "\nNo Apple Silicon GPU detected (system_profiler).\n"
                except FileNotFoundError:
                    info += "\nNo GPU detection method available.\n"
            else:
                info += "\nNo GPU detected (nvidia-smi not installed).\n"
    
    return info


def check_runtime_environment(env: Env) -> str:
    implementation = FBWorkspace()
    # 1) Check if strace exists in env
    strace_check = implementation.execute(env=env, entry="which strace || echo MISSING").strip()
    if strace_check.endswith("MISSING"):
        raise RuntimeError("`strace` not found in the target environment.")

    # 2) Check if coverage module works in env
    coverage_check = implementation.execute(env=env, entry="python -m coverage --version || echo MISSING").strip()
    if coverage_check.endswith("MISSING"):
        raise RuntimeError("`coverage` module not found or not runnable in the target environment.")