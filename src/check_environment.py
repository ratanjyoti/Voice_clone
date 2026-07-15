import os
import platform
import shutil
import subprocess
import sys


def get_memory_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        return None


def get_cpu_name() -> str:
    cpu = platform.processor()

    if cpu:
        return cpu

    try:
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor).Name",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    except Exception:
        pass

    return "Unknown"


def check_nvidia_gpu() -> None:
    print("\nNVIDIA GPU information")
    print("-" * 50)

    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found.")
        print("No usable NVIDIA GPU was detected, or NVIDIA drivers are unavailable.")
        return

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("nvidia-smi exists, but GPU information could not be read.")
        print(result.stderr.strip())


def check_pytorch() -> None:
    print("\nPyTorch information")
    print("-" * 50)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"GPU detected by PyTorch: {torch.cuda.get_device_name(0)}")
            properties = torch.cuda.get_device_properties(0)
            total_vram = properties.total_memory / (1024**3)
            print(f"GPU VRAM: {total_vram:.2f} GB")

    except ImportError:
        print("PyTorch is not installed yet. This is expected at this stage.")


def main() -> None:
    print("Infinia Voice Case Study - Environment Report")
    print("=" * 50)

    print(f"Operating system: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Processor: {get_cpu_name()}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"Logical CPU count: {os.cpu_count()}")

    memory = get_memory_gb()
    if memory is None:
        print("RAM: psutil is not installed")
    else:
        print(f"RAM: {memory} GB")

    check_nvidia_gpu()
    check_pytorch()


if __name__ == "__main__":
    main()