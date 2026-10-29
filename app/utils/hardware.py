"""
Hardware detection and capability reporting utilities.

This module detects available compute devices (CPU, CUDA GPUs) and reports
their capabilities, memory, and compatibility with various precision formats.
"""

import platform
import logging
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_gb: float
    # Ada Lovelace (SM 89) and later support FP8 natively
    supports_fp8: bool = False
    # Most modern GPUs support BF16
    supports_bf16: bool = False

    @property
    def compute_capability_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


@dataclass
class HardwareInfo:
    """Complete hardware profile for the inference engine."""
    # System
    os: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    total_ram_gb: float = 0.0

    # CUDA
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""
    torch_version: str = ""

    # GPUs
    gpus: list[GPUInfo] = field(default_factory=list)

    # Native extension support
    native_extension_available: bool = False

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        """Get the primary (first) GPU, if any."""
        return self.gpus[0] if self.gpus else None

    @property
    def total_gpu_memory_gb(self) -> float:
        """Total GPU memory across all devices."""
        return sum(g.total_memory_gb for g in self.gpus)

    def best_device(self) -> str:
        """Return the best available device string."""
        if self.cuda_available and self.gpus:
            return "cuda"
        return "cpu"

    def best_dtype(self) -> torch.dtype:
        """Return the best supported dtype for the primary GPU."""
        if self.cuda_available and self.primary_gpu:
            if self.primary_gpu.supports_bf16:
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def supports_precision(self, precision: str) -> bool:
        """Check if a precision mode is supported by current hardware."""
        if precision in ("float32", "auto"):
            return True
        if precision in ("float16", "bfloat16"):
            return self.cuda_available
        if precision == "int4":
            # INT4 requires CUDA and bitsandbytes
            if not self.cuda_available:
                return False
            try:
                import bitsandbytes  # noqa: F401
                return True
            except ImportError:
                return False
        if precision == "fp8":
            # FP8 requires Ada Lovelace or later (SM >= 89)
            if not self.cuda_available or not self.primary_gpu:
                return False
            return self.primary_gpu.supports_fp8
        return False

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=" * 60,
            "Hardware Profile",
            "=" * 60,
            f"  OS:          {self.os}",
            f"  CPU:         {self.cpu_name} ({self.cpu_cores}C/{self.cpu_threads}T)",
            f"  RAM:         {self.total_ram_gb:.1f} GB",
            f"  PyTorch:     {self.torch_version}",
            f"  CUDA:        {'Available' if self.cuda_available else 'Not available'}",
        ]
        if self.cuda_available:
            lines.append(f"  CUDA Ver:    {self.cuda_version}")
            lines.append(f"  cuDNN Ver:   {self.cudnn_version}")
        for gpu in self.gpus:
            lines.append(f"  GPU {gpu.index}:      {gpu.name}")
            lines.append(f"    Memory:    {gpu.total_memory_gb:.1f} GB")
            lines.append(f"    Compute:   SM {gpu.compute_capability_str}")
            lines.append(f"    BF16:      {'Yes' if gpu.supports_bf16 else 'No'}")
            lines.append(f"    FP8:       {'Yes' if gpu.supports_fp8 else 'No'}")
        if not self.gpus:
            lines.append("  GPU:         None detected")
        lines.append(f"  Native Ext:  {'Available' if self.native_extension_available else 'Not loaded'}")
        lines.append("=" * 60)
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """
    Detect all available hardware and return a HardwareInfo profile.

    This function inspects the CPU, system RAM, CUDA GPUs, and their
    capabilities. It is called once at startup.
    """
    info = HardwareInfo()

    info.os = f"{platform.system()} {platform.release()}"
    info.cpu_name = platform.processor() or "Unknown"
    info.cpu_cores = psutil.cpu_count(logical=False) or 1
    info.cpu_threads = psutil.cpu_count(logical=True) or 1
    info.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    info.torch_version = torch.__version__

    info.cuda_available = torch.cuda.is_available()
    if info.cuda_available:
        info.cuda_version = torch.version.cuda or "unknown"
        cudnn_ver = torch.backends.cudnn.version()
        info.cudnn_version = str(cudnn_ver) if cudnn_ver else "unknown"

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = (props.major, props.minor)
            gpu = GPUInfo(
                index=i,
                name=props.name,
                compute_capability=cc,
                total_memory_gb=props.total_memory / (1024 ** 3),
                supports_bf16=(cc >= (8, 0)),
                supports_fp8=(cc >= (8, 9)),
            )
            info.gpus.append(gpu)

    try:
        import llm_engine_native  # noqa: F401
        info.native_extension_available = True
    except ImportError:
        info.native_extension_available = False

    logger.info("Hardware detection complete")
    logger.debug(info.summary())
    return info


def get_gpu_memory_stats(device: int = 0) -> dict:
    """
    Get current GPU memory usage statistics.

    Returns dict with allocated, reserved, and free memory in bytes.
    Returns empty dict if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)
        ),
        "total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }


def format_bytes(num_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
