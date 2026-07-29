#!/usr/bin/env python3
"""
test_gpu.py — Quick GPU verification for EDOS project on H100 NVL 94GB
Usage: python test_gpu.py
"""

import torch
import sys


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    print_section("PyTorch & CUDA Environment")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is NOT available. Check your PyTorch installation.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"GPU count       : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / (1024 ** 3)

        print_section(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Device name          : {props.name}")
        print(f"  Compute capability   : {props.major}.{props.minor}")
        print(f"  Total VRAM           : {total_vram_gb:.2f} GB")
        print(f"  Multi-Processor count: {props.multi_processor_count}")

        # Quick VRAM allocation test
        allocated_before = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved_before = torch.cuda.memory_reserved(i) / (1024 ** 2)
        print(f"  VRAM allocated       : {allocated_before:.2f} MB")
        print(f"  VRAM reserved        : {reserved_before:.2f} MB")

    print_section("Functional Test — Matrix Multiplication")
    device = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float32)

    # Warm-up
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"  MatMul 4096×4096 @ FP32: {elapsed_ms:.2f} ms")
    print(f"  Result shape           : {c.shape}")
    print(f"  Result device          : {c.device}")

    # VRAM after test
    allocated_after = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"  VRAM allocated (post)  : {allocated_after:.2f} MB")

    print_section("Summary")
    print("  ✓ CUDA is available and functional")
    print("  ✓ GPU(s) detected and accessible")
    print("  ✓ Matrix multiplication executed successfully")
    print("\n  Ready to proceed with EDOS training!\n")


if __name__ == "__main__":
    main()
