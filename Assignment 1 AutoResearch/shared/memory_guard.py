"""
Memory guard to prevent MacBook crashes.
Monitors memory usage and raises an error before the system runs out.
"""
import os
import gc
import resource
import subprocess


def get_memory_info():
    """Get current memory usage of this process in MB."""
    # ru_maxrss is in bytes on macOS
    usage_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    usage_mb = usage_bytes / (1024 * 1024)
    return usage_mb


def get_system_available_mb():
    """Get available system memory in MB (macOS)."""
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        page_size = 16384  # default on Apple Silicon
        for line in lines:
            if "page size" in line.lower():
                page_size = int("".join(c for c in line if c.isdigit()))
                break

        free_pages = 0
        for line in lines:
            if "Pages free" in line:
                free_pages = int("".join(c for c in line.split(":")[1] if c.isdigit()))
                break

        return (free_pages * page_size) / (1024 * 1024)
    except Exception:
        return 99999  # fail open


def check_memory(label="", limit_mb=8000):
    """
    Check if process memory exceeds limit. Log current usage.
    Returns current usage in MB.
    """
    usage = get_memory_info()
    available = get_system_available_mb()
    if label:
        print(f"  [mem] {label}: process={usage:.0f}MB, system_free={available:.0f}MB")

    if usage > limit_mb:
        gc.collect()
        usage = get_memory_info()
        if usage > limit_mb:
            raise MemoryError(
                f"Process using {usage:.0f}MB, limit is {limit_mb}MB. "
                f"Stopping to prevent system crash."
            )
    return usage


def cleanup(*objects):
    """Delete objects and force garbage collection."""
    for obj in objects:
        del obj
    gc.collect()


def set_memory_limit_mb(limit_mb):
    """Set a soft memory limit for the process (advisory on macOS)."""
    limit_bytes = limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error):
        # macOS may not support RLIMIT_AS, that's OK
        pass
