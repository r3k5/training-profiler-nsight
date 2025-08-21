from contextlib import contextmanager

# We try to use both CPU-side NVTX (nvtx package) and CUDA-side markers.
try:
    import nvtx as _nvtx_cpu  # CPU-side NVTX marks (shows up in Nsight Systems)
except Exception:
    _nvtx_cpu = None

try:
    import torch
    _nvtx_gpu_available = torch.cuda.is_available()
except Exception:
    _nvtx_gpu_available = False


@contextmanager
def nvtx_range(name: str):
    """
    Context manager that pushes an NVTX range recognized by Nsight Systems.
    It uses both CPU (nvtx) and GPU (torch.cuda.nvtx) markers when available.
    """
    # CPU-side push
    if _nvtx_cpu is not None:
        _cpu_range = _nvtx_cpu.annotate(message=name, color="blue")
        _cpu_range.__enter__()
    else:
        _cpu_range = None

    # GPU-side push
    if _nvtx_gpu_available:
        try:
            torch.cuda.nvtx.range_push(name)
        except Exception:
            pass

    try:
        yield
    finally:
        # GPU-side pop
        if _nvtx_gpu_available:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        # CPU-side pop
        if _cpu_range is not None:
            _cpu_range.__exit__(None, None, None)