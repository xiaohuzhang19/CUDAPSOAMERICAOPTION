"""
CUDA environment setup — replaces the pyopencl-based openCLEnv.

Requirements (Google Colab with GPU runtime, or any NVIDIA GPU machine):
    pip install pycuda
"""

import pycuda.autoinit          # initialises CUDA and creates a context on import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np


class CUDAEnv:
    """Thin holder for the active CUDA device and its metadata."""
    device = cuda.Device(0)
    deviceName = 'CUDA_' + device.name()

    @staticmethod
    def synchronize():
        """Block until all pending GPU work completes (replaces queue.finish())."""
        cuda.Context.synchronize()


# --------------------------------------------------------------------------- #
# Memory helpers                                                               #
# --------------------------------------------------------------------------- #

def alloc_and_copy(arr: np.ndarray) -> cuda.DeviceAllocation:
    """Allocate device memory and copy a host numpy array into it."""
    d = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(d, arr)
    return d


def alloc_empty(nbytes: int) -> cuda.DeviceAllocation:
    """Allocate uninitialised device memory of *nbytes* bytes."""
    return cuda.mem_alloc(nbytes)


def copy_to_host(dst: np.ndarray, src: cuda.DeviceAllocation):
    """Copy a device buffer into a pre-allocated host numpy array."""
    cuda.memcpy_dtoh(dst, src)


def copy_to_device(dst: cuda.DeviceAllocation, src: np.ndarray):
    """Overwrite an existing device buffer with a host numpy array."""
    cuda.memcpy_htod(dst, src)


# --------------------------------------------------------------------------- #
# Launch-configuration helper                                                  #
# --------------------------------------------------------------------------- #

def grid1d(n: int, block: int = 256):
    """Return *(block_tuple, grid_tuple)* for a 1-D launch covering *n* threads."""
    return (block, 1, 1), ((n + block - 1) // block, 1, 1)


# --------------------------------------------------------------------------- #
# Diagnostic                                                                   #
# --------------------------------------------------------------------------- #

def checkCUDA():
    print('\n' + '=' * 60)
    print('CUDA Devices')
    print('=' * 60)
    n = cuda.Device.count()
    for i in range(n):
        dev = cuda.Device(i)
        attr = dev.get_attributes()
        print(f'Device {i}: {dev.name()}')
        print(f'  Compute capability   : {dev.compute_capability()}')
        print(f'  Total memory         : {dev.total_memory() / 2**30:.1f} GB')
        print(f'  Multiprocessors      : {attr[cuda.device_attribute.MULTIPROCESSOR_COUNT]}')
        print(f'  Max threads/block    : {attr[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}')
        print(f'  Warp size            : {attr[cuda.device_attribute.WARP_SIZE]}')
        print(f'  Shared mem/block     : {attr[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] / 2**10:.0f} KB')
        print()


if __name__ == '__main__':
    checkCUDA()
