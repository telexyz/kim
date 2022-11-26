import pytest
from kim import backend_ndarray as nd
import os


CPU_CUDA = [
    # "cpu_numpy", 
    # "cpu", 
    "cuda", 
    # "cuda_triton"
]

_DEVICES = [
    # nd.cpu_numpy(),
    # nd.cpu(), 
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")),
    # pytest.param(nd.cuda_triton(), marks=pytest.mark.skipif(not nd.cuda_triton().enabled(), reason="No GPU"))
]

device_name = os.environ.get("KIM_DEVICE", None)
if device_name is not None:
    for i, v in enumerate(CPU_CUDA):
        if device_name == v:
            CPU_CUDA = [device_name]
            _DEVICES = [_DEVICES[i]]
            break

if os.environ.get("KIM_BACKEND", None) == "np": # skip test for original numpy backend (no devices for this backend)
    CPU_CUDA = []
    _DEVICES = []
