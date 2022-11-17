"""Logic for backend selection"""
import os

BACKEND = os.environ.get("KIM_BACKEND", "nd")

if BACKEND == "nd":
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cuda_triton,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )
    NDArray = array_api.NDArray

elif BACKEND == "np":
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device
    NDArray = array_api.ndarray

else:
    raise RuntimeError("Unknown kim array backend %s" % BACKEND)