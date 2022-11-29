#!/usr/bin/python3

import numpy as np
import torch
from kim import backend_ndarray as nd

k = 2**11

# numpy matmul
A_np = np.random.randn(k, k)
B_np = np.random.randn(k, k)

# cpu matmul
A_cpu = nd.array(A_np, device=nd.cpu(), dtype="float32")
B_cpu = nd.array(B_np, device=nd.cpu(), dtype="float32")

# cuda simple matmul
A_cuda_simple = nd.array(A_np[:-1,:-1], device=nd.cuda())
B_cuda_simple = nd.array(B_np[:-1,:-1], device=nd.cuda())

# cuda tiled matmul
A_cuda_tiled = nd.array(A_np[:-4,:-4], device=nd.cuda())
B_cuda_tiled = nd.array(B_np[:-4,:-4], device=nd.cuda())

# cuda shared mememory tiled matmul
A_cuda = nd.array(A_np, device=nd.cuda())
B_cuda = nd.array(B_np, device=nd.cuda())

# cuda triton
# A_triton = nd.array(A_np, device=nd.cuda_triton())
# B_triton = nd.array(B_np, device=nd.cuda_triton())

## torch
A_torch = torch.tensor(A_np, dtype=torch.float32, device=torch.device("cuda"))
B_torch = torch.tensor(B_np, dtype=torch.float32, device=torch.device("cuda"))

def init():
	# numpy matmul
	A_np = np.random.randn(k, k)
	B_np = np.random.randn(k, k)

	# cuda shared mememory tiled matmul
	A_cuda = nd.array(A_np, device=nd.cuda())
	B_cuda = nd.array(B_np, device=nd.cuda())

	# cuda triton
	# A_triton = nd.array(A_np, device=nd.cuda_triton())
	# B_triton = nd.array(B_np, device=nd.cuda_triton())

	## torch
	A_torch = torch.tensor(A_np, dtype=torch.float32, device=torch.device("cuda"))
	B_torch = torch.tensor(B_np, dtype=torch.float32, device=torch.device("cuda"))


import timeit
n = 50
print(A_np.shape, "@", B_np.shape, "repeat", n)
print("torch:       ", timeit.timeit(lambda: init() or A_torch @ B_torch, number=n))
# print("cuda triton: ", timeit.timeit(lambda: init() or A_triton @ B_triton, number=n))
print("cuda shared: ", timeit.timeit(lambda: init() or A_cuda @ B_cuda, number=n))
print("cuda tiled:  ", timeit.timeit(lambda: init() or A_cuda_tiled @ B_cuda_tiled, number=n))
# print("cuda simple:", timeit.timeit(lambda: init() or A_cuda_simple @ B_cuda_simple, number=n))
# print("numpy:       ", timeit.timeit(lambda: init() or A_np @ B_np, number=n))
# print("cpu:        ", timeit.timeit(lambda: init() or A_cpu @ B_cpu, number=n))

''' > python3 speed_matmul.py
(2048, 2048) @ (2048, 2048) repeat 100
torch:        0.719944774002215
cuda shard:   1.801550069998484
cuda tiled:   2.204254189000494
cuda simple: 23.082233955999982
numpy:        8.142443960998207
'''
