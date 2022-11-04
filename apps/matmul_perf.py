import numpy as np
import torch
from kim import backend_ndarray as nd
k = 2**11
# numpy matmul
A_np = np.random.randn(k, k)
B_np = np.random.randn(k, k)
# cpu matmul
A_cpu = nd.array(A_np, device=nd.cpu())
B_cpu = nd.array(B_np, device=nd.cpu())
# cuda simple matmul
A_cuda_simple = nd.array(A_np[:-2,:-2], device=nd.cuda())
B_cuda_simple = nd.array(B_np[:-2,:-2], device=nd.cuda())
# cuda shared mememory tiled matmul
A_cuda_best = nd.array(A_np, device=nd.cuda())
B_cuda_best = nd.array(B_np, device=nd.cuda())
## torch
A_torch = torch.tensor(A_np)
B_torch = torch.tensor(B_np)

%%timeit
A_np @ B_np

%%timeit
A_cuda_simple @ B_cuda_simple

%%timeit
A_cuda_best @ B_cuda_best

%%timeit
torch.cuda.FloatTensor(k, k) @ torch.cuda.FloatTensor(k, k)
