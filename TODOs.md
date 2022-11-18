## GPU techniques and tools to turbocharge deep learning

**Motivation**: simple GPU backend implementation like cuda nd_backend in hw3 did not yied much difference from numpy cpu backend. To make Needle to be helpful in real datasets and problems, we need to speed it up more by digging into GPU advanced techniques and tools including cuda cores, mixed precision, specific GPU kernels (spmm, flash-attention), fused-ops ... 

Especially to save memory when dealing with large model or to utilize small mem GPUs better, `mix-presision`, `quantization` and `sparsity` are very promissing.

- [ ] Finish hw4 (WIP)

- [ ] Impl triton backend for ndarray to test the water (WIP)

- [ ] Use triton for matmul and advanced kernels and fused ops

- - -

- [ ] Finish paper [Sparse GPU Kernels for Deep Learning](docs/sparse.pdf)

- [ ] Impl spmm (spare matmul) in triton?

- - -

- [ ] https://github.com/dlsyscourse/public_notebooks/blob/main/transformer_implementation.ipynb

- [ ] Try flash attention triton version on rtx 3050ti
