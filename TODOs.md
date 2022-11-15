## Direction: GPU techniques and tools to turbocharge deep learning

**Motivation**: simple GPU backend implementation like cuda nd_backend in hw3 did not yied much difference from numpy cpu backend. To make Needle to be helpful in real datasets and problems, we need to speed it up more by digging into GPU advanced techniques and tools including cuda cores, mixed precision, specific GPU kernels (spmm, flash-attention), fused-ops ... 

Especially to save memory when dealing with large model or to utilize small mem GPUs better, `mix-presision`, `quantization` and `sparsity` are very promissing.

- [-] hw4

- [ ] Impl triton backend for ndarray to test the water

- [ ] Use triton do utilize tensor cores for matmul and mixed precision (like PyTorch AMP)

- [ ] Try flash attention triton version on rtx 3050ti

- [ ] Impl spmm (spare matmul) in triton

- [ ] Try https://github.com/TimDettmers/bitsandbytes

- - -


- [ ] Finish Paper [Sparse GPU Kernels for Deep Learning](docs/sparse.pdf)

- [ ] Impl [conv2d](https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb)

- [ ] Impl    [rnn](https://github.com/dlsyscourse/public_notebooks/blob/main/rnn_implementation.ipynb)

- [x] Impl    [gan](https://github.com/dlsyscourse/public_notebooks/blob/main/17_generative_adversarial_networks_implementation.ipynb)

- [x] Integrate CUDA backend to autograd
