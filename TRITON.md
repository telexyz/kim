https://www.reddit.com/r/MachineLearning/comments/otdpkx/n_introducing_triton_opensource_gpu_programming

Sure! I'd say that the main purpose of Triton is to make GPU programming more broadly accessible to the general ML community. It does so by making it feel more like programming multi-threaded CPUs and adding a whole bunch of pythonic, torch-like syntacting sugar.

So concretely say you want to write a row-wise softmax with it. In CUDA, you'd have to manually manage the GPU SRAM, partition work between very fine-grained cuda-thread, etc. In Tensorflow, Torch or TVM, you'd basically have a very high-level `reduce` op that operates on the whole tensor. And Triton sits somewhere between that, so it lets you define a program that basically says "For each row of the tensor, in parallel, load the row, normalize it and write it back". It still works with memory pointers so you can actually handle complex data-structure, like block-sparse softmax. Triton is actually what was used by the Deepspeed team to implement block-sparse attention about a year or so ago.

Refs:

https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/ops/sparse_attention/matmul.py

https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

https://github.com/openai/triton/tree/master/python/tutorials