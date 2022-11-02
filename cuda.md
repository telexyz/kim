https://forums.developer.nvidia.com/t/difference-between-threadidx-blockidx-statements/12161

If you declared a two dimensional block size, say (3,3), then `threadIdx.x` would be 0,1,2 and you would now have a `threadIdx.y` value corresponding to 0,1,2. There are actually nine threads associated with the (3,3) block size. For instance, the thread indices (0,0) (0,1) (1,2) etc refer to independent threads.

This convention is very useful for two dimensional applications like working with matrices. Remember, `threadIdx.x` starts at 0 for each block. Your block can be up to three dimensions which allows for a `threadIdx.z` index as well.

Compare to ndarray:
- `Dim` just like `shape` (1,9,5)
- `Idx` just like `index` (0..Dim[x]-1)

Then:
- `threadIdx.y`: 0..`blockDim.y`-1
- `threadIdx.x`: 0..`blockDim.x`-1
- blockSize = blockDim.x * blockDim.y (number of threads in a block)
- blockId = blockIdx.y * gridDim.x + blockIdx.x
- threadId = threadIdx.y * blockDim.x + threadIdx.x
- globalId = blockId * blockSize + threadId

```c
// Cooperative Fetching
// sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
int nthreads = blockDim.y * blockDim.x; // = block size, just like ndarray.shape
int tid = threadIdx.y * blockDim.x + threadIdx.x;
for(int offset = 0; offset < S * L; offset += nthreads) {
  int y = (offset + tid) / L;
  int x = (offset + tid) % L;
  sA[y, x] = A[k + y, yblock * L + x];
}
```