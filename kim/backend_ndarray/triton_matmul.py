# Modified from https://github.com/openai/triton/blob/master/python/tutorials/03-matrix-multiplication.py

import torch
import triton
import triton.language as tl

__BLOCK_SIZE_K = 32
__GROUP_SIZE_M = 8
# Note: more than one configs will make test code slow !!!
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N':  64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  32}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N':  64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,

    # Matrix dimensions
    M, N, K,

    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,

    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, # M = n1 * BLOCK_SIZE_M
    BLOCK_SIZE_N: tl.constexpr, # N = n2 * BLOCK_SIZE_N
    ACTIVATION: tl.constexpr,   # n1, n2 are > 0 integers
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # Details https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations

    BLOCK_SIZE_K: tl.constexpr = __BLOCK_SIZE_K
    GROUP_SIZE_M: tl.constexpr = __GROUP_SIZE_M

    # program ID
    pid = tl.program_id(axis=0)

    # Mỗi pid (program id) tính toán trên 1 khối BLOCK_SIZE_M x BLOCK_SIZE_N
    # nên blocks tương đương programs

    # number of blocks (programs) along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # number of blocks (programs) along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # nhóm blocks theo chiều dọc (M), GROUP_SIZE_M blocks vào thành nhóm
    # number of blocks (programs) in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # id of the group this program is in
    group_id = pid // num_pid_in_group

    # (row-)id of the first program in the group
    first_pid_in_group_m = group_id * GROUP_SIZE_M

    # if `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_in_group_m, GROUP_SIZE_M)

    # Đoạn này triton _tự động_ phân chia programs vào *launch grid* theo pid_m và pid_n
  
    # *within groups*, programs are ordered in a column-major order
    # row-id of the program in the *launch grid*
    pid_m = first_pid_in_group_m + (pid % group_size_m)

    # col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # Details https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html#pointer-arithmetics

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_kk = tl.arange(0, BLOCK_SIZE_K)

    # Đoạn này chưa hiểu quy ước [:, None] và [None, :] có nghĩa là gì?
    # Để mở rộng từ tl.arange từ 1 chiều sang 2 chiều?

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_kk[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_kk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # We accumulate along the K dimension
        c += tl.dot(a, b)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # you can fuse arbitrary activation functions here
    if ACTIVATION == "leaky_relu": c = leaky_relu(c)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

# %%
# We can now create a convenience wrapper function that only takes two input tensors
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel


def matmul(a: torch.Tensor, b: torch.Tensor, activation="") -> torch.Tensor:
    # checks constraints
    assert a.ndim == 2 and b.ndim == 2, "only support 2D @ 2D"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    assert (
        K % __BLOCK_SIZE_K == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # n1 = M // BLOCK_SIZE_M, n2 = n // BLOCK_SIZE_N

    # Khởi tạo n1 * n2 programs, mỗi program (pid) tính kết quả nhân ma trận cho
    # 1 block BLOCK_SIZE_M * BLOCK_SIZE_M và ghi vào ma trận c
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c
