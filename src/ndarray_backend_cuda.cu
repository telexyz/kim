#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 4

const size_t L = 8 * TILE; // => 8 * 8 = 64 threads; 16 * 16 = 256 threads
const size_t S = 8 * TILE; // => 16 * 4 * 8 * 4 * 4-byte = 8k shared memory

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) {
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  }
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, 
    CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact operation.
   * This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    size_t a_idx = 0; 
    size_t remain = gid;
    size_t stride = size;
    size_t indexx = 0;
    for(size_t i = 0; i < shape.size; i++) {
      stride = stride / shape.data[i];
      indexx = remain / stride;
      remain = remain % stride;
      a_idx += strides.data[i] * indexx;
    }
    out[gid] = a[a_idx + offset];
    /// END YOUR SOLUTION
  }
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory. Unlike the C++ version, in CUDA this will 
   * primarily call the relevant CUDA kernel. In this case, we illustrate 
   * how you should set this up (i.e., we give you the code for this fuction, 
   * and also the prototype for the CompactKernel() function).  For the functions 
   * after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset)
   */
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, 
    VecToCuda(shape), VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, 
    CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    size_t out_idx = 0;
    size_t remain = gid;
    size_t stride = size;
    size_t indexx = 0;
    // 
    for(size_t i = 0; i < shape.size; i++) {
      stride = stride / shape.data[i];
      indexx = remain / stride;
      remain = remain % stride;
      out_idx += strides.data[i] * indexx;
    }
    out[out_idx + offset] = a[gid];
    /// END YOUR SOLUTION
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA. 
   * You will most likely want to implement a EwiseSetitemKernel() function, 
   * similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, 
    VecToCuda(shape), VecToCuda(strides), offset);  
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, 
    CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    size_t out_idx = 0;
    size_t remain = gid;
    size_t stride = size;
    size_t indexx = 0;
    // 
    for(size_t i = 0; i < shape.size; i++) {
      stride = stride / shape.data[i];
      indexx = remain / stride;
      remain = remain % stride;
      out_idx += strides.data[i] * indexx;
    }
    out[out_idx + offset] = val;
    /// END YOUR SOLUTION
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will not be 
   *      the same as out.size, because out is a non-compact subset array);
   *      it _will_ be the same as the product of items in shape, 
   *      but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, 
    VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower

 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe

 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

/// BEGIN YOUR SOLUTION
__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = expf(a[gid]); }
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = tanhf(a[gid]); }
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = logf(a[gid]); }
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] > b[gid] ? 1.0 : 0.0; }
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] >= val ? 1.0 : 0.0; }
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] == b[gid] ? 1.0 : 0.0; }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] == val ? 1.0 : 0.0; }
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = fmaxf(a[gid], b[gid]); }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = fmaxf(a[gid], val); }
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = powf(a[gid], val); }
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] * b[gid]; }
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] * val; }
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] / b[gid]; }
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) { out[gid] = a[gid] / val; }
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void SimpleMatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < M * P) {
    const size_t i = gid / P;
    const size_t j = gid % P;
    scalar_t tmp = 0;
    for (size_t k = 0; k < N; k++) {
      tmp += a[i * N + k] * b[k * P + j];
    }
    out[gid] = tmp;
  }
}


__global__ void MatmulTiledKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, uint32_t N, uint32_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    const size_t P_T = P / TILE;
    const size_t ybase = (gid / P_T) * TILE;
    const size_t xbase = (gid % P_T) * TILE;
    
    float c_t[TILE][TILE], a_t[TILE], b_t[TILE];
    for (size_t i = 0; i < TILE; ++i)
      for (size_t j = 0; j < TILE; ++j)
        c_t[i][j] = 0;

    for (size_t k = 0; k < N; ++k) {
      // Khởi tạo mảng a_t, b_t
      for (size_t o = 0; o < TILE; ++o) { 
        a_t[o] = a[(ybase + o)*N + k];
        b_t[o] = b[k*P + (xbase + o)];
      }
      // Tính toán trên local vars
      for (size_t i = 0; i < TILE; ++i)
        for (size_t j = 0; j < TILE; ++j)
          c_t[i][j] += a_t[i] * b_t[j];
    }
    // Update kết quả
    for (size_t i = 0; i < TILE; ++i)
      for (size_t j = 0; j < TILE; ++j)
        out[(ybase + i)*P + (xbase + j)] = c_t[i][j];
    /// END YOUR SOLUTION
  }
}

__global__ void MatmulSharedMemKernel(const scalar_t* a, const scalar_t* b, 
  scalar_t* out, uint32_t P, uint32_t N) {

  // https://youtu.be/jYCxVirq4d0?t=2113
  // out là ma trận C trong video trên gồm M hàng, P cột,
  // xử lý theo block (L,L), L=16*TILE
  // dữ liệu lấy từ A là khối (L,S), từ B là khối (S, L)

  // Mỗi thread nhân sub-matrix(TILE, TILE)
  // Như trong trục tọa độ 2 chiều thì thì x trục tung = hàng, y trục dọc = cột

  // tới vị trí đầu của block
  const size_t yblock = blockIdx.y * blockDim.y * L;
  const size_t xblock = blockIdx.x * blockDim.x * L;
  
  float c_t[TILE][TILE], a_t[TILE], b_t[TILE];
  // local vars will be mapped to registers <= https://youtu.be/jYCxVirq4d0?t=1811
  for (size_t i = 0; i < TILE; ++i)
    for (size_t j = 0; j < TILE; ++j)
      c_t[i][j] = 0;

  __shared__ float a_s[S][L], b_s[S][L]; // khối A(L,S), khối B(S,L)

  // dịch chuyển khối A(L,S) tới hết hàng, và khối B(S,L) tới hết cột
  // A có kích cỡ M x N, B có kích cỡ N x P nên dùng chung biến k được
  for (size_t k = 0; k < N; k += S) {
    __syncthreads();
    // sA[:, :] = A[k : k + S, yblock : yblock + L];
    // sB[:, :] = B[k : k + S, xblock : xblock + L];
    for (size_t s = 0; s < S; ++s) {
      for (size_t l = 0; l < L; ++l) {
        a_s[s][l] = a[(yblock + l)*N +      (k + s)]; // a: M*N
        b_s[s][l] = a[     (k + s)*P + (xblock + l)]; // b: N*P
      }
    }
    __syncthreads();

    for (int ki = 0; ki < S; ++ki) {
      // Khởi tạo mảng a_t, b_t từ cột trong a_shared, và hàng b_shared
      for (size_t t = 0; t < TILE; ++t) { 
        // a[:] = sA[ki, threadIdx.y * V : threadIdx.y * V + V];
        // b[:] = sA[ki, threadIdx.x * V : threadIdx.x * V + V];
        a_t[t] = a_s[ki][threadIdx.y*TILE + t];
        b_t[t] = b_s[ki][threadIdx.x*TILE + t];
      }
      // Tính toán trên local vars
      for (size_t i = 0; i < TILE; ++i)
        for (size_t j = 0; j < TILE; ++j)
          c_t[i][j] += a_t[i] * b_t[j];
    }
  }

  const size_t ybase = yblock + threadIdx.y*TILE;
  const size_t xbase = xblock + threadIdx.y*TILE;
  // Update kết quả cho TILE * TILE tại ybase, xbase
  for (size_t i = 0; i < TILE; ++i)
    for (size_t j = 0; j < TILE; ++j)
      out[(ybase + i)*P + xbase + j] = c_t[i][j];
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out,
  uint32_t M, uint32_t N, uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.
   * You will want to look at the lecture and notes on GPU-based linear algebra
   * to see how to do this. We would encourage you to use cooperative fetching, 
   * shared memory register tiling, and other ideas covered in the class notes.
   * 
   * Note that unlike the tiled matmul function in the CPU backend, here you should 
   * implement a single function that works across all size matrices, 
   * whether or not they are a multiple of a tile size. As with previous CUDA
   * implementations, this function here will largely just set up the kernel call,
   *  and you should implement the logic in a separate MatmulKernel() call.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  // if (false) { /*
  if (M % L == 0 && P % L == 0 && N % S == 0) {
    // => Can do shared-mem tiling
    // Mỗi thread tính (TILE, TILE) sub-matrix
    dim3 block(L / TILE, L / TILE, 1);
    dim3 grid(P / L, M / L, 1); // => M = blockDim.y * L, P = blockDim.x * L
    // (M/L)*(P/L)*(L/TILE)*(L/TILE) = (M*L)/(TILE*TILE) = out->size/(TILE*TILE)
    MatmulSharedMemKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, P, N);
  /**/
  } else if (M % TILE == 0 && P % TILE == 0) {
    // Trường hợp M, P chia hết cho TILE thì dùng tile matmul
    size_t size = out->size / (TILE * TILE);
    CudaDims dim = CudaOneDim(size);
    MatmulTiledKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, 
        size, N, P);

  } else {
    // Nếu không thì dùng simple matmul mỗi thread tính 1 phần tử out[i,j]
    CudaDims dim = CudaOneDim(out->size);
    SimpleMatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    size_t offset = gid*reduce_size;
    scalar_t max = a[offset];
    for (size_t k = 1; k < reduce_size; k++) {
      const scalar_t tmp = a[offset + k];
      if (max < tmp) { max = tmp; }
    }
    out[gid] = max;
    /// END YOUR SOLUTION
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   * Even though it is inefficient, for simplicity you can perform each reduction
   * in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}



__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    /// BEGIN YOUR SOLUTION
    size_t offset = gid*reduce_size;
    scalar_t sum = a[offset];
    for (size_t k = 1; k < reduce_size; k++) {
      sum += a[offset + k];
    }
    out[gid] = sum;
    /// END YOUR SOLUTION
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.
   * Again, for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from GPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, 
    std::vector<size_t> strides, size_t offset) {
    // 
    std::vector<size_t> numpy_strides = strides;
    // biến đổi elems từ begin() tới end() và ghi vào bắt đầu từ begin()
    std::transform(numpy_strides.begin(), numpy_strides.end(), 
      numpy_strides.begin(), [](size_t& c) { return c * ELEM_SIZE; });

    /* copy memory to host */

    // khởi tạo vùng nhớ mới trong hosst
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) { throw std::bad_alloc(); }

    // a là CudaArray và a.ptr trỏ tới vùng nhớ trong GPU
    cudaError_t err = cudaMemcpy(
      host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(
      shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });


  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err = cudaMemcpy(
      out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });


  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}