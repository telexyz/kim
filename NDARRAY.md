## Register tiled matmul

https://youtu.be/es6s6T1bTtI?t=1482

Compute one sub-matrix at once instead of one-element at once.
- Dividing big matrix into v1 by v2 region.
- Load v1 by v3 and v2 by v3 tiled
- Internal dot product


```c
dram float C[n/v1][n/v2][v1][v2];
dram float A[n/v1][n/v3][v1][v3]; 
dram float B[n/v2][n/v3][v2][v3]; 

for (int i = 0; i < n/v1; ++i) { 
	for (int j = 0; j < n/v2; ++j) { 

		register float c[v1][v2] = 0; // output sub-matrix

		for (int k = 0; k < n/v3; ++k) { 
			register float a[v1][v3] = A[i][k]; // lhs input sub-matrix
			register float b[v2][v3] = B[j][k]; // rhs input sub-matrix
			c += dot(a, b.T);
		}

		C[i][j] = c; 
	}
}
```

## Cache line aware tiling

https://youtu.be/es6s6T1bTtI?t=1918

Trước khi load data vào register, ta load data vào L1 cache trước ...

```c
dram float C[n/b1][n/b2][b1][b2]; 
dram float A[n/b1][b1][n];
dram float B[n/b2][b2][n];

for (int i = 0; i < n/b1; ++i) {
	l1cache float a[b1][n] = A[i]; 
	for (int j = 0; j < n/b2; ++j) {
		l1cache b[b2][n] = B[j];
		C[i][j] = dot(a, b.T); // <= apply register tiling
	}
}
```

## Kết hợp cả 2

```c
dram float A[n/b1][b1/v1][n][v1]; 
dram float B[n/b2][b2/v2][n][v2];
for (int i = 0; i < n/b1; ++i) { 
	l1cache float a[b1/v1][n][v1] = A[i]; 
	for (int j = 0; j < n/b2; ++j) {
		l1cache b[b2/v2][n][v2] = B[j]; 
		for (int x = 0; x < b/v1; ++x)
		for (int y = 0; y < b/v2; ++y) { 
			register float c[v1][v2] = 0; 
			for (int k = 0; k < n; ++k) {
				register float ar[v1] = a[x][k]; 
				register float br[v2] = b[y][k];
            	C += dot(ar, br.T)
         	}
		} 
	}
}
```

- - -

The philosophy behind the NDArray class is that we want _all_ the logic for handling this structure of the array to be written in Python.  Only the "true" low level code that actually performs the raw underlying operations on the flat vector (as well as the code to manage these flat vectors, as they might need to e.g., be allocated on GPUs), is written in C++.  The precise nature of this separation will likely start to make more sense to you as you work through the assignment, but generally speaking everything that can be done in Python, is done in Python; often e.g., at the cost of some inefficiencies ... we call `.compact()` (which copies memory) liberally in order to make the underlying C++ implementations simpler.


## NDArray Data Structure

An NDArray contains the following fields:

- `handle`  The backend handle that build a flat array which stores the data
- `shape`   The shape of the NDArray
- `strides` The strides (bước) that shows how do we access multi-dimensional elements
- `offset`  The offset of the first element.
- `device`  The backend device that backs the computation


## Transformation as Strided Computation

We can leverage the strides and offset to perform transform/slicing with zero copy.

- `Broadcast`: insert strides that equals 0
- `Tranpose`: swap the strides
- `Slice`: change the offset and shape

For most of the computations, however, we will call `array.compact()` first to get a *contiguous* and *aligned memory* before running the computation.

