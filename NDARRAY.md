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

