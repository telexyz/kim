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

