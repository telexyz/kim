https://forum.dlsyscourse.org/t/q3-tensor-dtype-mismatch/2297/10

The actual reason is numpy’s type promotions (described in numpy.result_type reference page 1).

If numpy determines that it can’t perform operation losslessly using dtypes of its arguments, then numpy will choose dtype of output value (type promotion) to be large enough to hold result without loss in precision.

And it could happen not only when dividing float32 by int32, but in any other calculation using these dtypes. You can use np.result_type(dtype1, dtype2) to check type of output value.

np.result_type('float32', 'int32') results in float64

## conv

```py
import numpy as np
import ctypes
A = np.arange(36, dtype=np.float32).reshape(6,6)

# 4*(np.array((6,1,6,1)))
B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=(24, 4, 24, 4))

C = B.reshape(16,9)

W = np.arange(9, dtype=np.float32).reshape(3,3)

(B.reshape(16,9) @ W.reshape(9)).reshape(4,4)
[```