https://forum.dlsyscourse.org/t/q3-tensor-dtype-mismatch/2297/10

The actual reason is numpy’s type promotions (described in numpy.result_type reference page 1).

If numpy determines that it can’t perform operation losslessly using dtypes of its arguments, then numpy will choose dtype of output value (type promotion) to be large enough to hold result without loss in precision.

And it could happen not only when dividing float32 by int32, but in any other calculation using these dtypes. You can use np.result_type(dtype1, dtype2) to check type of output value.

np.result_type('float32', 'int32') results in float64

