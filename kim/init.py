import math
import kim

def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    if device is None: device = kim.default_device()
    array = device.rand(*shape) * (high - low) + low
    return kim.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
    

def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    if device is None: device = kim.default_device()
    array = device.randn(*shape) * std + mean
    return kim.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    if device is None: device = kim.default_device()
    array = device.full(shape, c, dtype=dtype)
    return kim.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad)


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    if device is None: device = kim.default_device()
    array = device.rand(*shape) <= p
    return kim.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    if device is None: device = kim.default_device()
    i = kim.as_numpy(i).astype("int32")
    return kim.Tensor(
        device.one_hot(n, i, dtype=dtype), device=device, requires_grad=requires_grad
    )


def zeros_like(array, *, device=None, requires_grad=False):
    if device is None: device = array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    if device is None: device = array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", shape=None, a=math.sqrt(5), **kwargs):
    if shape is None: shape = (fan_in, fan_out)

    # Calculate gain, and fan for diff nonlinearity
    if nonlinearity == "relu":
        gain = math.sqrt(2)
        fan = fan_in
    else:
        assert nonlinearity == "leaky_relu", "Only relu & leaky_relu supported currently"
        fan, _ = _calculate_fan_in_and_fan_out(shape)
        gain = math.sqrt(2.0 / (1 + a ** 2))

    # Calculate uniform bounds from standard deviation
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    return randn(fan_in, fan_out, mean=0, std=(2 / fan_in)**0.5, **kwargs)


# Copy from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2: raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = shape[0]
    num_output_fmaps = shape[1]
    receptive_field_size = 1
    if dimensions > 2:
        for s in shape[2:]: receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
