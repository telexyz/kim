"""The module.
"""
from typing import List, Callable, Any
from kim.autograd import Tensor
from kim import ops
import kim.init as init
import numpy as np
import kim

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter): return [value]
    if isinstance(value, Module): return value.parameters()

    params = []
    if isinstance(value, dict):
        for k, v in value.items():
            params += _unpack_params(v)
    # 
    elif isinstance(value, (list, tuple)):
        for v in value:
            params += _unpack_params(v)
    # 
    return params


def _child_modules(value: object) -> List["Module"]:
    modules = []

    if isinstance(value, Module):
        modules = [value] + _child_modules(value.__dict__)

    elif isinstance(value, dict):
        for k, v in value.items():
            modules += _child_modules(v)

    elif isinstance(value, (list, tuple)):
        for v in value:
            modules += _child_modules(v)

    return modules


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# - - - short modules - - -

class Identity(Module):
    def forward(self, x):
        return x


class Flatten(Module):
    # Takes in a tensor of shape (B, X_0, X_1, ...), and flattens all non-batch dimensions 
    # so that the output is of shape (B, X_0 * X_1 * ...)
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], kim.prod(X.shape) // X.shape[0]))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(ops.negate(x))) ** -1


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        bs, ndim = logits.shape
        y_one_hot = init.one_hot(ndim, y, dtype=logits.dtype, device=logits.device)
        logits_y = ops.summation(logits * y_one_hot, axes=(1,))
        logsum = ops.logsumexp(logits, axes=(1,))
        return ops.summation(logsum - logits_y) / bs

# - - - - - - - - - - - - -

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules: x = m(x)
        return x


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = 1 - p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training: return x
        return (x * init.randb(*x.shape, p=self.p, device=x.device)) / self.p


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_init = init.kaiming_uniform(in_features, out_features, dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_init)
        if bias is True:
            bias_init = init.kaiming_uniform(out_features, 1, dtype=dtype, device=device)
            self.bias = Parameter(ops.transpose(bias_init))
        else: self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = ops.matmul(X, self.weight)
        if self.bias is not None: out += ops.broadcast_to(self.bias, out.shape)
        return out


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, self.dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(1, self.dim, dtype=dtype, device=device))
        self.running_mean = init.zeros(self.dim, dtype=dtype, device=device)
        self.running_var = init.ones(self.dim, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        batch, dim = x.shape
        assert dim == self.dim

        if self.training:
            # only calculate mean, var and update running_mean, running_var in training
            mean = ops.summation(x, axes=0) / batch
            mean_full = mean.reshape((1, dim)).broadcast_to(x.shape)
            var = ops.summation((x - mean_full) ** 2, axes=0) / batch

            self.running_mean = ((1-self.momentum)*self.running_mean + self.momentum*mean).detach()
            self.running_var  = ((1-self.momentum)*self.running_var  + self.momentum*var).detach()
        else:
            # inference use running_mean and running_var estimated in training
            mean_full = self.running_mean.reshape((1, dim)).broadcast_to(x.shape)
            var = self.running_var

        var_full = var.reshape((1, dim)).broadcast_to(x.shape)
        norm = (x - mean_full) / ((var_full + self.eps) ** 0.5)
        
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        return w * norm + b

'''
https://www.geeksforgeeks.org/expression-for-mean-and-variance-in-a-running-stream
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
'''
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, self.dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(1, self.dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        batch, dim = x.shape
        assert dim == self.dim

        mean = ops.summation(x, axes=1) / dim
        mean = mean.reshape((batch, 1)).broadcast_to(x.shape)

        var = ops.power_scalar(x - mean, 2)
        var = ops.summation(var, axes=1) / dim
        var = var.reshape((batch, 1)).broadcast_to(x.shape)

        norm = (x - mean) / ((var + self.eps) ** 0.5)
        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        return w*norm + b


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, i, o, k, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__() # normalize kernel_size and stride so that:
        if isinstance(k, tuple): k = k[0] # Only supports square kernels and padding=same
        if isinstance(stride, tuple): stride = stride[0]

        self.in_channels = i
        self.out_channels = o
        self.kernel_size = k
        self.stride = stride

        # Initialize the (k, k, i, o) weight tensor using Kaiming uniform initialization 
        # with default settings.
        # Previously, we have implemented Kaiming uniform/normal initializations, where we essentially 
        # assigned fan_in = input_size and fan_out = output_size.
        # For convolution, this becomes somewhat more detailed, in that you should multiply both of these 
        # by the "receptive field size", which is in this case just the product of the kernel sizes 
        # -- which in our case are always going to be the same, i.e., `k x k` kernels.
        weight_init = init.kaiming_uniform(i* k**2, o* k**2, shape=(k, k, i, o), 
            dtype=dtype, device=device, requires_grad=True)
        self.weight = Parameter(weight_init)

        if bias:
            # Initialize the (o,) bias tensor using uniform initialization on the interval 
            # +/- 1.0/(in_channels * kernel_size**2)**0.5
            x = 1.0/((i* k**2)**0.5)
            self.bias = Parameter(init.rand(o, low=-x, high=x, 
                dtype= dtype, device=device, requires_grad=True))
        else:
            self.bias = None


    def forward(self, x: Tensor) -> Tensor:
        # Ensure nn.Conv works for (N, C, H, W) tensors even though we implemented 
        #          the conv op for (N, H, W, C) tensors
        xt = x.transpose(axes=(1,2)).transpose(axes=(2,3))

        # Calculate the appropriate padding to ensure input and output dimensions are the same
        out = ops.conv(xt, self.weight, padding=self.kernel_size//2, stride=self.stride)
        out = out.transpose(axes=(2,3)).transpose(axes=(1,2))

        # Calculate the convolution, then add the properly-broadcasted bias term if present
        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_channels, 1, 1))
            out += bias.broadcast_to(out.shape)
        # 
        return out


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.bias = bias
        x = 1 / hidden_size**0.5

        self.W_ih = Parameter(init.rand( input_size, hidden_size, low=-x, high=x), device=device)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-x, high=x), device=device)

        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-x, high=x), device=device)
            self.bias_hh = Parameter(init.rand(hidden_size, low=-x, high=x), device=device)


    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features (bs: batch_size)
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state for each element in the batch.
        """
        bs, input_size = X.shape
        if h is None: h = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)
        _, hidden_size = h.shape

        assert bs == h.shape[0]
        assert self.input_size == input_size
        assert self.hidden_size == hidden_size

        out = X @ self.W_ih + h @ self.W_hh

        if self.bias:
            out += self.bias_ih.reshape((1, hidden_size)).broadcast_to((bs, hidden_size))
            out += self.bias_hh.reshape((1, hidden_size)).broadcast_to((bs, hidden_size))

        if self.nonlinearity == 'tanh':
            return ops.tanh(out)
        else:
            return ops.relu(out)



class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer, of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer, of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer, of shape (hidden_size,).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.bias = bias

        # Init RNN Layers
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
            dtype=dtype, device=device)] 

        self.rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
            dtype=dtype, device=device) for _ in range(num_layers - 1)]


    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs:
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.

        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """

        seq_len, bs, input_size = X.shape
        assert input_size == self.input_size

        # Init hiddens from h0
        if h0 is None:
            hiddens = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
        else:
            assert h0.shape == (self.num_layers, bs, self.hidden_size)
            hiddens = list(ops.split(h0, 0)) # chuyển thành list để assign elems được

        # Init outputs from X
        outputs = list(ops.split(X, 0)) # chuyển thành list để assign elems được

        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]
            for i in range(seq_len):
                x = rnn_cell(outputs[i], hiddens[layer])
                hiddens[layer] = x
                outputs[i] = x

        assert len(outputs) == seq_len
        assert len(hiddens) == self.num_layers

        return ops.stack(outputs, 0), ops.stack(hiddens, 0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        self.bias = bias
        x = 1 / hidden_size**0.5

        self.W_ih = Parameter(init.rand( input_size, 4*hidden_size, low=-x, high=x), device=device)
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-x, high=x), device=device)

        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-x, high=x), device=device)
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-x, high=x), device=device)


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch
.       c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]

        if h is None: 
            h0 = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)
            c0 = init.zeros(bs, self.hidden_size, dtype=self.dtype, device=self.device)
        else:
            h0, c0 = h

        out = X @ self.W_ih + h0 @ self.W_hh

        if self.bias:
            out += self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)
            out += self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)

        _sigmoid = Sigmoid()
        i, f, g, o = ops.split(out, 1, chunks=4)
        i, f, g, o = _sigmoid(i), _sigmoid(f), ops.tanh(g), _sigmoid(o)

        c = f*c0 + i*g
        h = o*ops.tanh(c)

        return h, c



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.bias = bias

        # Init LSTM Layers
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, dtype=dtype, device=device)]

        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device) 
            for _ in range(num_layers - 1)]


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        assert input_size == self.input_size

        # Init hiddens_h,c from h
        if h is None:
            hiddens_h = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
            hiddens_c = [init.zeros(bs, self.hidden_size, device=self.device)] * self.num_layers
        else:
            h0, c0 = h
            assert h0.shape == (self.num_layers, bs, self.hidden_size)
            assert c0.shape == (self.num_layers, bs, self.hidden_size)
            hiddens_h = list(ops.split(h0, 0))
            hiddens_c = list(ops.split(c0, 0))

        # Init outputs from X
        outputs = list(ops.split(X, 0)) # chuyển thành list để assign elems được

        for layer in range(self.num_layers):
            lstm_cell = self.lstm_cells[layer]
            for i in range(seq_len):
                h, c = lstm_cell(outputs[i], (hiddens_h[layer], hiddens_c[layer]))
                hiddens_h[layer] = h
                hiddens_c[layer] = c
                outputs[i] = h

        assert len(outputs) == seq_len
        assert len(hiddens_h) == self.num_layers
        assert len(hiddens_c) == self.num_layers
        assert outputs[-1] == hiddens_h[-1]

        return ops.stack(outputs, 0), (ops.stack(hiddens_h, 0), ops.stack(hiddens_c, 0))



class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.weight = Parameter(init.rand(num_embeddings, embedding_dim, low=0, high=1, device=device))
        self.eye = np.eye(num_embeddings, dtype=dtype)

    def to_one_hot(self, x: Tensor) -> Tensor:
        return Tensor(self.eye[x.numpy().astype("int")], device=x.device)
        #      x of shape (seq_len, bs) convert to one-hot vectors
        # return of shape (seq_len, bs, num_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embed_dim)
        """
        y = self.to_one_hot(x)
        y.requires_grad = False
        return y @ self.weight

# - - - - - - - - - - - - -
# Transformer

class SoftMax(Module):
    def forward(self, z: Tensor):
        x = ops.exp(z)
        return x / ops.summation(x, axes=-1, keepdims=True)


class SelfAttention(Module):
    def __init__(self, input_size: int, device=None):
        super.__init__()
        # x = (1 / input_size)**0.5
        # self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-x, high=x), device=device)

        # self.W_KQV = Parameter(init.rand())
        # self.W_out = Parameter(init.)

    def forward(self, x):
        K,Q,V = ops.split(x * W_KQV, 1, chunks=3)
        attn = SoftMax()(K @ Q.transpose() / np.sqrt(x.shape[1]))
        return attn @ V @ W_out, attn

class MultiHeadAttention(Module):
    def forward(self, x):
        skip


class Transformer(Module):
    def forward(self, x):
        skip
