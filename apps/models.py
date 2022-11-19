import sys
sys.path.append('./python')
import kim as kim
import kim.nn as nn
import math
import numpy as np
np.random.seed(0)


def ConvBN(in_channels, out_channels, kernel_size, stride, dtype="float32", device=None):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size, stride=stride, dtype=dtype,device=device),
        nn.BatchNorm2d(out_channels, dtype=dtype,device=device),
        nn.ReLU(),
    )

class ResNet9(kim.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.model = nn.Sequential(
            # 
            ConvBN(3,16,7,4, dtype=dtype,device=device),
            ConvBN(16,32,3,2, dtype=dtype,device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32,32,3,1, dtype=dtype,device=device),
                    ConvBN(32,32,3,1, dtype=dtype,device=device),
                ),
            ),
            # 
            ConvBN(32,64,3,2, dtype=dtype,device=device),
            ConvBN(64,128,3,2, dtype=dtype,device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128,128,3,1, dtype=dtype,device=device),
                    ConvBN(128,128,3,1, dtype=dtype,device=device),
                ),
            ),
            # 
            nn.Linear(128,128, dtype=dtype,device=device),
            nn.ReLU(),
            nn.Linear(128,10, dtype=dtype,device=device),
        )


    def forward(self, x):
        print(">>> ResNet9 forward:", x.shape)
        return self.model(x)


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = kim.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = kim.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = kim.data.DataLoader(cifar10_train_dataset, 128, kim.cpu(), dtype="float32")
    print(dataset[1][0].shape)