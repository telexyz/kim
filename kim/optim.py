"""Optimization module"""
import kim
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}
        for w in self.params:
            self.u[w] = kim.Tensor(kim.cpu().zeros(*w.shape))

    def step(self):
        # https://stats.stackexchange.com/questions/259752/sgd-l2-penalty-weights
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            l2 = (2 * self.weight_decay) * w.data
            beta = self.momentum # update momentum of w
            self.u[w] = beta * self.u[w] + (1 - beta) * w.grad.data
            w.data = w.data + ( -self.lr ) * (self.u[w] + l2)
            # w.data = w.data + (-self.lr ) * w.grad.data # vanilla
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
