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
        for w in self.params:
            self.u[w] = self.momentum*self.u[w] + (1 - self.momentum)*w.grad.data
            w.data = (1 - self.lr*self.weight_decay)*w.data - self.lr*self.u[w]
            # self.u[w] = self.momentum*self.u[w] + (1 - self.momentum)*(w.grad.data + self.weight_decay*w.data)
            # w.data = w.data - self.lr * self.u[w]
# https://forum.dlsyscourse.org/t/q3-numerical-issue-in-sgd-and-adam/2279


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

        self.u = {}
        self.v = {}
        for w in self.params:
            self.u[w] = kim.Tensor(kim.cpu().zeros(*w.shape))
            self.v[w] = kim.Tensor(kim.cpu().zeros(*w.shape))


    def step(self):
        self.t += 1
        for w in self.params:
            grad = w.grad.data
            self.u[w] = self.beta1*self.u[w] + (1-self.beta1)*grad
            self.v[w] = self.beta2*self.v[w] + (1-self.beta2)*grad*grad

            u_hat = self.u[w] / (1 - pow(self.beta1, self.t))
            v_hat = self.v[w] / (1 - pow(self.beta2, self.t))

            update = u_hat / (np.power(v_hat, 0.5) + self.eps)
            w.data = (1 - self.lr*self.weight_decay)*w.data - self.lr*update
