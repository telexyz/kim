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
            self.u[w] = kim.Tensor(kim.default_device().zeros(*w.shape))

    def step(self):
        for w in self.params:
            grad = (w.grad + w * self.weight_decay).detach()
            self.u[w].data = (self.momentum*self.u[w] + (1 - self.momentum)*grad).detach()
            w.data = (w - self.lr * self.u[w]).detach()

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
            self.u[w] = kim.Tensor(kim.default_device().zeros(*w.shape))
            self.v[w] = kim.Tensor(kim.default_device().zeros(*w.shape))


    def step(self):
        self.t += 1
        for w in self.params:
            if w.grad is None: continue
            grad = w.grad.data + w.data * self.weight_decay
            self.u[w].data = self.beta1*self.u[w].data + (1-self.beta1)*grad
            self.v[w].data = self.beta2*self.v[w].data + (1-self.beta2)*(grad**2)

            u_hat = self.u[w].data / (1 - (self.beta1 ** self.t))
            v_hat = self.v[w].data / (1 - (self.beta2 ** self.t))

            update = u_hat / ((v_hat ** 0.5) + self.eps)
            w.data = w.data - self.lr * update.data
