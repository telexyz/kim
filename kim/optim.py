"""Optimization module"""
import kim
import numpy as np


class Optimizer:
    def __init__(self, params):
        # Ghi nhớ lại những params cần được update
        self.params = [x for x in params if x.update_params]
        print(">>> optim params", len(self.params))

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, device=None):
        super().__init__(params)
        if device is None: device = kim.default_device()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}
        for w in self.params:
            # Với mỗi param (weight w) cần khởi tạo chỉ số phụ u bằng chính kích thước của w => x2 vram
            self.u[w] = device.zeros(*w.shape)

    def step(self):
        for w in self.params:
            grad = w.grad.cached_data + w.cached_data * self.weight_decay
            self.u[w] = self.momentum*self.u[w] + (1 - self.momentum)*grad
            w.cached_data -= self.lr * self.u[w]
            # u của w được update sau mỗi step, và cần được lưu lại, nếu offload thì phải offload cả u

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        device=None,
    ):
        super().__init__(params)
        if device is None: device = kim.default_device()
        self.lr = float(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.iter = 0

        self.u = {}
        self.v = {}
        for w in self.params:
            self.u[w] = device.zeros(*w.shape)
            self.v[w] = device.zeros(*w.shape)


    def step(self):
        self.iter += 1
        for w in self.params:
            if w.grad is None: continue
            # weight_decay = 0 thì grad = w.grad
            grad = w.grad.cached_data + w.cached_data * self.weight_decay

            self.u[w] = self.beta1*self.u[w] + (1-self.beta1)*grad
            self.v[w] = self.beta2*self.v[w] + (1-self.beta2)*(grad**2)

            u_hat = self.u[w] / (1 - (self.beta1 ** self.iter))
            v_hat = self.v[w] / (1 - (self.beta2 ** self.iter))

            update = u_hat / ((v_hat ** 0.5) + self.eps)
            w.cached_data -= self.lr * update
