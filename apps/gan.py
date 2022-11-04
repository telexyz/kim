import sys
sys.path.append("..")

import kim
import kim.nn as nn
import kim.init as init
import numpy as np

# Modularizing GAN "Loss"
class GANLoss:
    def __init__(self, model_D, opt_D):
        self.model_D = model_D
        self.opt_D = opt_D
        self.loss_D = nn.SoftmaxLoss()

    def _update_D(self, real_X, fake_X):
        real_Y = self.model_D(real_X)
        fake_Y = self.model_D(fake_X.detach())
        batch_size = real_X.shape[0]
        ones = init.ones(batch_size, dtype="int32")
        zeros = init.zeros(batch_size, dtype="int32")
        loss = self.loss_D(real_Y, ones) + self.loss_D(fake_Y, zeros)
        loss.backward()
        self.opt_D.step()

    def forward(self, fake_X, real_X):
        self._update_D(real_X, fake_X)
        fake_Y = self.model_D(fake_X)
        batch_size = real_X.shape[0]
        ones = init.ones(batch_size, dtype="int32")
        loss = self.loss_D(fake_Y, ones)
        return loss

model_G = nn.Sequential(nn.Linear(2, 2))
opt_G = kim.optim.Adam(model_G.parameters(), lr = 0.01)

model_D = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

opt_D = kim.optim.Adam(model_D.parameters(), lr=0.01)
gan_loss = GANLoss(model_D, opt_D)


def train_gan(data, batch_size, num_epochs):
    assert data.shape[0] % batch_size == 0
    for epoch in range(num_epochs):
        begin = (batch_size * epoch) % data.shape[0]
        X = data[begin: begin+batch_size, :]
        Z = np.random.normal(0, 1, (batch_size, 2))
        X = kim.Tensor(X)
        Z = kim.Tensor(Z)
        fake_X = model_G(Z)
        loss = gan_loss.forward(fake_X, X)
        loss.backward()
        opt_G.step()



from matplotlib import pyplot as plt

A = np.array([[1, 2], [-0.2, 0.5]]).astype("float32")
mu = np.array([2, 1]).astype("float32")
# total number of sample data to generated
num_sample = 3200
data = np.random.normal(0, 1, (num_sample, 2)).astype("float32") @ A + mu

plt.scatter(data[:,0], data[:,1], color="blue", label="real data")
plt.legend()

train_gan(data, 32, 2000)
