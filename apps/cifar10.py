#!/home/t/anaconda3/envs/kim/bin/python3

import sys
sys.path.append('..')
import kim

from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10


device = kim.default_device()
model = ResNet9(device=device, dtype="float32")


dataset = kim.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
dataloader = kim.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,
         device=device,
)
train_cifar10(model, dataloader, n_epochs=3, optimizer=kim.optim.Adam,
      lr=0.001, weight_decay=0.001)

test_dataset = kim.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=False)
test_dataloader = kim.data.DataLoader(\
         dataset=test_dataset,
         batch_size=128,
         shuffle=False,
         device=device,
)
evaluate_cifar10(model, test_dataloader)

# python3 train_cifar10.py