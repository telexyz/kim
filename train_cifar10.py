import sys
sys.path.append('./apps')
import kim as ndl
from models import ResNet9
from simple_training import train_cifar10

device = ndl.default_device()
dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,
         # collate_fn=ndl.data.collate_ndarray,
         device=device,
         # dtype="float32",
)
model = ResNet9(device=device, dtype="float32")
train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001)
# python3 train_cifar10.py