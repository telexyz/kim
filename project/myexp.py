import sys; sys.path.insert(0, '..')
from kim.data import DataLoader
from mydat import ImagingOHLCV, OHLCV, DATA_DIR
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
# from yp import save_yaml, load_yaml

def get_train_val_test_dataset(freq, img_resolution, price_prop, train_size, val_size, test_size, batch_size):
    imager = ImagingOHLCV(img_resolution, price_prop=price_prop)
    ds = OHLCV(DATA_DIR, size=train_size + val_size, frequency=freq,
               imager=imager,
               seed=5,
               min_date='1993-01-01',
               max_date='2000-12-31')
    ds_train, ds_val = ds.train_val_split(train_prop=0.7, seed=27)

    ds_test = OHLCV(DATA_DIR, size=test_size, frequency=freq,
                    imager=imager,
                    seed=5 + 19,
                    min_date='2001-01-01',
                    max_date='2019-12-31')

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_val, dl_test


def mymodel():
    def conv_block(cin, cout):
        return [nn.Conv2d(cin, cout, (5, 3), padding=(2, 1)),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(),
                nn.MaxPool2d((2, 1))]

    def output_layer(hidden_size, num_classes, p=0.5):
        return [nn.Flatten(),
                nn.Dropout(p),
                nn.Linear(15360, num_classes)]

    m = conv_block(1, 64) + conv_block(64, 128)
    m += output_layer(15360, 2)
    return nn.Sequential(*m)


def epoch(dl, model, loss_fn, optimizer):
    pbar = tqdm(total=len(dl.dataset) // dl.batch_size)
    losses = []
    training = (optimizer is not None)
    msg_header = "train" if training else "valid"
    for i, (input, target) in enumerate(dl):
        # print(input.shape) # (64, 15, 32)
        new_shape = tuple(list(input.shape) + [1])
        input = torch.Tensor(input.numpy().reshape(new_shape).swapaxes(1,3)).cuda()
        target = (target.numpy() > 0).astype("long")  # > 0 mean profitable
        target = torch.Tensor(target).long().cuda()
        # print(target)
        output = model(input)
        loss = loss_fn(output, target)
        losses.append(loss.detach().cpu().numpy())
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description_str(f"{msg_header} Batch: {i} Loss: {np.mean(losses):.6f}")
        pbar.update()
    return float(np.mean(losses))


def train(dl_train, dl_valid):
    model = mymodel().cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(0, 30):
        train_loss = epoch(dl_train, model, loss_fn, optimizer)
        if i % 2 == 1: valid_loss = epoch(dl_valid, model, loss_fn, None)

if __name__ == "__main__":
    dl_train, dl_valid, dl_test = get_train_val_test_dataset(5, 32, 0.75, 20_000, 1000, 1000, 128)
    train(dl_train, dl_valid)

''' 
>>> KIM_DEVICE=cpu python3 myexp.py
configs/SO5.yaml
data:
  trading_days: 5
  image_resolution: 32
  price_proportion: 0.75
  batch_size: 160
  num_images_train: 20_000
  num_images_valid: 1_000
  num_images_test: 20_000
model:
  name: 32x15
optimiser:
  lr: 1e-5
  wd: 0

from myexp import *
mymodel()
dl_train, dl_val, dl_test = get_train_val_test_dataset(5, 32, 0.75, 5000, 1000, 1000, 64)
'''