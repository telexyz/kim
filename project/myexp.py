import sys; sys.path.insert(0, '..')
from mydat import ImagingOHLCV, OHLCV, DATA_DIR, DataLoader
from tqdm import tqdm
import torch
import kim; kim.nn.Conv2d = kim.nn.Conv
import numpy as np


def get_train_val_test_dataset(freq, img_resolution, price_prop, train_size, val_size, test_size, batch_size):
    imager = ImagingOHLCV(img_resolution, price_prop=price_prop)
    size = train_size + val_size
    ds = OHLCV(DATA_DIR, size=size, frequency=freq, imager=imager,
               seed=5,
               min_date='1993-01-01',
               max_date='2000-12-31')
    ds_train, ds_val = ds.train_val_split(train_prop=train_size / size, seed=27)

    ds_test = OHLCV(DATA_DIR, size=test_size, frequency=freq,
                    imager=imager,
                    seed=5 + 19,
                    min_date='2001-01-01',
                    max_date='2019-12-31')

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_val, dl_test


def mymodel(nn):
    def conv_block(cin, cout):
        return [nn.Conv2d(cin, cout, (5, 3), padding=(2, 1)),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(),
                nn.MaxPool2d((2, 1))]

    def output_layer(hidden_size, num_classes, p=0.5):
        return [nn.Flatten(),
                nn.Dropout(p),
                nn.Linear(hidden_size, num_classes)]

    m = conv_block(1, 64) + conv_block(64, 128)
    m += output_layer(15360, 2)
    return nn.Sequential(*m)


def epoch(dl, model, loss_fn, optimizer, n):
    pbar = tqdm(total=len(dl.dataset) // dl.batch_size)
    accuracy, losses = 0, 0
    training = (optimizer is not None)
    msg_header = ("[ train ]" if training else "[ valid ]") + f" Epoch: {n}"
    kimmy = isinstance(optimizer, kim.optim.Optimizer)
    if kimmy: model.train() if training else model.eval()
    for i, (input, target_) in enumerate(dl):
        B,W,H = input.shape
        input = input.swapaxes(1, 2).reshape((B,1,H,W))
        # print(input.shape) # (B, 1, 32, 15), 1-channel, 32x15 image
        if kimmy:
            input = kim.Tensor(input)
            target = kim.Tensor(target_)
            output = model(input)
            loss = loss_fn(output, target)
            ouput_, loss_ = output, loss
        else:
            input = torch.Tensor(input).cuda()
            target = torch.Tensor(target_).long().cuda()
            output = model(input)
            loss = loss_fn(output, target)
            ouput_, loss_ = output.detach().cpu(), loss.detach().cpu()

        accuracy += 100 * np.sum(np.argmax(ouput_.numpy(), axis=1) == target_) / dl.batch_size
        losses += loss_.numpy()

        if training:
            optimizer.zero_grad() if not kimmy else optimizer.reset_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description_str(f"{msg_header} Batch: {i} Acc: {accuracy/(i+1):.1f}% Loss: {losses/(i+1):.4f}")
        pbar.update()


def train(dl_train, dl_valid, lib=kim):
    if lib == torch:
        model = mymodel(torch.nn).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        model = mymodel(kim.nn)
        loss_fn = kim.nn.SoftmaxLoss()
    optimizer = lib.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(0, 22):
        train_loss = epoch(dl_train, model, loss_fn, optimizer, i)
        if i % 2 == 1: valid_loss = epoch(dl_valid, model, loss_fn, None, i)

if __name__ == "__main__":
    dl_train, dl_valid, dl_test = get_train_val_test_dataset(5, 32, 0.75, 95_000, 5_000, 1000, 160)
    train(dl_train, dl_valid, lib=kim)

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