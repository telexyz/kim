import sys; sys.path.insert(0, '..')
from imaging import ImagingOHLCV, view_chart, DataLoader, OHLCV
from tqdm import tqdm
import kim as ndl
import numpy as np
import pandas as pd

from kim.nn import Conv, Sequential, MaxPool2d, Flatten, Linear, SoftmaxLoss, Dropout, LeakyReLU, BatchNorm2d, Module
from kim import ops
from kim import nn
from kim.optim import Adam
from yp import save_pickle, save_yaml, load_pickle, load_yaml
from pathlib import Path

DATA_DIR = Path("data/stocks").expanduser()
import logging
logger = logging.getLogger(__name__)


class LogLoss(Module):
    """Documentation for LogLoss

    """

    def __init__(self, input_is_probability=False):
        self.input_is_probability = input_is_probability

    def forward(self, yhat, y):
        """
        :param yhat: the probabality of being positive
        :param y: true label
        :returns: log loss
        """
        if not self.input_is_probability:
            yhat = ops.sigmoid(yhat)

        logloss = - y * ops.log(yhat) - (1-y) * ops.log(1 - yhat)
        return logloss


def get_train_val_dataset(freq, img_resolution, price_prop, train_size, val_size_fast, val_size, batch_size):
    # img_resolution = 32
    # price_prop = 0.75
    # freq = 5

    imager = ImagingOHLCV(img_resolution, price_prop=price_prop)
    ds = OHLCV(DATA_DIR, size=train_size, frequency=freq,
               imager=imager,
               seed=5,
               min_date='1993-01-01',
               max_date='2000-12-31')

    ds_val_fast = OHLCV(DATA_DIR, size=val_size_fast, frequency=freq,
                        imager=imager,
                        seed=5 + 19,
                        min_date='2001-01-01',
                        max_date='2019-12-31')

    ds_val = OHLCV(DATA_DIR, size=val_size, frequency=freq,
                   imager=imager,
                   seed=5 + 19,
                   min_date='2001-01-01',
                   max_date='2019-12-31')

    dl = DataLoader(ds, batch_size, True)
    dl_val_fast = DataLoader(ds_val_fast, batch_size, False)
    dl_val = DataLoader(ds_val, batch_size, False)

    return dl, dl_val_fast, dl_val


def conv_block(cin, cout, device):
    m = [Conv(cin, cout, (5, 3), device=device),
         BatchNorm2d(cout, device=device),
         LeakyReLU(),
         MaxPool2d()]
    return m


def output_layer(hidden_size, num_classes, device, p=0.5):
    m = [Flatten(),
         Dropout(p),
         Linear(hidden_size, num_classes, device=device)]
    return m


def get_model(name: str, device='cuda') -> nn.Module:
    if device == 'cuda':
        device = ndl.cuda()
    else:
        device = ndl.cpu()

    if name == '32x15':
        m = []
        m += conv_block(1, 64, device=device)
        m += conv_block(64, 128, device=device)
        m += output_layer(15360, 2, device=device)
        model = Sequential(*m)
    else:
        raise NotImplementedError("...")
    return model


def epoch(dl, model, loss_fn, opt=None, device=None, msg_header=''):
    # np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()

    device = model.parameters()[0].device
    pbar = tqdm(total=len(dl.dataset) // dl.batch_size)
    losses = []

    cnt = 0
    pred = []   # probabality of being positive return.

    for X, y in dl:
        # (N, W, H) -> (N, H, W) -> (N, 1, H, W). the input to nn.Conv
        # if any of the X is not finite number, stop.
        if not np.isfinite(X.numpy()).all():
            save_pickle([X.numpy(), y.numpy()], 'tmp.pkl')
            0/a

        n, w, h = X.shape
        X = X.transpose((1, 2)).reshape((n, 1, h, w))

        X = ndl.Tensor(X, device=device)
        y01 = ndl.Tensor(y.numpy() > 1, device=device)
        yhat = model(X)

        loss = loss_fn(yhat, y01)
        losses.append(loss.numpy())
        pred.append(yhat.numpy()[:, 1])

        pbar.set_description_str(
            msg_header + f"Batch: {cnt} Loss: {np.mean(losses):.6f}")
        pbar.update()
        cnt += 1

        if model.training:
            opt.reset_grad()
            loss.backward()
            opt.step()

        # to save memory.
        del loss, yhat, X, y01

    out = {'loss': float(np.mean(losses)),
           'prediction': np.concatenate(pred)}
    return out


def save_model(model, filename):
    print(">>> save model params", len(model.parameters()))
    param_np = [x.numpy() for x in model.parameters()]
    save_pickle(param_np, filename)


def load_model(model, filename):
    param_np = load_pickle(filename)
    assert len(param_np) == len(model.parameters()), "%s != %s" % (len(param_np), len(model.parameters()))
    for i, param in enumerate(model.parameters()):
        param.data = ndl.Tensor(param_np[i], device=param.device)
    return model


def train(model, dl_set, output_folder, lr, weight_decay, checkpoint):
    loss_fn = SoftmaxLoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dl, dl_val = dl_set
    output_folder = Path(output_folder)

    losses = {'train_loss': [], 'val_loss': []}

    print(">>> init model params", len(model.parameters()))

    for i in checkpoint.epoches_:
        logger.info(f"Train and eval for epoch {epoch}")
        train_res = epoch(dl, model, loss_fn, opt, msg_header=f"Train - Epoch: {i} ")
        val_res = epoch(dl_val, model, loss_fn, opt=None, msg_header=f"valid - Epoch: {i} ")
        train_loss = train_res['loss']
        val_loss = val_res['loss']
        losses['train_loss'].append(train_loss)
        losses['val_loss'].append(val_loss)

        # note prediction for the trainign dataset is discarded.
        # val prediction is also not saved. because it is a small dataset here.
        checkpoint.save(i, {'model': model, 'losses': {
                        'train_loss': train_loss, 'train_val_loss': val_loss}})


def predict(model, dl_val, checkpoint, predict_step):
    loss_fn = SoftmaxLoss()

    # losses = []
    for i in range(1, checkpoint.max_epoch, predict_step):
        chkpt_file = checkpoint.checkpoint_dir / f'val_predict_{i}.pkl'
        if not chkpt_file.exists():
            logger.info("run prediction for epoch %s", epoch)
            model = checkpoint.load_model_param_from_checkpoint(model, i)
            val_res = epoch(dl_val, model, loss_fn, opt=None, msg_header=f"valid - Epoch: {i} ")
            checkpoint.save(i, {'val_predict': val_res['prediction'], 'losses': {'val_loss': val_res['loss']}})
        # auxlirary
        # save_yaml(losses, output_folder / 'losses.yaml')

class Checkpoint(object):
    """Documentation for Checkpoint

    """
    def __init__(self, checkpoint_dir, training, max_epoch):
        """FIXME: briefly describe function

        :param checkpoint_dir: 
        :param training: if is training, primary file is model_{epoch}.pkl, otherwise, val_predict_{epoch}.pkl.
        :param max_epoch:
        :returns: 

        """
        self.checkpoint_dir = Path(checkpoint_dir) / \
            'chkpt'  # to avoid clutter model directory.
        self.max_epoch = max_epoch + 1
        self.training = training

        self.epoches_ = range(self.last_checkpoint+1, self.max_epoch)

    def is_checkpoint(self, epoch):
        return True

    def save(self, epoch, data):
        if not self.is_checkpoint(epoch):
            return
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        for data_name, data_val in data.items():
            if data_name == 'model':
                save_model(data_val, self.checkpoint_dir /
                           f'{data_name}_{epoch}.pkl')
            elif data_name == 'losses':
                # note, losses.csv are used in the root folder, not
                # the sub 'chkpt' folder, because there's one
                # losses.csv per model, others have many.
                losses_file = self.checkpoint_dir.parent / 'losses.csv'
                if losses_file.exists():
                    losses = pd.read_csv(losses_file).set_index('epoch')
                    for loss_name, loss in data_val.items():
                        losses.loc[epoch, loss_name] = loss
                else:
                    losses = pd.DataFrame(
                        data_val, index=pd.Index([epoch], name='epoch'))
                losses.to_csv(losses_file)
            elif isinstance(data_val, (np.ndarray, )):
                save_pickle(data_val, self.checkpoint_dir /
                            f'{data_name}_{epoch}.pkl')
            else:
                save_yaml(data_val, self.checkpoint_dir /
                          f'{data_name}_{epoch}.yaml')

    def load_model_param_from_checkpoint(self, model, epoch=None):
        if epoch is None:
            epoch = self.last_checkpoint
        model = load_model(model, self.checkpoint_dir / f'model_{epoch}.pkl')
        return model

    def __iter__(self):
        return list(range(self.last_checkpoint, self.max_epoch))

    @property
    def last_checkpoint(self):
        file_id = 'model' if self.training else 'val_predict'
        fs = self.checkpoint_dir.glob(f"{file_id}_*.pkl")
        epoch = [int(f.stem.replace(f'{file_id}_', '')) for f in fs]
        if len(epoch) == 0:
            return 0
        return max(epoch)


def main(config_id, task, device, max_epoch, predict_step=1, output_folder=None):
    config = load_yaml(Path("configs") / f'{config_id}.yaml')
    if output_folder is None:
        output_folder = Path("models") / config_id
        output_folder.mkdir(exist_ok=True, parents=True)
    else:
        output_folder = Path(output_folder)
    setup_simple_logging(output_folder/'train.log')
    logger.info("output folder is set to %s", output_folder)
    chkpt = Checkpoint(output_folder, task == 'train', max_epoch)
    model = get_model(config['model']['name'], device)
    if chkpt.last_checkpoint != 0:
        logger.info("Load model parameters")
        model = chkpt.load_model_param_from_checkpoint(model)

    dp = config['data']
    dl, dl_val_fast, dl_val = get_train_val_dataset(freq=dp['trading_days'],
                                                    img_resolution=dp['image_resolution'],
                                                    price_prop=dp['price_proportion'],
                                                    train_size=dp['num_images_train'],
                                                    val_size_fast=dp['num_images_valid_fast'],
                                                    val_size=dp['num_images_valid'],
                                                    batch_size=dp['batch_size'])

    # if use_checkpoint:
    #     load_model(output_folder / 'model.chkpt'
    lr, wd = config['optimiser']['lr'], config['optimiser']['wd']

    if task == 'train':
        train(model, [dl, dl_val_fast], output_folder=output_folder,
              lr=lr, weight_decay=wd, checkpoint=chkpt)
    elif task == 'predict_full_val':
        predict(model, dl_val, checkpoint=chkpt, predict_step=predict_step)
    else:
        raise ValueError(f"unknown action {task}")


if __name__ == "__main__":
    from yp import setup_simple_logging
    import fire
    fire.Fire(main)
