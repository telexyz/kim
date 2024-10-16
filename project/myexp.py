from os.path import exists
import sys; sys.path.insert(0, '..')
from mydat import ImagingOHLCV, OHLCV, DATA_DIR
from tqdm import tqdm
import torch
import kim
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, gen_img=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.bs = batch_size
        self.gen_img = gen_img
        if not shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(self.bs, len(dataset), self.bs))

    @property
    def batch_size(self): return self.bs

    def __iter__(self):
        if self.shuffle:
            a = np.arange(len(self.dataset))
            np.random.shuffle(a)
            self.ordering = np.array_split(
                a, range(self.bs, len(self.dataset), self.bs))
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self.ordering):
            raise StopIteration
        order = self.ordering[self.n]
        self.n += 1
        bx, by = [], []
        for i in order:
            d = self.dataset[i]
            m, y = d[0], d[1]
            if self.gen_img: bx.append(self.dataset.m_to_img(m))
            else: bx.append(m)
            by.append(y)
        return np.array(bx), np.array(by)


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

# Pseudo dropout to copy torch dropout
class PseudoDropout(kim.nn.Module):
    def __init__(self, mask, p=0.5):
        super().__init__()
        self.p = p
        self.mask = mask
        self.mask.requires_grad = False

    def forward(self, x: kim.Tensor) -> kim.Tensor:
        if not self.training: return x
        return (x * self.mask) / self.p

class Sequential(kim.nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def __getitem__(self, i):
        return self.modules[i]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: kim.Tensor) -> kim.Tensor:
        for m in self.modules: x = m(x)
        return x

    def replace_dropout(self, mask):
        for i, x in enumerate(self.modules):
            if isinstance(x, kim.nn.Dropout) or isinstance(x, PseudoDropout):
                self.modules[i] = PseudoDropout(kim.Tensor(mask, requires_grad=False))
                return

kim.nn.Sequential = Sequential

def get_torch_dropout_mask(model, x):
    for layer in model:
        x = layer(x)
        if isinstance(layer, torch.nn.Dropout):
            return x, kim.NDArray(x.detach().cpu().numpy()) != 0

def copy_init_weights_to_torch(model: kim.nn.Sequential, model_: torch.nn.Sequential):
    for i, x in enumerate(model):
        if isinstance(x, kim.nn.Conv):  # i=0; model[i]
            model_[i].weight.data = torch.tensor(
                x.weight.numpy().transpose(3, 2, 0, 1))
            model_[i].bias.data = torch.tensor(x.bias.numpy())
        if isinstance(x, kim.nn.BatchNorm2d):  # i=1; model[i]
            model_[i].weight.data = torch.tensor(x.weight.numpy().reshape((kim.prod(x.weight.shape),)))
            model_[i].bias.data = torch.tensor(x.bias.numpy().reshape((kim.prod(x.bias.shape),)))
        if isinstance(x, kim.nn.Linear):  # i=9; model[i]
            model_[i].weight.data = torch.tensor(x.weight.numpy().transpose())
            model_[i].bias.data = torch.tensor(x.bias.numpy())


kim.nn.NONLINEARITY = "leaky_relu"
def mymodel(nn, dropout=True):
    def conv_block(cin, cout):
        if nn == kim.nn: conv = nn.Conv(cin, cout, (5, 3))
        else: conv = nn.Conv2d(cin, cout, (5, 3), padding=(2, 1))
        return [conv,
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(),
                nn.MaxPool2d((2, 1))]

    m = conv_block(1, 64) + conv_block(64, 128)
    m.append(nn.Flatten())
    if dropout: m.append(nn.Dropout(0.5))
    m.append(nn.Linear(15360, 2))
    return nn.Sequential(*m)


def epoch(dl, model, loss_fn, optimizer, n):
    if len(dl.dataset) == 0: return
    accuracy, losses = 0, 0
    training = (optimizer is not None)
    msg_header = ("[ train ]" if training else "[ valid ]") + f" Epoch: {n}"

    kimmy = isinstance(loss_fn, kim.nn.Module)
    if kimmy: 
        model.train() if training else model.eval()
        dl.gen_img = False
        dl.gen_img = True
    print(">>> images gen by" + ("numpy" if dl.gen_img else "cuda"))
    pbar = tqdm(total=len(dl.dataset) // dl.batch_size)
    for i, (input, target_) in enumerate(dl):
        if kimmy:
            if not dl.gen_img:
                resolution = dl.dataset.imager.resolution
                batch, days, _ = input.shape  # (256, 5, 6) (Batch, Days, data)
                x = kim.NDArray(input); device = x.device
                y = kim.NDArray.make((batch, 1, resolution, 3*days), device=device)
                device.gen_img(x._handle, y._handle, batch, days, resolution)
                # img = y.numpy()[0,0]
                # print(x.numpy()[0], x.shape)
                # print(">>>\n", img, y.shape); assert False
                input = kim.Tensor(y)
            else:
                B, W, H = input.shape
                input = input.swapaxes(1, 2).reshape((B, 1, H, W))
                input = kim.Tensor(input)
            target = kim.Tensor(target_)
            output = model(input)
            loss = loss_fn(output, target)
            ouput_, loss_ = output, loss
        else:
            B, W, H = input.shape
            input = input.swapaxes(1, 2).reshape((B, 1, H, W))
            # print(input.shape) # (B, 1, 32, 15), 1-channel, 32x15 image
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
    return accuracy/(i+1), losses/(i+1)


def compare_losses(epochs):
    dl_train, _, _ = get_train_val_test_dataset(5, 32, 0.75, 1600, 0, 0, 160)
    loss_fn_ = torch.nn.CrossEntropyLoss()
    model_ = mymodel(torch.nn)
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=1e-5)
    loss_fn = kim.nn.SoftmaxLoss()
    model = mymodel(kim.nn); model.train()
    optimizer = kim.optim.Adam(model.parameters(), lr=1e-5)

    # Assign same weights between models
    copy_init_weights_to_torch(model, model_)

    for e in range(0, epochs):
        for i, (input, target) in enumerate(dl_train):
            B, W, H = input.shape
            input = input.swapaxes(1, 2).reshape((B, 1, H, W))

            # Torch first to copy dropout behavior to kim
            input_ = torch.Tensor(input)
            target_ = torch.Tensor(target).long()
            input_, mask = get_torch_dropout_mask(model_, input_)
            model.replace_dropout(mask)
            output_ = model_[-1](input_)  # equivelant to output_ = model_(input_)
            loss_ = loss_fn_(output_, target_)
            loss = loss_fn(model(kim.Tensor(input)), kim.Tensor(target)) # kimmy

            optimizer_.zero_grad(); loss_.backward(); optimizer_.step()
            optimizer.reset_grad(); loss.backward(); optimizer.step()

            loss = loss.numpy()
            loss_ = loss_.detach().cpu().numpy()
            diff = abs(loss - loss_)
        print(f"epoch {e}:", loss, loss_, diff)

def load_model(lib=kim):
    done = 0
    if lib == torch:
        loss_fn = torch.nn.CrossEntropyLoss()
        model = mymodel(torch.nn).cuda()
        if exists("models/torch.chkpt"):
            data = torch.load("models/torch.chkpt")
            done = data['epoch'] + 1
            model.load_state_dict(data['model'])
    else:
        loss_fn = kim.nn.SoftmaxLoss()
        model = mymodel(kim.nn)
        if exists("models/kim.chkpt"):
            data = torch.load("models/kim.chkpt")
            done = data['epoch'] + 1
            for i, param in enumerate(model.parameters()):
                param.data = kim.Tensor(data['params'][i])
    return model, loss_fn, done


def test(dl_test, lib=kim):
    model, loss_fn, _ = load_model(lib=lib)
    epoch(dl_test, model, loss_fn, None, 0)

def train(dl_train, dl_valid, epoches=50, lib=kim):
    model, loss_fn, done = load_model(lib=lib)
    optimizer = lib.optim.Adam(model.parameters(), lr=1e-5)

    if lib == torch:
        # model = torch.compile(model)
        torch.set_float32_matmul_precision('medium')

    for i in range(done, epoches):
        epoch(dl_train, model, loss_fn, optimizer, i)
        epoch(dl_valid, model, loss_fn, None, i)
        if lib == torch:
            torch.save({'epoch': i, 'model': model.state_dict()}, "models/torch.chkpt")
        else:
            torch.save({'epoch': i, 'params': [x.numpy() for x in model.parameters()]}, "models/kim.chkpt")

if __name__ == "__main__":
    # dl_train, dl_valid, dl_test = get_train_val_test_dataset(5, 32, 0.75, 600_000, 100_000, 300_000, 256)
    # train(dl_train, dl_valid, lib=torch)
    kim.timelog.RECORD_TIMESPENT = True
    kim.timelog.RECORD_CUDA_TIMESPENT = True
    dl_train, dl_valid, dl_test = get_train_val_test_dataset(5, 32, 0.75, 10280, 0, 0, 256)
    train(dl_train, dl_valid, lib=torch, epoches=3)
    # compare_losses(5)
    # dl_train, dl_valid, dl_test = get_train_val_test_dataset(5, 32, 0.75, 1, 0, 100*1028, 256+128)
    # test(dl_test, lib=kim)
    # kim.timelog.print_timespents()
