import kim
import kim.nn as nn
import numpy as np

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    f = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(f),
        nn.ReLU(),
    )
# https://raw.githubusercontent.com/dlsyscourse/hw2/32490e61fbae67d2b77eb48187824ca87ed1a95c/figures/residualblock.png

def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    f = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob)
            for i in range(num_blocks) ],
        nn.Linear(hidden_dim, num_classes),
    )
    return f
# https://raw.githubusercontent.com/dlsyscourse/hw2/32490e61fbae67d2b77eb48187824ca87ed1a95c/figures/mlp_resnet.png

def epoch(dataloader, model, optimizer=None):
    np.random.seed(4)
    loss_func = nn.SoftmaxLoss()

    if optimizer:
        model.train()
        losses = []
        errors = []
        for i, batch in enumerate(dataloader):
            x, y = batch[0], batch[1]
            out = model(x)
            loss = loss_func(out, y)
            error = np.mean(out.numpy().argmax(axis=1) != y.numpy())
            losses.append(loss.numpy())
            errors.append(error)

            optimizer.reset_grad()
            loss.backward()
            optimizer.step()
        result = (np.mean(errors), np.mean(losses))

    else:
        model.eval()
        dataset = dataloader.dataset
        x, y = dataset[range(len(dataset))]
        x = kim.Tensor(x)
        y = kim.Tensor(y)
        out = model(x)
        loss = loss_func(out, y).numpy()
        error = np.mean(out.numpy().argmax(axis=1) != y.numpy())
        result = (error, loss)

    print(result)
    return result

'''
https://forum.dlsyscourse.org/t/q5-how-were-the-average-error-rate-and-the-average-loss-over-all-samples-computed/2295

https://forum.dlsyscourse.org/t/q3-numerical-issue-in-sgd-and-adam/2279/16
'''

def train_mnist(batch_size=100, epochs=10, optimizer=kim.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    test_dataset = kim.data.MNISTDataset(\
            data_dir + "/t10k-images-idx3-ubyte.gz",
            data_dir + "/t10k-labels-idx1-ubyte.gz")
    test_dataloader = kim.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    train_dataset = kim.data.MNISTDataset(\
            data_dir + "/train-images-idx3-ubyte.gz",
            data_dir + "/train-labels-idx1-ubyte.gz")
    train_dataloader = kim.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)

    test_err, test_loss = epoch(test_dataloader, model)

    return (train_err, train_loss, test_err, test_loss)


if __name__ == "__main__":
    train_mnist(data_dir="../data")
