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


# https://forum.dlsyscourse.org/t/q5-how-were-the-average-error-rate-and-the-average-loss-over-all-samples-computed/2295

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    raise NotImplementedError()


def train_mnist(batch_size=100, epochs=10, optimizer=kim.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    raise NotImplementedError()


if __name__ == "__main__":
    train_mnist(data_dir="../data")
