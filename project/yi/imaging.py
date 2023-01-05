import logging
from copy import deepcopy
from pathlib import Path
from kim.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')


logger = logging.getLogger(__name__)


class MarketData(Dataset):
    """Documentation for  MarketDataMarketData

    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def __getitem__(self, ticker):
        dfile = self.data_dir / f'{ticker}.parquet'
        try:
            ds = pd.read_pickle(dfile)
        except:
            ds = pd.read_parquet(dfile)
        return ds

    def __len__(self):
        pass

    def list_tickers(self):
        fs = self.data_dir.glob("*.parquet")
        tickers = [f.stem for f in fs]
        return tickers


class OHLCV(Dataset):
    """Documentation for OHLCV

    """
    all_market_data: pd.DataFrame

    def __init__(self, data_dir, size, frequency, imager, seed, min_date, max_date):
        """

        :param data_dir: 
        :param size: 
        :param frequency: 
        :param imager: 
        :param seed: 
        :param min_date: 
        :param max_date: 
        :returns: 

        """
        self.market_data = MarketData(data_dir)
        self.tickers = self.market_data.list_tickers()
        self.size = size
        self.frequency = frequency
        self.imager = imager
        self.seed = seed
        self.failed_img = 0
        self.min_date = min_date
        self.max_date = max_date

        self._init()

    def __repr__(self):
        # if i don't have this line, print(ds) or type ds in console,
        # it would loop though all ds. i don't know why.
        return self.__class__.__name__

    def _init(self):
        out = []

        for ticker in self.market_data.list_tickers():
            out_ = self.market_data[ticker]
            out_['ticker'] = ticker
            out.append(out_)

        all_ds = pd.concat(out)

        # process market data.
        all_ds = all_ds[all_ds['adjVolume'] != 0]
        all_ds = all_ds[all_ds.index >= '1993-01-01']
        all_ds = all_ds[(all_ds.index >= self.min_date) &
                        (all_ds.index < self.max_date)]

        self.all_market_data = all_ds

    def __getitem__(self, idx):
        cnt = 0
        data = None
        while data is None:
            data = self._try_load(idx)
            if cnt > 10000:
                raise ValueError(
                    "tried to load data 100 times, not enough market data found. is this a bug?")
            cnt += 1
            self.failed_img += 1
        return data

    def _try_load(self, idx):
        # three consecutive chunks of data - past, current, and future - are extracted from market data.
        # past, current are used to calculate moving avearge price.
        # current is used to get OHLC and volumn bar
        # future is used to calcualte label/profitability
        rng = np.random.default_rng(seed=idx + self.failed_img)

        s = rng.choice(self.all_market_data.shape[0] - self.frequency)

        md_roi = self.all_market_data.iloc[s -
                                           self.frequency:s + 2 * self.frequency]
        if len(md_roi) != 3 * self.frequency:
            # not enough data.
            return None
        if md_roi['ticker'].nunique() != 1:
            # not enough data for the sampled (ticker, date)
            return None
        meta = {'ticker': md_roi['ticker'][0],
                'index': s,
                'date': md_roi.index[self.frequency]}

        # current
        price_ohlc = md_roi.iloc[self.frequency:2*self.frequency, :4].values
        volume = md_roi.iloc[self.frequency:2*self.frequency, 4].values
        # past and current
        ma_close = md_roi[:2 *
                          self.frequency]['adjClose'].rolling(self.frequency).mean()
        ma_close = ma_close.tail(self.frequency).values  # reformat

        img = self.imager(price_data=price_ohlc,
                          volumn_data=volume, ma_close=ma_close)

        if img is None:
            return None

        # future (label)
        X2 = md_roi.iloc[2*self.frequency:3*self.frequency, :5].values
        # diff between close price of last day and open price of the first day.
        y = X2[-1, 3] - X2[0, 0]

        return img, y, meta

    def __len__(self):
        return self.size

    def train_val_split(self, train_prop=0.7, seed=None):
        """split randomly to have train and val datasets.

        :param train_prop: proportion of the training dataset.

        when calling, there are 2-3 copies of all_market_data. might give OOM.
        """
        train_ds = deepcopy(self)
        val_ds = deepcopy(self)

        rng = np.random.default_rng(seed=seed)
        tr_idx = rng.uniform(size=len(self.all_market_data)) < train_prop

        # train
        train_ds.all_market_data = self.all_market_data[tr_idx]
        train_ds.size = int(self.size * train_prop)

        # val
        val_ds.all_market_data = self.all_market_data[~tr_idx]
        val_ds.size = self.size - train_ds.size

        return train_ds, val_ds


class ImagingOHLCV(object):
    """turn OHLC price and volume data into image.
    """

    def __init__(self, resolution=32, price_prop=0.75):
        super(ImagingOHLCV, self).__init__()
        self.resolution = resolution
        self.price_prop = price_prop

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, price_data, volumn_data=None, ma_close=None):
        if isinstance(price_data, (pd.DataFrame, )):
            price_data = price_data.values

        if volumn_data is None:
            price_data, volumn_data = price_data[:, :4], price_data[:, 4]

        unique_price = np.unique(price_data)
        if len(unique_price) == 1:
            # same price in this window, probably data errors. not longer going further.
            return None

        n = price_data.shape[0]
        X_img = np.zeros((n * 3, self.resolution))

        price_pixels = round(self.resolution * self.price_prop)
        vol_pixesl = self.resolution - price_pixels

        if ma_close is not None:
            price_data = np.hstack((price_data, ma_close.reshape((-1, 1))))

        price_min, price_max = price_data.min(), price_data.max()
        price_data = (price_data - price_min) / (price_max - price_min)
        X_loc = (price_data * (price_pixels - 1)).astype(int) + vol_pixesl

        for i in range(n):
            # low-high bar in the middle
            loc_x = (i * 3) + 1
            loc_y_top = X_loc[i, 1]
            loc_y_bottom = X_loc[i, 2]
            X_img[loc_x, loc_y_bottom:loc_y_top+1] = 1

            # open bar on the left
            loc_x = (i * 3) + 0
            loc_y = X_loc[i, 0]
            X_img[loc_x, loc_y] = 1

            # close bar on the right
            loc_x = (i * 3) + 2
            loc_y = X_loc[i, 3]
            X_img[loc_x, loc_y] = 1

        # for additional pricing data - flat line for each day.
        for j in range(4, price_data.shape[1]):
            X_loc_additional = X_loc[:, j]
            for i in range(n):
                X_img[i*3:(i+1)*3, X_loc_additional[i]] = 1

        # add volume
        X = volumn_data
        # index start from 0, so -1.
        vol_per_pixel = X.max() / (vol_pixesl - 1)
        X2 = (X / vol_per_pixel).astype(int)
        for i in range(n):
            loc_x = (i * 3) + 1
            loc_y = X2[i]
            X_img[loc_x, 0:(loc_y+1)] = 1

        return X_img


def view_chart(X_img, **kwargs):
    # this to make it looks right... still don't understand why.
    img = np.flip(X_img.T, 0)
    img = (img * 255).astype(np.uint8)
    ax = plt.imshow(img, cmap='Greys_r')
    return ax
    # pil_image = Image.fromarray(img)
    # pil_image.show(**kwargs)
    # plt.imshow(np.flip(X_img.T, 0))


def test(logfile):

    logging.basicConfig(
     filename=logfile,
     level=logging.INFO, 
     format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s',
     # datefmt='%H:%M:%S'
    )
    logger.info("loading google data")
    ds = pd.read_parquet("data/GOOGL.parquet")

    X = ds.tail(5).iloc[:, :4].values

    X = (X - X.min()) / (X.max() - X.min())

    IMG_RESOLUTION = 64

    price_pixels = round(IMG_RESOLUTION * 0.75)
    vol_pixesl = IMG_RESOLUTION - price_pixels

    imager = ImagingOHLCV(64)
    img = imager.transform(ds.tail(60))
    logger.info("End!")
    
    
if __name__ == "__main__":
    # from yp.utils.logger import setup_simple_logging
    # setup_simple_logging(file_only=False)
    import fire
    fire.Fire(test)
