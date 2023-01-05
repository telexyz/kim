from pathlib import Path
import pandas as pd
import numpy as np

def get_ticker_data(data_dir, ticker):
    dfile = data_dir / f'{ticker}.parquet'
    try: ds = pd.read_pickle(dfile)
    except: ds = pd.read_parquet(dfile)
    ds["ticker"] = ticker
    return ds


""" `XRX.parquet` file:
              adjOpen    adjHigh     adjLow   adjClose  adjVolume
date
1977-01-03  10.580262  10.694948  10.580262  10.603927     160297
1977-01-04  10.603927  10.649437  10.445551  10.467396     135157
1977-01-05  10.467396  10.627592  10.330865  10.398220     180281
1977-01-06  10.398220  10.512906  10.307199  10.376375     159223
1977-01-07  10.376375  10.421885  10.148823  10.330865     135157
...               ...        ...        ...        ...        ...
2022-12-15  15.950000  16.000000  15.460000  15.490000    1490764
2022-12-16  15.410000  15.650000  15.110000  15.190000    2363635
2022-12-19  15.250000  15.260000  13.980000  14.110000    2207664
2022-12-20  14.110000  14.830000  13.920000  14.710000    1974800
2022-12-21  14.910000  14.955000  14.730200  14.850000    1017520
[11594 rows x 5 columns]

OHLCV is short for Open High Low Close Volumn
"""
class OHLCV:
    # First we load all market data from data_dir to memory
    # Each stock equivelant to a ticker
    # A ticker data is stored in a {ticker_name}.parquet file (above table) 
    def __init__(self, data_dir, size, frequency, imager, seed, min_date, max_date):
        self.data_dir = Path(data_dir)
        self.tickers = [f.stem for f in self.data_dir.glob("*.parquet")]
        self.size = size
        self.frequency = frequency
        self.imager = imager
        self.seed = seed
        self.failed_img = 0
        self.min_date = min_date
        self.max_date = max_date
        self.all_market_data_ = None

    def clone(self):
        return OHLCV(self.data_dir, self.size, self.frequency, self.imager, self.seed, self.min_date, self.max_date)

    @property
    def all_market_data(self):
        if self.all_market_data_ is None:
            all_ds = [get_ticker_data(self.data_dir, t) for t in self.tickers]
            all_ds = pd.concat(all_ds)
            all_ds = all_ds[(all_ds['adjVolume'] != 0) & (all_ds.index >= '1993-01-01')]
            all_ds = all_ds[(all_ds.index >= self.min_date) & (all_ds.index < self.max_date)]
            self.all_market_data_ = all_ds
        return self.all_market_data_

    def __repr__(self): return self.__class__.__name__
    def __len__(self): return self.size

    def __getitem__(self, idx):
        self.prev_failed_img = self.failed_img
        data = self._try_load(idx)
        while data is None:
            self._record_load_fail()
            data = self._try_load(idx)
        return data

    def _record_load_fail(self):
        self.failed_img += 1
        if self.failed_img - self.prev_failed_img > 10000:
            raise ValueError("tried to load data 10k times, not enough market data found. is this a bug?")

    def _try_load(self, idx):
        # three consecutive chunks of data - past, current, and future - are extracted from market data.
        # - past, current are used to calculate moving average price.
        # - current is used to get OHLC and volumn bar
        # - future is used to calculate label / profitability
        rng = np.random.default_rng(seed=idx + self.failed_img)
        s = rng.choice(self.all_market_data.shape[0] - self.frequency)
        md_roi = self.all_market_data.iloc[s - self.frequency:s + 2 * self.frequency]

        if len(md_roi) != 3 * self.frequency: return None  # not enough data.
        if md_roi['ticker'].nunique() != 1: return None # not enough data for the sampled (ticker, date)

        meta = {'ticker': md_roi['ticker'][0], 'index': s, 'date': md_roi.index[self.frequency]}

        # current
        price_ohlc = md_roi.iloc[self.frequency:2*self.frequency, :4].values
        volume = md_roi.iloc[self.frequency : 2*self.frequency, 4].values

        # past and current
        ma_close = md_roi[ : 2 * self.frequency]['adjClose'].rolling(self.frequency).mean()
        ma_close = ma_close.tail(self.frequency).values  # reformat

        img = self.imager(price_data=price_ohlc, volumn_data=volume, ma_close=ma_close)
        if img is None: return None

        # future (label)
        X2 = md_roi.iloc[2*self.frequency : 3*self.frequency, :5].values
        # diff between close price of last day and open price of the first day.
        y = X2[-1, 3] - X2[0, 0]

        return img, (y > 0).astype("int"), meta

    def train_val_split(self, train_prop=0.7, seed=None):
        """split randomly to have train and val datasets.
        :param train_prop: proportion of the training dataset.
        """
        train_ds, val_ds = self.clone(), self.clone()

        rng = np.random.default_rng(seed=seed)
        tr_idx = rng.uniform(size=len(self.all_market_data)) < train_prop

        # train
        train_ds.all_market_data_ = self.all_market_data_[tr_idx]
        train_ds.size = int(self.size * train_prop)

        # val
        val_ds.all_market_data_ = self.all_market_data_[~tr_idx] # not in tr_idx
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


def test(ticker):
    print(f"loading stock to `data/%s.png`" % (ticker))
    dfile = f"data/stocks/%s.parquet" % (ticker)
    try: ds = pd.read_pickle(dfile)
    except: ds = pd.read_parquet(dfile)

    imager = ImagingOHLCV(resolution=64)
    X_img = imager.transform(ds.tail(60))
    show_img(X_img, ticker)
    return X_img

def show_img(X_img, ticker):
    img = (X_img * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(f"data/%s.png" % (ticker))
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='Greys_r')
    plt.show()

DATA_DIR = Path("data/data_yfinance").expanduser()

'''
from mydat import *; import random
test("GOOGL"); test("SONY")
imager = ImagingOHLCV(64, price_prop=0.75)
ds = OHLCV(DATA_DIR, size=6000, frequency=60, imager=imager, seed=5, min_date='1993-01-01', max_date='2000-12-31')
X_img, y, meta = random.choice(ds); print(meta); show_img(X_img, meta["ticker"])

X_img, y, meta = ds[1]; print(meta); show_img(X_img, meta["ticker"])
>>> ds[1] -> (X, y, meta)
# Ảnh thực tế sẽ quay 90 độ ngược chiều kim đồng hồ
# Rows are values, Cols are Time (one day = 3 cols)
array([[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0],
day-1  [1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0],
day-2  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0],
day-3  [1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0],
day-4  [1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0],
day-5  [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0],
       [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0]]), 
    0.17513274929999945, # <= label / profitability
    {'ticker': 'AXP', 'index': 89791, 'date': Timestamp('1995-06-08 00:00:00+0000', tz='UTC')}

img = X_img.T # cần transpose để chuyển hàng thành cột và ngược lại
img.shape (H, W) = (32, 15) # 15 vì 3 cột tương ứng với 1 ngày, nên 5 ngày = 15 cột
(32, 15,  1) => Conv(5, 3,  1,  64) => MaxPool(2, 1) => (16, 15,  64)
(16, 15, 64) => Conv(5, 3, 64, 128) => MaxPool(2, 1) => ( 8, 15, 128)
(8 * 15) * 128 channels = 15360 weights
'''