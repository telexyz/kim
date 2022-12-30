import logging
logger = logging.getLogger(__name__)

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

class ImagingOHLCV(object):
    """turn OHLC price and volume data into image.
    """
    def __init__(self, resolution=32, price_prop=0.75):
        super(ImagingOHLCV, self).__init__()
        self.resolution = resolution
        self.price_prop = price_prop

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, price_data, volumn_data=None):
        if isinstance(price_data, (pd.DataFrame, )):
            price_data = price_data.values

        if volumn_data is None:
            price_data, volumn_data = price_data[:, :4], price_data[:, 4]

        n = price_data.shape[0]
        X_img = np.zeros((n * 3, self.resolution))

        price_pixels = round(self.resolution * self.price_prop)
        vol_pixesl = self.resolution - price_pixels
        price_data = (price_data - price_data.min()) / (price_data.max() - price_data.min())

        X_loc = (price_data * (price_pixels - 1)).astype(int) + vol_pixesl
        for i in range(n):
            # low-high bar in the middle
            loc_x = (i * 3) + 1
            loc_y_top = X_loc[i, 1]
            loc_y_bottom = X_loc[i, 2]
            X_img[loc_x, loc_y_bottom:loc_y_top+1] = 1
