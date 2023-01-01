"""short hand function for loading/saving pickle, feather, and yaml format.
"""
import yaml
import os
import pickle
import platform
import pandas as pd
import logging
import logging.handlers

logger = logging.getLogger(__name__)


def load_pickle(f):
    # if f.endswith('.ft'):
    #     return load_feather(f)
    logger.debug(f"reading {f}")
    if platform.system() == 'Darwin':
        # # read
        # n_bytes = 2 ** 31
        max_bytes = 2**31 - 1
        bytes_in = bytearray(0)
        input_size = os.path.getsize(f)
        with open(f, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        return pickle.loads(bytes_in)
    else:
        return pickle.load(open(f, 'rb'))


def save_pickle(X, f):
    if platform.system() == 'Darwin':
        # pickle.dump(X, open(f, 'wb'), protocol=4)
        n_bytes = 2**31
        max_bytes = 2**31 - 1
        # # write
        bytes_out = pickle.dumps(X)
        with open(f, 'wb') as f_out:
            for idx in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])
    else:
        pickle.dump(X, open(f, 'wb'), protocol=4)

def save_yaml(X, f):
    yaml.dump(X, open(f, 'w'), indent=4, default_flow_style=False, sort_keys=False)

def load_yaml(f):
    return yaml.safe_load(open(f))


def to_org_table(df, floatfmt='.2f'):
    from tabulate import tabulate
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda x: '_'.join(x))
    df = df.reset_index()
    res = tabulate(df, headers=df.columns, tablefmt='orgtbl',
                   showindex=False,
                   floatfmt=floatfmt)
    print(res)
    return res

def setup_simple_logging(log_file_path=None, file_only=True):
    if file_only:
        handlers = []
    else:
        handlers = [logging.StreamHandler()]
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path))

    logging.basicConfig(
        format="%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=handlers,
        level=logging.INFO)
