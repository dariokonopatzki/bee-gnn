import os
import sys
from pathlib import Path
import datetime as dt
import subprocess

import numpy as np

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)


def int_to_onehot_array(i):
    l = []
    for j in range(9):
        if j != i:
            l.append(0)
        else:
            l.append(1)
    return np.array(l)


def increase_datestring(date_string, num_days):
    """Helper function"""
    future_date = string_to_datetime(date_string) + dt.timedelta(num_days)
    future_date = datetime_to_string(future_date)
    return future_date


def string_to_datetime(str_):
    return dt.datetime.strptime(str_, '%Y-%m-%d')


def datetime_to_string(datetime):
    return datetime.strftime('%Y-%m-%d')


def install_additional_packages():
    print('Installing additional packages.')
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           '/kaggle/input/torch-geometric/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl'])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           '/kaggle/input/torch-geometric/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl'])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           '/kaggle/input/torch-geometric/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl'])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           '/kaggle/input/torch-geometric/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl'])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           '/kaggle/input/torch-geometric/torch_geometric-2.0.2-py3-none-any.whl'])
