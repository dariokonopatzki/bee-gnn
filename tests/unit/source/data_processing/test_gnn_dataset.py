import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

from source.config import INTERACTIONS_START_DATE
from source.utils import string_to_datetime
from source.data_processing.gnn_dataset import create_data_components_for_day, \
    GNNDataset


def test_create_data_components():
    data_with_targets = {
        0: ['2016-07-29', 1, 1, 1],
        1: ['2016-07-29', 2, 30, 0],
        2: ['2016-07-29', 1, 2, 1],
        3: ['2016-07-30', 1, 2, 1],
        4: ['2016-07-30', 2, 31, 0],
        5: ['2016-07-30', 4, 0, 1],
    }

    date = '2016-07-30'
    data_with_targets = pd.DataFrame.from_dict(data_with_targets,
                                               orient='index',
                                               columns=['date', 'bee_id', 'age',
                                                        'alive_in_future'])
    interactions_1__0 = np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0]])
    interactions_1__1 = np.array([[0, 0, 0], [0, 0, 4], [0, 4, 0]])
    interactions_0 = np.zeros((3, 3, 2))
    interactions_1 = np.stack([interactions_1__0, interactions_1__1], axis=-1)
    interactions = np.stack([interactions_0, interactions_1], axis=0)
    bee_id_2_bee_id_in_interactions = {1: 2, 2: 1, 4: 0}
    x_true = np.array([[2], [31], [0]])
    y_true = np.array([[1], [0], [1]])
    edge_index_true = np.array([[1, 0], [0, 1]])  # ordered by interaction_id
    edge_attr_true = np.array([[2, 4], [2, 4]])

    date_data_with_targets = data_with_targets[data_with_targets.date.eq(date)]
    date_interactions_id = (
            string_to_datetime(date) - INTERACTIONS_START_DATE).days
    date_interactions = interactions[date_interactions_id]

    x, y, edge_index, edge_attr = create_data_components_for_day(
        date_individuals=date_data_with_targets.loc[:,
                         date_data_with_targets.columns != 'alive_in_future'],
        date_targets=date_data_with_targets[['alive_in_future']],
        date_interactions=date_interactions,
        bee_id_2_interactions_id=bee_id_2_bee_id_in_interactions)
    assert np.array_equal(x, x_true)
    assert np.array_equal(y, y_true)
    assert np.array_equal(edge_index, edge_index_true)
    assert np.array_equal(edge_attr, edge_attr_true)
