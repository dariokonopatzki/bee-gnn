import os
import sys
import pandas as pd
import h5py
import swifter
from pathlib import Path

from sklearn.preprocessing import StandardScaler

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

from source.config import MODEL_NAME, PHASE_2_DATES, INPUT_FEATURE_NAMES, \
    LOCATIONS, TARGET_NAME, NUM_PREDICTION_TIME_STEPS, INPUT_PATH, \
    INTERACTIONS_START_DATE, OUTPUT_PATH
from source.utils import string_to_datetime, increase_datestring


def load_data():
    phase_2_data, transform = get_phase_2_individuals()
    if MODEL_NAME == 'GNN':
        for phase in ['train', 'valid', 'test']:
            phase_2_data[phase]['phase'] = phase
            if phase == 'train':
                dates = phase_2_data[phase]['individuals'].date.unique()
                interaction_ids = sorted([
                    (string_to_datetime(date) - INTERACTIONS_START_DATE).days
                    for date in dates])
                with h5py.File(INPUT_PATH / 'interaction_networks_20160729to20160827.h5',
                               "r") as f:
                    interactions = f['interactions'][interaction_ids]
                interactions = interactions.reshape(-1, interactions.shape[-1])
                transform['interactions'] = StandardScaler().fit(interactions)
    return phase_2_data, transform


def get_phase_2_individuals():
    individuals, targets = get_individuals()

    phase_2_data = dict()
    for phase in ['train', 'valid', 'test']:
        dates = PHASE_2_DATES[phase]
        individuals_ = individuals[individuals.date.isin(dates)]
        targets_ = targets.loc[individuals_.index]
        phase_2_data[phase] = {
            'individuals': individuals_, 'targets': targets_
        }
        if phase == 'train':
            individuals_ = individuals_[INPUT_FEATURE_NAMES].values
            transform = {'individuals': StandardScaler().fit(individuals_)}
    return phase_2_data, transform


def get_individuals():
    individuals_data_path = OUTPUT_PATH / 'daily_data_with_targets.csv'
    if not os.path.isfile(individuals_data_path):
        individuals_data_with_targets = get_individuals_with_targets()
        individuals_data_with_targets.to_csv(individuals_data_path, index=False)
    else:
        individuals_data_with_targets = pd.read_csv(individuals_data_path)

    targets = individuals_data_with_targets[TARGET_NAME]
    individuals_data = individuals_data_with_targets.drop(TARGET_NAME, axis=1)
    return individuals_data, targets


def get_individuals_with_targets():
    individuals_data = pd.read_csv(INPUT_PATH / 'bee_daily_data.csv',
                                   usecols=['bee_id', 'date',
                                            'age'] + LOCATIONS)
    targets = individuals_data.swifter.progress_bar(False).apply(
        row_to_alive_in_future,
        axis=1,
        args=(individuals_data,),
        result_type='expand')
    individuals_data_with_targets = pd.concat([individuals_data, targets],
                                              axis=1)
    individuals_data_with_targets = individuals_data_with_targets.dropna()
    individuals_data_with_targets = individuals_data_with_targets.reset_index(
        drop=True)
    return individuals_data_with_targets


def row_to_alive_in_future(row, individual_data):
    bee_df = individual_data[individual_data.bee_id.eq(row.bee_id)]

    future_date = increase_datestring(date_string=row.date,
                                      num_days=NUM_PREDICTION_TIME_STEPS)
    bee_alive_on_future_date = (future_date in bee_df.date.values)
    d = {'alive_in_future': int(bee_alive_on_future_date)}
    return d
