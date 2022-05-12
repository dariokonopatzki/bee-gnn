import os
import sys
from pathlib import Path
import datetime as dt

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

# region Most frequently changed run parameters. -------------------------------
MODEL_NAME = 'MLP'  # 'MLP' or 'GNN'
NUM_TRIALS = 1  # Number of Optuna trials for hyperparameter tuning.
EARLY_STOPPING_PATIENCE = 10
MAX_NUM_EPOCHS = 100
# Predict whether a bee is still alive on current_date+NUM_PREDICTION_TIME_STEPS.
# Should be larger than 0; tested 7.
NUM_PREDICTION_TIME_STEPS = 7
# -------------------------------------------------------------------------------

INPUT_PATH = Path('/kaggle/input/honey-bee-temporal-social-network/')
OUTPUT_PATH = Path('/kaggle/working/')

SEEDS = {2022, 5, 10}

IS_ON_KAGGLE = (PROJECT_DIR == Path(f'/kaggle/working/'))

LOCATIONS = ['dance_floor', 'honey_storage', 'near_exit', 'brood_area_total']
TARGET_NAME = ['alive_in_future']

INTERACTIONS_START_DATE = dt.datetime.strptime('2016-07-29', '%Y-%m-%d')
INDIVIDUALS_DATA_START_DATE = dt.datetime.strptime('2016-08-01', '%Y-%m-%d')

INPUT_FEATURE_NAMES = ['age'] + LOCATIONS

NUM_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)
NUM_EDGE_FEATURES = 9

PHASE_2_DATES = {
    'train': [
        (INDIVIDUALS_DATA_START_DATE + dt.timedelta(td)).strftime('%Y-%m-%d')
        for td in range(12)],
    'valid': [
        (INDIVIDUALS_DATA_START_DATE + dt.timedelta(td)).strftime('%Y-%m-%d')
        for td in range(12, 15)],
    'test': [
        (INDIVIDUALS_DATA_START_DATE + dt.timedelta(td)).strftime('%Y-%m-%d')
        for td in range(15, 18)]
}
