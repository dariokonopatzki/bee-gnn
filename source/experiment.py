import os
import sys
import warnings
from pathlib import Path
import logging
import shutil
import tarfile
import subprocess
import numpy as np

import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, \
    ModelCheckpoint
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from matplotlib import pyplot as plt

# region Boilerplate code needed for imports to work properly across platforms.
PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)
if PROJECT_DIR == Path('/kaggle/working/'):
    sourcecode_dir = Path(f'/kaggle/input/{PROJECT_NAME}-source/')
    new_sourcecode_dir = Path('/kaggle/working/source/')
    shutil.copytree(sourcecode_dir, new_sourcecode_dir)
    for filename in new_sourcecode_dir.iterdir():
        if tarfile.is_tarfile(new_sourcecode_dir / filename):
            with tarfile.open(new_sourcecode_dir / filename) as f:
                f.extractall(new_sourcecode_dir / filename.stem)
    os.makedirs(PROJECT_DIR / 'data', exist_ok=True)
    os.symlink(Path('/kaggle/input/honey-bee-temporal-social-network/'),
               PROJECT_DIR / 'data/input',
               target_is_directory=True)
    # Install additional packages.
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'swifter'])
# endregion

from source.config import MODEL_NAME, SEEDS, NUM_INPUT_FEATURES, \
    EARLY_STOPPING_PATIENCE, NUM_TRIALS, MAX_NUM_EPOCHS, OUTPUT_PATH, \
    IS_ON_KAGGLE
from source.utils import install_additional_packages
from source.data_processing.loading import load_data
from source.best_hyperparameters import best_hyperparameters

if MODEL_NAME == 'GNN':
    if IS_ON_KAGGLE:
        install_additional_packages()
    from torch_geometric.loader import DataLoader
    from source.data_processing.gnn_dataset import GNNDataset as Dataset
    from source.models.gnn import GNN as MODEL, sample_hyperparameters
else:
    from torch.utils.data import DataLoader
    from source.data_processing.mlp_dataset import MLPDataset as Dataset
    from source.models.mlp import MLP as MODEL, sample_hyperparameters

warnings.filterwarnings('ignore', '.*does not have many workers.*')
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)


def objective(trial):
    hyperparameters = sample_hyperparameters(trial)
    return objective_(hyperparameters)


def objective_(hyperparameters):
    seed_losses = []
    for seed in SEEDS:
        seed_everything(seed)
        phase_2_dataloader = dict()
        for phase in ['train', 'valid']:
            shuffle = phase == 'train'
            phase_2_dataloader[phase] = DataLoader(dataset=phase_2_dataset[
                phase], batch_size=hyperparameters['bs'], shuffle=shuffle)
        trainer, model = get_fitted_trainer(hyperparameters,
                                            phase_2_dataloader['train'],
                                            phase_2_dataloader['valid'])
        seed_losses.append(model.best_validation_loss)
    best_mean_validation_loss = sum(seed_losses) / len(seed_losses)
    return best_mean_validation_loss


def get_fitted_trainer(hyperparameters, train_loader, valid_loader,
                       return_best_checkpoint=False):
    model = MODEL(hyperparameters=hyperparameters,
                  num_input_features=NUM_INPUT_FEATURES)
    callbacks = [EarlyStopping(monitor="valid_loss",
                               mode="min",
                               min_delta=0.01,
                               patience=EARLY_STOPPING_PATIENCE),
                 TQDMProgressBar(refresh_rate=0)]
    if return_best_checkpoint:
        model_checkpoint_path = OUTPUT_PATH / 'checkpoint/best'
        if os.path.exists(model_checkpoint_path):
            shutil.rmtree(model_checkpoint_path)
        callbacks.append(ModelCheckpoint(dirpath=OUTPUT_PATH / 'checkpoint/best',
                                         monitor='valid_loss'))
    trainer = pl.Trainer(max_epochs=MAX_NUM_EPOCHS,
                         callbacks=callbacks,
                         accelerator='auto',
                         devices='auto',
                         log_every_n_steps=1,
                         enable_model_summary=False,
                         enable_checkpointing=return_best_checkpoint,
                         logger=False)
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
    return trainer, model


def detailed_objective(hyperparameters):
    seed_losses = []
    y_predictions = []
    for seed in SEEDS:
        seed_everything(seed)
        phase_2_dataloader = dict()
        for phase in ['train', 'valid', 'test']:
            shuffle = phase == 'train'
            phase_2_dataloader[phase] = DataLoader(dataset=phase_2_dataset[
                phase], batch_size=hyperparameters['bs'], shuffle=shuffle)
        trainer, model = get_fitted_trainer(hyperparameters,
                                            phase_2_dataloader['train'],
                                            phase_2_dataloader['valid'],
                                            return_best_checkpoint=True)
        seed_losses.append(model.best_validation_loss)

        predictions = trainer.predict(model,
                                      phase_2_dataloader['test'],
                                      ckpt_path='best')
        y_predictions.append(torch.cat(predictions).numpy())
    mean_validation_loss = sum(seed_losses) / len(seed_losses)
    y_pred = np.mean(y_predictions, axis=0)
    print(f'Valid loss: {mean_validation_loss}')

    if MODEL_NAME == 'GNN':
        y_true = torch.cat([batch.y for batch in
                            phase_2_dataloader['test']]).numpy()
    else:
        y_true = torch.cat([batch[1] for batch in
                            phase_2_dataloader['test']]).numpy()

    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.savefig(OUTPUT_PATH / MODEL_NAME)
    plt.show()
    print(f'ROC AUC: {roc_auc_score(y_true, y_pred)}')


if __name__ == '__main__':
    seed_everything(42)

    # Delete existing csv in case config.TARGET_NAME or
    # config.NUM_PREDICTION_TIME_STEPS changed from last run.
    individuals_data_path = OUTPUT_PATH / 'daily_data_with_targets.csv'
    if os.path.exists(individuals_data_path):
        os.remove(individuals_data_path)

    phase_2_data, transform = load_data()
    phase_2_dataset = {
        phase: Dataset(**phase_2_data[phase], transform=transform) for phase in
        ['train', 'valid', 'test']}

    if best_hyperparameters is None:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=NUM_TRIALS)
        best_hyperparameters = sample_hyperparameters(study.best_trial)
    else:
        best_hyperparameters = best_hyperparameters[MODEL.__name__]
    detailed_objective(best_hyperparameters)
