import os
import sys
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

from source.config import TARGET_NAME


class MLP(pl.LightningModule):
    def __init__(self, hyperparameters, num_input_features):
        super().__init__()
        self.__dict__.update(hyperparameters)
        self.best_validation_loss = math.inf
        dense_layers = [torch.nn.Linear(in_features=num_input_features,
                                        out_features=self.num_hidden_units)]
        for _ in range(self.num_hidden_layers - 1):
            dense_layers.append(torch.nn.Linear(in_features=self.num_hidden_units,
                                                out_features=self.num_hidden_units))
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        self.dropout = torch.nn.Dropout(p=self.dropout_probability)
        self.out = torch.nn.Linear(in_features=self.num_hidden_units,
                                   out_features=len(TARGET_NAME))

    def forward(self, x):
        """A single pass through the network. Returns logit."""
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.out(x)
        return x

    def step(self, batch, phase):
        """forward() pass with loss calculation and logging."""
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        self.log(f'{phase}_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    # One step function per forward pass, calling step() with the phase's name
    # for correct logging.
    def training_step(self, batch):
        loss = self.step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, 'valid')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, 'test')
        return loss

    def predict_step(self, batch, batch_idx):
        """forward() pass but returning probability rather than logit."""
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        y_probs = torch.sigmoid(y_hat)
        return y_probs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def validation_epoch_end(self, outputs):
        """
        At epoch end, calculate current validation_loss and log it if it is
        minimal.
        """
        valid_loss = (sum(outputs) / len(outputs)).item()
        if valid_loss < self.best_validation_loss:
            self.best_validation_loss = valid_loss


def sample_hyperparameters(trial):
    hyperparameters = dict()

    hyperparameters['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    hyperparameters['bs'] = trial.suggest_categorical('bs',
                                                      [256, 512, 1024, 2048])
    hyperparameters[
        'num_hidden_layers'] = trial.suggest_int('num_hidden_layers', 1, 4)
    hyperparameters['num_hidden_units'] = trial.suggest_int('num_hidden_units',
                                                            4,
                                                            128)
    hyperparameters['dropout_probability'] = trial.suggest_float(
        'dropout_probability',
        0.0,
        0.75)
    return hyperparameters
