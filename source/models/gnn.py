import os
import sys
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

from torch_geometric.nn import NNConv

from source.config import TARGET_NAME


class GNN(pl.LightningModule):
    def __init__(self, hyperparameters, num_input_features):
        super().__init__()
        self.__dict__.update(hyperparameters)

        self.best_validation_loss = math.inf
        self.num_node_features = num_input_features

        # Define the graph convolutional layers.
        gcn_layers = []
        edge_nn_0 = nn.Sequential(nn.Linear(in_features=9,
                                            out_features=self.num_hidden_edge_units),
                                  nn.ReLU(),
                                  nn.Linear(in_features=self.num_hidden_edge_units,
                                            out_features=self.num_node_features * self.num_hidden_gcn_units))
        gcn_layers.append(NNConv(in_channels=self.num_node_features,
                                 out_channels=self.num_hidden_gcn_units,
                                 nn=edge_nn_0,
                                 aggr='mean'))

        for _ in range(self.num_hidden_gcn_layers):
            edge_nn = nn.Sequential(nn.Linear(in_features=9,
                                              out_features=self.num_hidden_edge_units),
                                    nn.ReLU(),
                                    nn.Linear(in_features=self.num_hidden_edge_units,
                                              out_features=self.num_hidden_gcn_units * self.num_hidden_gcn_units))
            gcn_layers.append(NNConv(in_channels=self.num_hidden_gcn_units,
                                     out_channels=self.num_hidden_gcn_units,
                                     nn=edge_nn,
                                     aggr='mean'))
        self.gcn_layers = torch.nn.ModuleList(gcn_layers)
        self.dropout_graph = nn.Dropout(p=self.dropout_probability_gcn)

        # Define the dense layers.
        dense_layers = []
        dense_layers.append(nn.Linear(self.num_hidden_gcn_units,
                                      self.num_hidden_dense_units))
        for _ in range(self.num_hidden_dense_layers):
            dense_layers.append(nn.Linear(in_features=self.num_hidden_dense_units,
                                          out_features=self.num_hidden_dense_units))
        self.dense_layers = nn.ModuleList(dense_layers)
        self.out = nn.Linear(self.num_hidden_dense_units, len(TARGET_NAME))
        self.dropout_dense = nn.Dropout(p=self.dropout_probability_dense)

    def forward(self, data):
        """A single pass through the network. Returns logit."""
        batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index, edge_attr)
            x = self.dropout_graph(x)
            x = F.relu(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.dropout_dense(x)
            x = F.relu(x)
        x = self.out(x)

        return x

    def step(self, batch, phase):
        """forward() pass with loss calculation and logging."""
        target = batch.y
        output = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log(f'{phase}_loss',
                 value=loss.item(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch.num_graphs)
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
        prediction = self.forward(batch)
        y_probs = torch.sigmoid(prediction)
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
    hyperparameters['bs'] = trial.suggest_categorical('bs', [1])
    hyperparameters['num_hidden_edge_units'] = trial.suggest_int(
        'num_hidden_edge_units',
        4,
        16)
    hyperparameters['num_hidden_gcn_layers'] = trial.suggest_int(
        'num_hidden_gcn_layers',
        1,
        2)
    hyperparameters['num_hidden_gcn_units'] = trial.suggest_int(
        'num_hidden_gcn_units',
        4,
        16)
    hyperparameters['dropout_probability_gcn'] = trial.suggest_float(
        'dropout_probability_gcn',
        0.0,
        0.75)
    hyperparameters['num_hidden_dense_layers'] = trial.suggest_int(
        'num_hidden_dense_layers',
        1,
        3)
    hyperparameters['num_hidden_dense_units'] = trial.suggest_int(
        'num_hidden_dense_units',
        4,
        64)
    hyperparameters['dropout_probability_dense'] = trial.suggest_float(
        'dropout_probability_dense',
        0.0,
        0.75)
    return hyperparameters
