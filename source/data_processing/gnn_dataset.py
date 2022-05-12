import os
import sys
from pathlib import Path
import h5py

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

PROJECT_NAME = 'bees'
PROJECT_DIR = os.environ.get(f'{PROJECT_NAME.upper()}DIR', '/kaggle/working/')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)
PROJECT_DIR = Path(PROJECT_DIR)

from source.config import INPUT_PATH, OUTPUT_PATH, PHASE_2_DATES, \
    INTERACTIONS_START_DATE
from source.utils import string_to_datetime


class GNNDataset(InMemoryDataset):
    def __init__(self, individuals, targets, phase, transform):
        self.individuals = individuals
        self.targets = targets
        self.individual_transform = transform['individuals']
        self.interactions_transform = transform['interactions']
        self.phase = phase
        with h5py.File(INPUT_PATH / 'interaction_networks_20160729to20160827.h5',
                       "r") as f:
            self.bee_id_2_interaction_id = {bee_id: interaction_id for
                                            interaction_id, bee_id in
                                            enumerate(f['bee_ids'][:])}

        super().__init__(OUTPUT_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.phase}_data.pt']

    def process(self):
        data_list = []

        dates = PHASE_2_DATES[self.phase]
        for date in dates:
            date_interaction_id = (
                    string_to_datetime(date) - INTERACTIONS_START_DATE).days
            date_individuals = self.individuals[self.individuals.date.eq(date)]
            date_targets = self.targets[self.individuals.date.eq(date)]
            with h5py.File(INPUT_PATH / 'interaction_networks_20160729to20160827.h5',
                           "r") as f:
                date_interactions = f['interactions'][date_interaction_id]
            x, y, edge_index, edge_attr = create_data_components_for_day(
                date_individuals,
                date_targets,
                date_interactions,
                self.bee_id_2_interaction_id)

            x = self.individual_transform.transform(x)
            edge_attr = self.interactions_transform.transform(edge_attr)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            edge_index = torch.tensor(edge_index)
            edge_attr = torch.tensor(edge_attr)
            data = Data(x, edge_index, edge_attr, y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_data_components_for_day(date_individuals, date_targets,
                                   date_interactions, bee_id_2_interactions_id):
    date_individuals_sorted = date_individuals.sort_values(axis=0, by='bee_id')
    targets_sorted = date_targets.reindex(date_individuals_sorted.index)

    x = date_individuals_sorted.loc[:,
        ~date_individuals_sorted.columns.isin(['date', 'bee_id'])].values
    y = targets_sorted.values

    # Our graph has one node for every bee alive today. We create node_ids
    # starting at zero, and a mapping interaction_id (index in interactions
    # tensor) -> node_id
    date_bee_ids_sorted = date_individuals_sorted.bee_id.values
    date_interaction_ids_sorted = list([bee_id_2_interactions_id[bee_id] for
                                        bee_id in list(date_bee_ids_sorted)])
    interaction_id_2_node_id = {interaction_id: node_id for
                                node_id, interaction_id in
                                enumerate(date_interaction_ids_sorted)}

    num_date_interactions = date_interactions.shape[0]
    num_edge_features = date_interactions.shape[-1]
    date_interactions_flat = date_interactions.reshape((-1, num_edge_features))

    # We create a grid of all possible pairs of interaction (bee) ids, i.e.
    # [0,0], [0,1] ... [num_interaction_ids, num_interaction_ids], and then
    # keep only the rows where (i) both of the bees corresponding to the
    # interaction_id are alive today, and (ii) any interaction type has a
    # non-zero entry (i.e. at least one interaction took place between the two
    # bees).
    possible_interaction_pairs = np.mgrid[:num_date_interactions,
                                 :num_date_interactions].reshape(2, -1).T
    usable_interaction_ids = list(interaction_id_2_node_id.keys())
    source_is_usable = np.isin(possible_interaction_pairs[:, 0],
                               usable_interaction_ids)
    target_is_usable = np.isin(possible_interaction_pairs[:, 1],
                               usable_interaction_ids)
    is_interaction = np.any(~np.isclose(date_interactions_flat, 0), axis=1)
    is_usable = source_is_usable & target_is_usable & is_interaction
    usable_interaction_pairs = possible_interaction_pairs[is_usable]
    usable_node_pairs = np.vectorize(interaction_id_2_node_id.get)(
        usable_interaction_pairs)
    edge_index = np.transpose(usable_node_pairs)

    # The edge features are just the flattened interaction arrays of shape (9,)
    # which correspond to a usable interaction pair from above.
    edge_attr = date_interactions_flat[is_usable]

    return x, y, edge_index, edge_attr
