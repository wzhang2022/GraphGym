from typing import List

import networkx as nx

import numpy as np
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader
from graphgym.config import cfg
from graphgym.register import register_loader


def create_loader_with_node_distances(datasets: List[GraphDataset]):
    for dataset in datasets:
        for graph in dataset.graphs:
            distances = np.ones((graph.num_nodes, graph.num_nodes), dtype=np.int64) * -1
            for i, distance_dict in nx.shortest_path_length(graph.G):
                for j in distance_dict:
                    distances[i][j] = distance_dict[j]
            setattr(graph, 'shortest_path', distances)

    loader_train = DataLoader(datasets[0], collate_fn=Batch.collate(),
                              batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False)
    loaders = [loader_train]
    for i in range(1, len(datasets)):
        loaders.append(DataLoader(datasets[i], collate_fn=Batch.collate(),
                                  batch_size=cfg.train.batch_size,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False))

    return loaders


# register_loader('node_distance_loader', create_loader_with_node_distances)
