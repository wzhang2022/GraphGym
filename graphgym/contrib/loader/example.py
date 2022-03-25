import os
import pickle

import networkx as nx
import numpy as np
from deepsnap.dataset import GraphDataset
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.cluster import SpectralClustering
from torch_geometric.datasets import *
from tqdm import tqdm

from graphgym.config import cfg
from graphgym.register import register_loader


def load_dataset_example(format, name, dataset_dir):
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            graphs = GraphDataset.pyg_to_graphs(dataset_raw)
            return graphs


register_loader('example', load_dataset_example)
def load_pyg_dataset_with_spectral_clusters(format, name, dataset_dir):
    if cfg.dataset.name == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name=cfg.dataset.name)
        graphs = GraphDataset.pyg_to_graphs(dataset)
        split_idx = dataset.get_idx_split()
        dir_name = 'ogbg_molhiv'
    else:
        raise ValueError('Unknown dataset format: {}'.format(cfg.dataset.name))
    new_dir_name = f'{dataset_dir}/{dir_name}/cluster_data'
    if format == 'spectral_cluster_dataset':
        if os.path.isdir(new_dir_name):
            dataset_dir = new_dir_name
            with open(f"{dataset_dir}/clusters={cfg.gnn.n_clusters}.pkl", 'rb+') as f:
                cluster_list = pickle.load(f)
        else:
            dataset_dir = new_dir_name
            cluster_list = []
            for _, graph in enumerate(tqdm(graphs)):
                clustering_algo = SpectralClustering(n_clusters=cfg.gnn.n_clusters, affinity='precomputed')
                if graph.G.number_of_nodes() < cfg.gnn.n_clusters:
                    cluster_list.append(np.arange(graph.G.number_of_nodes()))
                else:
                    clusters = clustering_algo.fit(nx.to_numpy_array(graph.G))
                    cluster_list.append(clusters.labels_)

            os.mkdir(dataset_dir)
            with open(f"{dataset_dir}/clusters={cfg.gnn.n_clusters}.pkl", 'wb') as f:
                pickle.dump(cluster_list, f)
        for cluster_labels, graph in zip(cluster_list, graphs):
            setattr(graph, 'cluster_labels', cluster_labels)
        return graphs, split_idx


register_loader('cluster_loader', load_pyg_dataset_with_spectral_clusters)
