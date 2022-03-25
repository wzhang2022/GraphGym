import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.utils import degree

from graphgym.register import register_layer


class GCNClusteredConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, **kwargs):
        super(GCNClusteredConv, self).__init__()
        self.lin = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        num_nodes, node_feat_dim = batch.node_feature.shape
        device = batch.node_feature.device

        # get the batch cluster labels into single tensor of shape (num_clusters,)
        new_cluster_labels = np.concatenate(batch.cluster_labels)
        index_start = 0
        label_shift = 0
        for cluster_labels in batch.cluster_labels:
            new_cluster_labels[index_start: index_start + len(cluster_labels)] += label_shift
            index_start = index_start + len(cluster_labels)
            label_shift = np.max(new_cluster_labels[:index_start]) + 1
        num_clusters = np.max(new_cluster_labels) + 1
        batch_cluster_labels = torch.as_tensor(new_cluster_labels).to(device)

        # aggregate the transformed node features by cluster label
        # node_degrees = degree(batch.edge_index[0], num_nodes=num_nodes)
        transformed_node_feature = self.lin(batch.node_feature)
        cluster_embeddings = torch_scatter.scatter_mean(
            src=transformed_node_feature,
            index=batch_cluster_labels.long(),
            dim=0
        )

        # assert cluster_embeddings.shape == (num_clusters, transformed_node_feature.shape[1])

        # message pass from clusters to node neighbors, weighted by repetitions of edge
        # edge_indices_i = batch_cluster_labels[batch.edge_index[0]].unsqueeze(1).repeat((1, node_feat_dim))
        cluster_to_node_edges = torch.stack([batch_cluster_labels[batch.edge_index[0]], batch.edge_index[1]], dim=0)
        unique_edges, unique_edge_idx, edge_counts = torch.unique(
            cluster_to_node_edges, dim=1, return_inverse=True, return_counts=True
        )
        messages = torch.gather(cluster_embeddings, dim=0, index=unique_edges[0].long())
        # assert messages.shape == (unique_edges.shape[0], transformed_node_feature.shape[1])
        messages *= edge_counts.unsqueeze(1).float()

        batch.node_feature = torch_scatter.scatter_add(
            src=messages, index=unique_edges[1], dim=0, dim_size=num_nodes
        ) / degree(batch.edge_index[1], num_nodes=num_nodes, dtype=float)
        # assert batch.node_feature.shape == (num_nodes, transformed_node_feature.shape[1])
        return batch


register_layer('clustergcn', GCNClusteredConv)
