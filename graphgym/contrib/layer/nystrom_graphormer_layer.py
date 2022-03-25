import math
from functools import wraps
from typing import List

import deepsnap.batch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from einops import rearrange, repeat, reduce
from torch import einsum, scatter
from torch_geometric.utils import to_dense_batch, degree
from torch_scatter import scatter_add, scatter_mean

from graphgym.config import cfg
from graphgym.contrib.layer.graphormer_layer import exists, cache_fn, default
from graphgym.register import register_layer


class NystromAttention(nn.Module):
    # TODO: enable arbitrary masking, thus allowing autoregressive models

    def __init__(
            self,
            input_dim,
            output_dim,
            heads,
            head_dim,
            landmarks,
            dropout
    ):
        super().__init__()

        self.scale = head_dim ** -0.5
        self.heads = heads

        inner_dim = head_dim * heads
        self.num_landmarks = landmarks
        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x, context=None, mask=None):
        """
        :param x: Tensor of shape (bs, num_q_tokens, num_heads * head_dim)
        :param context: Tensor of shape (bs, num_kv_tokens, num_heads * head_dim)
        :param mask: Tensor of shape (bs, num_q_tokens, num_kv_tokens). Each sample is rectangular; i.e. for a sample
        with n queries and m kv-pairs, the upper left n x m block is all True, everywhere else is False
        :return:
        """
        context = default(context, x)
        # get x and context masks
        q_mask = mask.any(dim=2)        # shape (bs, num_q)
        kv_mask = mask.any(dim=1)       # shape (bs, num_kv)

        # pad number of input tokens if not divisible
        assert x.shape[1] % self.num_landmarks == 0
        assert context.shape[1] % self.num_landmarks == 0
        q = self.to_q(x) * (self.scale ** 0.5) * q_mask.unsqueeze(2)
        k, v = (self.to_kv(context) * (self.scale ** 0.5) * kv_mask.unsqueeze(2)).chunk(2, dim=-1)
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))          # compute for each head
        q_landmarks, k_landmarks = map(lambda t: reduce(t, 'b (l n) d -> b l d', "mean", l=self.num_landmarks), (q, k))
        kernel_1 = F.softmax(einsum("b i d, b j d -> b i j", q, k_landmarks), dim=-1)            # (bs * h, num_q, l)
        kernel_2 = F.softmax(einsum("b i d, b j d -> b i j", q_landmarks, k_landmarks), dim=-1)  # (bs * h, l, l)
        sim_ql_k = einsum("b i d, b j d -> b i j", q_landmarks, k)                               # (bs * h, l, num_k)
        softmax_mask = repeat(kv_mask, "b n -> (b h) () n", h=h)
        kernel_3 = F.softmax(sim_ql_k - softmax_mask * torch.finfo(sim_ql_k.dtype).max, dim=-1)

        # multiply: kernel_1 * pinverse(kernel_2) * kernel_3 * v
        x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        x = rearrange(x, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(x)

    def iterative_inv(self, mat, n_iter=8):
        """
        :param mat: tensor of shape (bs, l, l)
        :param n_iter: number of iterations for approximation
        :return: pseudoinverse approximation of mat
        """
        id = torch.eye(mat.size(-1), device=mat.device)
        k = mat
        scale_1 = 1 / torch.max(torch.sum(torch.abs(k), dim=-2), dim=-1)[0]    # shape: (bs,)
        scale_2 = 1 / torch.max(torch.sum(torch.abs(k), dim=-1), dim=-1)[0]    # shape: (bs,)
        v = k.transpose(-1, -2) * scale_1[:, None, None] * scale_2[:, None, None]
        for _ in range(n_iter):
            kv = torch.matmul(k, v)
            v = torch.matmul(0.25 * v, 13 * id - torch.matmul(kv, 15 * id - torch.matmul(kv, 7 * id - kv)))
        return v


class NystromGraphormerLayer(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(NystromGraphormerLayer, self).__init__()
        assert cfg.dataset.max_graph_size > 0
        get_distance_bias = lambda: nn.Parameter(torch.zeros((cfg.dataset.max_graph_size, cfg.gnn.num_heads)))
        get_distance_bias = cache_fn(get_distance_bias)
        get_edge_projection = lambda: nn.Linear(cfg.dataset.edge_dim, dim_in)
        get_edge_projection = cache_fn(get_edge_projection)
        self.attention_distance_bias = get_distance_bias(_cache=True)
        self.centrality_encoding = nn.Parameter(torch.zeros((cfg.dataset.max_graph_size, dim_in)))
        self.project_edge_feature = get_edge_projection(_cache=True)
        self.attention_module = NystromAttention(
            input_dim=dim_in,
            output_dim=dim_in,
            heads=cfg.gnn.num_heads,
            head_dim=cfg.gnn.dim_head,
            dropout=cfg.gnn.dropout,
            landmarks = cfg.gnn.n_clusters
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.PReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.landmarks = cfg.gnn.n_clusters

    def forward(self, batch: deepsnap.batch.Batch):
        node_degrees = degree(
            batch.edge_index[0],
            num_nodes=batch.node_feature.shape[0]
        )
        dense_batch, dense_batch_mask = to_dense_batch(
            batch.node_feature + self.centrality_encoding[node_degrees.long().clip(max=cfg.dataset.max_graph_size - 1)],
            batch.batch
        )
        num_graphs, max_num_nodes, node_feat_dim = dense_batch.shape

        padded_num_tokens = math.ceil(max_num_nodes / self.landmarks) * self.landmarks
        dense_padded_batch = torch.zeros(
            (num_graphs, padded_num_tokens, node_feat_dim), dtype=dense_batch.dtype, device=dense_batch.device
        )
        dense_padded_batch_mask = torch.zeros(
            (num_graphs, padded_num_tokens), dtype=bool, device=dense_batch_mask.device
        )
        dense_padded_batch[:, :max_num_nodes, :] = dense_batch
        dense_padded_batch_mask[:, :max_num_nodes] = dense_batch_mask

        batch.shortest_path: List[np.ndarray]  # tensor of shape
        dist_indices = torch.zeros((num_graphs, padded_num_tokens, padded_num_tokens), dtype=int)
        for i, dist_matrix in enumerate(batch.shortest_path):
            dist_indices[i, :dist_matrix.shape[0], :dist_matrix.shape[1]] = torch.as_tensor(dist_matrix)
        dist_indices = dist_indices.clip(max=cfg.dataset.max_graph_size - 1)
        # sim_bias = self.attention_distance_bias[dist_indices.to(dense_batch.device)]
        mask = torch.einsum("bi,bj->bij", dense_padded_batch_mask, dense_padded_batch_mask)
        dense_batch_predictions = self.attention_module(x=dense_padded_batch, mask=mask)
        edge_feature = self.project_edge_feature(batch.edge_feature)
        edge_out = torch.zeros(batch.node_feature.shape).to(batch.node_feature.device)
        scatter_add(edge_feature, index=batch.edge_index[0], dim=0, out=edge_out)
        batch.node_feature = self.mlp(dense_batch_predictions[dense_padded_batch_mask] +
                                      edge_out / node_degrees.clip(min=1).float().unsqueeze(1) ** 0.5)
        return batch


register_layer('nystromgraphormerlayer', NystromGraphormerLayer)
