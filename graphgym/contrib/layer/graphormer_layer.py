from functools import wraps
from typing import List

import deepsnap.batch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from einops import rearrange, repeat
from torch import einsum, scatter
from torch_geometric.utils import to_dense_batch, degree
from torch_scatter import scatter_add, scatter_mean

from graphgym.config import cfg
from graphgym.register import register_layer


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class GraphormerAttention(nn.Module):
    def __init__(self, input_dim, output_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x, mask=None, sim_bias=None):
        h = self.heads
        q = self.to_q(x)
        bs, num_tokens, _ = x.shape

        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        global global_counter
        global_counter += 1
        if exists(sim_bias):
            assert sim_bias.shape == (bs, num_tokens, num_tokens, h)
            sim += rearrange(sim_bias, 'b i j h -> (b h) i j',  h=h)
        if exists(mask):
            assert mask.shape == (bs, num_tokens)
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~mask, torch.finfo(sim.dtype).min)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        x = self.to_out(out)
        return x


class GraphormerLayer(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GraphormerLayer, self).__init__()
        assert cfg.dataset.max_graph_size > 0
        get_distance_bias = lambda: nn.Parameter(torch.zeros((cfg.dataset.max_graph_size, cfg.gnn.num_heads)))
        get_distance_bias = cache_fn(get_distance_bias)
        get_edge_projection = lambda: nn.Linear(cfg.dataset.edge_dim, dim_in)
        get_edge_projection = cache_fn(get_edge_projection)
        self.attention_distance_bias = get_distance_bias(_cache=True)
        self.centrality_encoding = nn.Parameter(torch.zeros((cfg.dataset.max_graph_size, dim_in)))
        self.project_edge_feature = get_edge_projection(_cache=True)
        self.attention_module = GraphormerAttention(
            input_dim=dim_in,
            output_dim=dim_in,
            heads=cfg.gnn.num_heads,
            dim_head=cfg.gnn.dim_head,
            dropout=cfg.gnn.dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.PReLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, batch: deepsnap.batch.Batch):
        node_degrees = degree(
            batch.edge_index[0],
            num_nodes=batch.node_feature.shape[0]
        )
        dense_batch, dense_batch_mask = to_dense_batch(
            batch.node_feature + self.centrality_encoding[node_degrees.long().clip(max=cfg.dataset.max_graph_size - 1)],
            batch.batch
        )
        num_graphs, max_num_nodes, _ = dense_batch.shape
        batch.shortest_path: List[np.ndarray]  # tensor of shape
        dist_indices = torch.zeros((num_graphs, max_num_nodes, max_num_nodes), dtype=int)
        for i, dist_matrix in enumerate(batch.shortest_path):
            dist_indices[i, :dist_matrix.shape[0], :dist_matrix.shape[1]] = torch.as_tensor(dist_matrix)
        dist_indices = dist_indices.clip(max=cfg.dataset.max_graph_size - 1)
        sim_bias = self.attention_distance_bias[dist_indices.to(dense_batch.device)]
        dense_batch_predictions = self.attention_module(x=dense_batch, mask=dense_batch_mask, sim_bias=sim_bias)
        edge_feature = self.project_edge_feature(batch.edge_feature)
        edge_out = torch.zeros(batch.node_feature.shape).to(batch.node_feature.device)
        scatter_add(edge_feature, index=batch.edge_index[0], dim=0, out=edge_out)
        batch.node_feature = self.mlp(dense_batch_predictions[dense_batch_mask] +
                                      edge_out / node_degrees.clip(min=1).float().unsqueeze(1) ** 0.5)
        return batch


register_layer('graphormerlayer', GraphormerLayer)
