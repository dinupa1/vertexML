"""
Base class for predicting the vertex properties using GNN.

This model is originally developed in particleflow repository.
Original source: https://github.com/jpata/particleflow/blob/b1cb537b4e89b82048c73a42c750b4c6f4ae1990/mlpf/pyg/model.py

Following changes were made to original code:
    (a). Classification layer was removed in the Net() (no classification is required now)

dinupa3@gmail.com
08-26-2022
"""

from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear

from torch_scatter import scatter

import torch_geometric
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor


class Net(nn.Module):
    """
    GNN model based on GravNet

    Original source: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gravnet_conv.html

    Forward pass returns:
        predictions: tensor of predictions containing a concatenated representation of the vertex position and momentum
        target: tensor containing generated vertex information
    """

    def __init__(self, input_dim=4, output_dim=1, embedding_dim=8, hidden_dim1=10,
                 hidden_dim2=16, num_conv=1, space_dim=4, propagate_dim=8, k=3):

        super(Net, self).__init__()

        self.act = nn.ReLU

        # (1) DNN: embedding
        self.dnn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            self.act(),
            # nn.Linear(hidden_dim1, hidden_dim1),
            # self.act(),
            nn.Linear(hidden_dim1, hidden_dim1),
            self.act(),
            nn.Linear(hidden_dim1, embedding_dim),
        )

        # (2) GravNet: dynamic graph building
        self.conv = nn.ModuleList()
        for i in range(num_conv):
            self.conv.append(GravNetConv_vtx(embedding_dim, embedding_dim, space_dim, propagate_dim, k))

        # (3) DNN: regressing vertex
        self.dnn2 = nn.Sequential(
            nn.Linear(input_dim+embedding_dim, hidden_dim2),
            self.act(),
            # nn.Linear(hidden_dim2, hidden_dim2),
            # self.act(),
            nn.Linear(hidden_dim2, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, output_dim),
        )

    def forward(self, batch):
        # unfold the batch object
        input = batch.x

        # embedding the inputs
        embedding = self.dnn1(input)

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv):
            embedding = conv(embedding, batch.batch)

        # predict the vertex properties
        preds_vtx = self.dnn2(torch.cat([input, embedding], axis=-1))

        return preds_vtx


try:
    from torch_cluster import knn
except ImportError:
    knn = None



class GravNetConv_vtx(MessagePassing):
    """
    Copied from pytorch_geometric source code, with the following edits
        a. used reduce='sum' instead of reduce='mean' in the message passing
        b. removed skip connection

    Original source: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gravnet_conv.html
    """

    def __init__(self, in_channels: int, out_channels: int, space_dimensions: int,
                 propagate_dimensions: int, k: int, num_workers: int = 1, **kwargs):
        super().__init__(flow='source_to_target', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers


        self.lin_p = Linear(in_channels, propagate_dimensions)
        self.lin_s = Linear(in_channels, space_dimensions)
        self.lin_out = Linear(propagate_dimensions, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_p.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, x: Union[Tensor, Tensor], batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        is_bipartite: bool = True
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            is_bipartite = False

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in 'GravNetConv'")

        b: PairTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # embed the inputs before message passing
        msg_activations = self.lin_p(x[0])

        # transform to the space dimension to build the graph
        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1]) if is_bipartite else s_l

        # add error message when trying to preform knn without enough neighbors in the region
        if(torch.unique(b[0], return_counts=True)[1] < self.k).sum() != 0:
            raise RuntimeError(f'Not enough elements in a region to perform the k-nearest neighbors. Current k-value={self.k}')

        edge_index = knn(s_l, s_r, self.k, b[0], b[1]).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_r[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight) # 10 gives better spred

        # message passing
        out = self.propagate(edge_index, x=(msg_activations, None), edge_weight=edge_weight, size=(s_l.size(0), s_r.size(0)))
        return self.lin_out(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j* edge_weight.unsqueeze(1)

    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None) -> Tensor:
        out_mean = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out_mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, k={self.k})"