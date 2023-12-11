from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter

from torch_geometric.nn.inits import glorot, zeros, ones
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)

class SKP(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        conv='SKP',
        num_edges=None,
        **kwargs,
    ):
        if conv in ['softmax_SKP']:
            kwargs.setdefault('aggr', 'sum')
        else:
            kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.conv = conv

        self.lin_l = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.lin_r = self.lin_l

        self.edge_weights = Parameter(torch.Tensor(num_edges, self.heads))
        ones(self.edge_weights)    

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        ones(self.edge_weights)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj):
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = x_l

        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=None, size=None)

        if self.conv == 'softmax_SKP':
            out = out.mean(dim=1)
        else:
            out = out.sum(dim=1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.conv in ['SKP', 'KP'] :
            alpha = self.edge_weights
        elif self.conv == 'softmax_SKP':
            alpha = softmax(self.edge_weights, index, ptr, size_i)
        
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')