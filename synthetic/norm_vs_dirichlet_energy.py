import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


def get_data(name):
    path = '../data/' + name
    dataset = Planetoid(root=path, name=name, split='public')
    return dataset[0]


def dirichlet_energy(x, edge_index, edge_weights):
    edge_difference = edge_weights * (torch.norm(x[edge_index[0]] - x[edge_index[1]],dim=1) ** 2)
    return edge_difference.sum() / 2


class SimpleModel(nn.Module):
    def __init__(self, in_dim, h_dim, num_layers, conv, scale_weights=2.0):
        super().__init__()
        self.enc = nn.Linear(in_dim, h_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if conv == 'GCN':
                layer = pyg.nn.GCNConv(h_dim, h_dim, bias=False, add_self_loops=True)
                if scale_weights != 1:
                    layer.lin.weight.data = layer.lin.weight.data * scale_weights
            elif conv == 'GAT':
                layer = pyg.nn.GATConv(h_dim, h_dim, bias=False, add_self_loops=True)
                if scale_weights != 1:
                    layer.lin_src.weight.data = layer.lin_src.weight.data * scale_weights
            elif conv == 'SAGE':
                layer = pyg.nn.SAGEConv(h_dim, h_dim, bias=False)
                if scale_weights != 1:
                    layer.lin_l.weight.data = layer.lin_l.weight.data * scale_weights
                    layer.lin_r.weight.data = layer.lin_r.weight.data * scale_weights
            self.convs.append(layer)

        self.num_layers = num_layers

    def forward(self, data):
        A = pyg.utils.to_dense_adj(data.edge_index)[0]
        D_inv = torch.diag(1 / torch.sum(A, dim=1)).cuda()
        A_rw = D_inv @ A
        edge_index, edge_weights = pyg.utils.dense_to_sparse(A_rw)

        energies, norms = [], []

        x = self.enc(data.x)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            energies.append(dirichlet_energy(x, edge_index, edge_weights))
            norms.append(torch.norm(x)**2)

        return energies, norms


def run(conv, dataset, num_layers, weight_scale):
    data = get_data(dataset)
    data = pyg.transforms.LargestConnectedComponents()(data)
    data.edge_index = pyg.utils.remove_self_loops(data.edge_index)[0]

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(data.num_node_features, 16, num_layers, conv, weight_scale).to(device)

    data.to(device)
    with torch.no_grad():
        model.eval()
        energies, norms = model(data)

    return {
        'Dirichlet energy': torch.FloatTensor(energies).cpu(),
        'Norm': torch.FloatTensor(norms).cpu(),
    }


def plot_energies(stat_list, postfix):
    plt.figure(figsize=(6, 3))
    for stat in ['Dirichlet energy', 'Norm']:
        plt.set_cmap(plt.get_cmap('viridis'))
        colors = ['#404788', '#238A8D', '#55C667']
        for i in range(len(stat_list)):
            linestyle = '-' if stat == 'Dirichlet energy' else '--'
            plt.plot(np.arange(1, 129), stat_list[i][stat], linestyle, label=f"{stat_list[i]['conv']} ({stat})", c=colors[i])
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(ncol=2)
        plt.xticks([1, 2, 4, 8, 16, 32, 64, 128], [1, 2, 4, 8, 16, 32, 64, 128])
        plt.xlabel('Number of layers')
        plt.ylabel('Dirichlet energy / Norm')
    plt.savefig(f'constant_{postfix}.svg', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    dataset = 'Cora'
    num_layers = 128
    for weight_scale in [1.0, 2.0]:
        torch.manual_seed(10)
        stat_list = []
        for conv in ['SAGE', 'GAT', 'GCN']:
            stats = run(conv, dataset, num_layers, weight_scale)
            stats['conv'] = conv
            stat_list.append(stats)
        plot_energies(stat_list, weight_scale)
